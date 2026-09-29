"""Per-sample GC-bias estimation (Griffin-compatible output).

Produces, for each input alignment file, a table with columns
``length``, ``num_GC`` and ``smoothed_GC_bias`` -- the same format consumed by
``cfstats siteprofile`` (and by Griffin's ``griffin_coverage.py``).

The GC bias for a fragment length ``L`` and GC count ``g`` is

    bias(L, g) = observed_fragments(L, g) / expected_fragments(L, g)

normalised to a mean of 1 per length and median-smoothed across neighbouring
lengths. ``expected_fragments`` is estimated by sampling random genomic windows
from the reference (self-contained; no precomputed genome GC-frequency files
required).
"""

import os
import sys
import random
import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pysam

from cfstats.utils import collect_bam_files

log = logging.getLogger(__name__)

GC_SMOOTHING_STEP = 20


def _gc_count(seq):
    seq = seq.upper()
    return seq.count("G") + seq.count("C") + seq.count("S")


def _expected_gc_frequency(reference, size_range, n_samples, chroms, seed):
    """Estimate the expected per-length GC-count distribution from the reference."""
    fasta = pysam.FastaFile(reference)
    rng = random.Random(seed)

    refs = [c for c in fasta.references if (chroms is None or c in set(chroms))]
    reflen = {c: fasta.get_reference_length(c) for c in refs}
    max_len = size_range[1]
    refs = [c for c in refs if reflen[c] > max_len + 1]
    if not refs:
        fasta.close()
        raise ValueError("No reference contigs long enough for GC sampling.")

    weights = [reflen[c] for c in refs]
    lengths = range(size_range[0], size_range[1] + 1)
    freq = {L: np.zeros(L + 1) for L in lengths}

    for _ in range(n_samples):
        chrom = rng.choices(refs, weights=weights)[0]
        pos = rng.randint(0, reflen[chrom] - max_len - 1)
        seq = fasta.fetch(chrom, pos, pos + max_len).upper()
        if "N" in seq:
            continue
        gc_cumsum = np.cumsum([1 if b in "GCS" else 0 for b in seq])
        for L in lengths:
            freq[L][int(gc_cumsum[L - 1])] += 1

    fasta.close()
    return freq


def _worker_observed_counts(pl):
    """Count observed fragments by (length, num_GC) for a single alignment file."""
    samfile, args = pl
    lo, hi = args.size_range

    cram = pysam.AlignmentFile(samfile, reference_filename=args.reference)
    fasta = pysam.FastaFile(args.reference)

    counts = {L: np.zeros(L + 1) for L in range(lo, hi + 1)}

    for read in cram.fetch():
        if args.reqflag is not None and (read.flag & args.reqflag) != args.reqflag:
            continue
        if args.exclflag and (read.flag & args.exclflag) != 0:
            continue
        if read.mapping_quality < args.mapqual:
            continue
        # Use the forward read of each pair (positive template length) to avoid
        # double counting fragments.
        tl = read.template_length
        if read.is_reverse or tl <= 0:
            continue
        if tl < lo or tl > hi:
            continue

        frag_start = read.reference_start
        frag_end = frag_start + tl
        seq = fasta.fetch(read.reference_name, frag_start, frag_end)
        if "N" in seq.upper():
            continue
        num_gc = _gc_count(seq)
        counts[tl][num_gc] += 1

    cram.close()
    fasta.close()
    return {"samfile": samfile, "counts": counts}


def _median_smoothing(values, fraction=0.05):
    """Griffin-style sliding-window median smoothing over an ordered series."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    bin_size = max(int(n * fraction), 50)
    bin_size = min(bin_size, n)
    out = np.empty(n)
    for i in range(n):
        start = int(i - bin_size / 2)
        end = int(i + bin_size / 2)
        if start < 0:
            start, end = 0, bin_size
        if end > n:
            start, end = n - bin_size, n
        window = values[start:end]
        if np.all(np.isnan(window)):
            out[i] = np.nan
        else:
            out[i] = np.nanmedian(window)
    return out


def _build_bias_table(observed, expected, size_range):
    """Combine observed and expected counts into a smoothed GC-bias table."""
    rows = []
    for L in range(size_range[0], size_range[1] + 1):
        obs = observed[L]
        exp = expected[L]
        for g in range(L + 1):
            rows.append((L, g, g / L, obs[g], exp[g]))

    df = pd.DataFrame(rows, columns=["length", "num_GC", "GC_content",
                                     "number_of_fragments", "number_of_positions"])

    # Raw bias per length, normalised to mean 1.
    df["GC_bias"] = np.nan
    for L, sub in df.groupby("length"):
        bias = sub["number_of_fragments"] / sub["number_of_positions"]
        bias = bias.replace([np.inf, -np.inf], np.nan)
        bias = bias / np.nanmean(bias)
        df.loc[sub.index, "GC_bias"] = bias.values

    df = df.sort_values(by=["GC_content", "length"]).reset_index(drop=True)

    # Smooth across neighbouring lengths (Griffin GC_smoothing_step window).
    df["smoothed_GC_bias"] = np.nan
    for L in range(size_range[0], size_range[1] + 1):
        min_len = int(L - GC_SMOOTHING_STEP / 2)
        max_len = int(L + GC_SMOOTHING_STEP / 2)
        window = df[(df["length"] >= min_len) & (df["length"] <= max_len)].copy()
        window = window.sort_values(by=["GC_content", "length"])
        smoothed = _median_smoothing(window["GC_bias"].values, 0.05)
        window["smoothed"] = smoothed
        current = window[window["length"] == L].copy()
        current["smoothed"] = np.where(
            current["number_of_positions"] == 0, np.nan, current["smoothed"]
        )
        mean = np.nanmean(current["smoothed"])
        if mean and not np.isnan(mean):
            current["smoothed"] = current["smoothed"] / mean
        df.loc[current.index, "smoothed_GC_bias"] = current["smoothed"].values

    return df[["length", "num_GC", "smoothed_GC_bias"]].sort_values(
        by=["length", "num_GC"]
    ).reset_index(drop=True)


def gcbias(args, cmdline=True):
    """``cfstats gcbias`` entry point."""
    if args.reference is None:
        raise ValueError("Reference file (-r/--reference) is required for gcbias.")

    args.samfiles = collect_bam_files(args.samfiles, getattr(args, "bamlist", None))
    if not args.samfiles:
        raise ValueError("No input alignment files provided.")

    chroms = args.chroms if getattr(args, "chroms", None) else None

    log.info("Estimating expected GC frequency from reference (%d samples)...",
             args.gc_samples)
    expected = _expected_gc_frequency(
        args.reference, args.size_range, args.gc_samples, chroms, args.seed
    )

    nproc = getattr(args, "nproc", 1) or 1
    payload = [(s, args) for s in args.samfiles]
    if nproc > 1 and len(payload) > 1:
        with Pool(nproc) as pool:
            results = list(pool.imap_unordered(_worker_observed_counts, payload))
    else:
        results = [_worker_observed_counts(pl) for pl in payload]

    out_dir = getattr(args, "out_dir", None) or "."
    os.makedirs(out_dir, exist_ok=True)

    written = {}
    for result in results:
        samfile = result["samfile"]
        table = _build_bias_table(result["counts"], expected, args.size_range)
        sample = os.path.splitext(os.path.basename(samfile))[0]
        out_path = os.path.join(out_dir, f"{sample}.GC_bias.txt")
        table.to_csv(out_path, sep="\t", index=False)
        written[samfile] = out_path
        log.info("Wrote GC bias for %s -> %s", samfile, out_path)
        if cmdline:
            sys.stdout.write(f"{samfile}\t{out_path}\n")

    if not cmdline:
        return written
