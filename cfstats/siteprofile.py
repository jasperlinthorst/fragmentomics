"""Griffin-style composite nucleosome coverage profiling for cfstats.

For each site list and each alignment file, this computes a composite
GC-corrected fragment-midpoint coverage profile across a window centred on the
sites, normalises it to a mean of 1, smooths it, and extracts three features:

- ``mean_coverage``    : mean coverage over the save window.
- ``central_coverage`` : mean coverage over the (narrow) centre window.
- ``amplitude``        : magnitude of a chosen FFT component over the save window.

This is a region-oriented (per-site fetch) re-implementation of Griffin's
``griffin_coverage.py`` + ``griffin_merge_sites.py`` -- no genome-wide bigWig
intermediate is produced.
"""

import os
import sys
import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pysam
from scipy.signal import savgol_filter

from cfstats.sites import load_sites
from cfstats import utils

log = logging.getLogger(__name__)

RV_MARKERS = ("-", -1, "-1")


def _load_gc_bias(path):
    """Load a GC-bias table into a nested dict[length][num_GC] -> bias."""
    if path is None:
        return None
    df = pd.read_csv(path, sep="\t")
    df = df[["length", "num_GC", "smoothed_GC_bias"]]
    bias = {}
    for length, sub in df.groupby("length"):
        bias[int(length)] = dict(zip(sub["num_GC"].astype(int), sub["smoothed_GC_bias"]))
    return bias


def _resolve_chrom(chrom, samctgs):
    if chrom in samctgs:
        return chrom
    if "chr" + chrom in samctgs:
        return "chr" + chrom
    if chrom.startswith("chr") and chrom[3:] in samctgs:
        return chrom[3:]
    return None


def _site_coverage(bam, fasta, chrom, fetch_start, fetch_end, gc_bias, args):
    """GC-corrected fragment-midpoint coverage over [fetch_start, fetch_end)."""
    n = fetch_end - fetch_start
    cov = np.zeros(n, dtype=float)
    lo, hi = args.size_range

    for read in bam.fetch(chrom, max(0, fetch_start), fetch_end):
        if args.reqflag is not None and (read.flag & args.reqflag) != args.reqflag:
            continue
        if args.exclflag and (read.flag & args.exclflag) != 0:
            continue
        if read.mapping_quality < args.mapqual:
            continue
        tl = read.template_length
        if read.is_reverse or tl <= 0:
            continue
        if tl < lo or tl > hi:
            continue

        frag_start = read.reference_start
        frag_end = frag_start + tl
        midpoint = (frag_start + frag_end) // 2
        if midpoint < fetch_start or midpoint >= fetch_end:
            continue

        if gc_bias is not None and fasta is not None:
            seq = fasta.fetch(chrom, frag_start, frag_end).upper()
            num_gc = seq.count("G") + seq.count("C") + seq.count("S")
            b = gc_bias.get(tl, {}).get(num_gc, np.nan)
            weight = 1.0 / b if (b is not None and not np.isnan(b) and b > 0) else 0.0
        else:
            weight = 1.0

        cov[midpoint - fetch_start] += weight

    return cov


def _bin_signal(arr, step):
    """Average an array into non-overlapping bins of size ``step``."""
    n = (len(arr) // step) * step
    if n == 0:
        return np.array([])
    return arr[:n].reshape(-1, step).mean(axis=1)


def _profile_site_list(bam, fasta, sites, gc_bias, args, samctgs):
    """Composite binned/normalised/smoothed profile for one site list."""
    lo, hi = args.norm_window
    length = hi - lo

    acc = np.zeros(length, dtype=float)
    n_sites = 0

    for row in sites.itertuples(index=False):
        chrom = _resolve_chrom(str(row.chrom), samctgs)
        if chrom is None:
            continue
        pos = int(row.position)

        if row.strand in RV_MARKERS:
            fetch_start = pos - hi
            fetch_end = pos - lo
        else:
            fetch_start = pos + lo
            fetch_end = pos + hi
        if fetch_start < 0:
            continue

        cov = _site_coverage(bam, fasta, chrom, fetch_start, fetch_end, gc_bias, args)
        if row.strand in RV_MARKERS:
            cov = cov[::-1]
        acc += cov
        n_sites += 1

    if n_sites == 0:
        return None

    composite = acc / n_sites

    step = args.step
    binned = _bin_signal(composite, step)
    if binned.size == 0:
        return None
    bin_positions = lo + step * np.arange(binned.size) + step / 2.0

    # Normalise composite to a mean of 1 over the (whole) normalisation window.
    mean = np.nanmean(binned)
    if mean and not np.isnan(mean) and mean != 0:
        binned = binned / mean

    # Savitzky-Golay smoothing (~ smoothing_length in bp).
    if args.smoothing:
        win = int(round(args.smoothing_length / step))
        if win % 2 == 0:
            win += 1
        if win >= 5 and win < binned.size:
            binned = savgol_filter(binned, win, 3)

    return {
        "n_sites": n_sites,
        "bin_positions": bin_positions,
        "profile": binned,
    }


def _features(profile_result, args):
    bin_positions = profile_result["bin_positions"]
    profile = profile_result["profile"]

    save_lo, save_hi = args.save_window
    center_lo, center_hi = args.center_window

    save_mask = (bin_positions >= save_lo) & (bin_positions < save_hi)
    center_mask = (bin_positions >= center_lo) & (bin_positions < center_hi)

    mean_coverage = float(np.nanmean(profile[save_mask])) if save_mask.any() else np.nan
    central_coverage = (
        float(np.nanmean(profile[center_mask])) if center_mask.any() else np.nan
    )

    save_profile = profile[save_mask]
    if save_profile.size > args.fft_index:
        amplitude = float(np.abs(np.fft.fft(save_profile)[args.fft_index]))
    else:
        amplitude = np.nan

    return mean_coverage, central_coverage, amplitude


def _worker_siteprofile(pl):
    samfile, args, sites_by_list = pl

    gc_bias = _load_gc_bias(getattr(args, "gc_bias", None))
    bam = pysam.AlignmentFile(samfile, reference_filename=args.reference)
    fasta = pysam.FastaFile(args.reference) if args.reference else None
    samctgs = set(bam.references)

    rows = []
    profiles = {}
    for name, sites in sites_by_list.items():
        result = _profile_site_list(bam, fasta, sites, gc_bias, args, samctgs)
        if result is None:
            log.warning("No usable sites for '%s' in %s", name, samfile)
            continue
        mean_cov, central_cov, amplitude = _features(result, args)
        rows.append({
            "filename": samfile,
            "site_list": name,
            "n_sites": result["n_sites"],
            "mean_coverage": mean_cov,
            "central_coverage": central_cov,
            "amplitude": amplitude,
        })
        profiles[name] = (result["bin_positions"], result["profile"])

    bam.close()
    if fasta is not None:
        fasta.close()

    return {"samfile": samfile, "rows": rows, "profiles": profiles}


def _write_profile(out_dir, samfile, profiles):
    sample = os.path.splitext(os.path.basename(samfile))[0]
    os.makedirs(out_dir, exist_ok=True)
    for name, (positions, profile) in profiles.items():
        df = pd.DataFrame({"position": positions.astype(int), "coverage": profile})
        path = os.path.join(out_dir, f"{sample}.{name}.profile.tsv")
        df.to_csv(path, sep="\t", index=False)
        log.info("Wrote profile %s", path)


def siteprofile(args, cmdline=True):
    """``cfstats siteprofile`` entry point."""
    args.samfiles = utils.collect_bam_files(args.samfiles, getattr(args, "bamlist", None))
    if not args.samfiles:
        raise ValueError("No input alignment files provided.")

    if getattr(args, "gc_bias", None) is not None and args.reference is None:
        raise ValueError("--gc-bias requires -r/--reference (for fragment GC content).")

    utils.require_sample_names(args)

    chroms = args.chroms if getattr(args, "chroms", None) else None
    sites_by_list = load_sites(
        args.sitesfile,
        chrom_column=args.chrom_column,
        position_column=args.position_column,
        strand_column=args.strand_column,
        chroms=chroms,
    )
    log.info("Loaded %d site list(s).", len(sites_by_list))

    nproc = getattr(args, "nproc", 1) or 1
    payload = [(s, args, sites_by_list) for s in args.samfiles]

    all_rows = []
    header_written = False

    def _process(result):
        nonlocal header_written
        rows = result["rows"]
        if getattr(args, "save_profile", None):
            _write_profile(args.save_profile, result["samfile"], result["profiles"])
        if not rows:
            return
        if cmdline:
            for row in rows:
                if not header_written:
                    if not getattr(args, "name", True):
                        row = {k: v for k, v in row.items() if k != "filename"}
                    sys.stdout.write("\t".join(row.keys()) + "\n")
                    sys.stdout.flush()
                    header_written = True
                if not getattr(args, "name", True):
                    row = {k: v for k, v in row.items() if k != "filename"}
                sys.stdout.write("\t".join(str(v) for v in row.values()) + "\n")
                sys.stdout.flush()
        else:
            all_rows.extend(rows)

    if nproc > 1 and len(payload) > 1:
        with Pool(nproc) as pool:
            for result in pool.imap_unordered(_worker_siteprofile, payload):
                _process(result)
    else:
        for pl in payload:
            _process(_worker_siteprofile(pl))

    if not cmdline:
        if not all_rows:
            raise RuntimeError("No profiles could be computed.")
        table = pd.DataFrame(all_rows)
        if not getattr(args, "name", True):
            table = table.drop(columns=["filename"])
        return table

    return None
