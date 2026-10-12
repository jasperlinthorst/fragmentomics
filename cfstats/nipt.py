import copy
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from cfstats import bincounts, ff, utils
from cfstats.utils import collect_bam_files


def _parse_bin_label(label):
    try:
        chrom, start, end = str(label).rsplit("_", 2)
        return chrom, int(start), int(end)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Reference bin column {label!r} is not formatted as chrom_start_end"
        ) from exc


def _validate_profile(sample, null, std, ff_value):
    if not 0 < ff_value <= 1:
        raise ValueError("Fetal fraction must be greater than 0 and at most 1")
    if not sample.index.equals(null.index) or not null.index.equals(std.index):
        raise ValueError("Sample, reference mean, and reference standard deviation must use identical bins")
    values = np.concatenate((sample.to_numpy(), null.to_numpy(), std.to_numpy()))
    if not np.isfinite(values).all():
        raise ValueError("NIPT profiles contain non-finite values")
    if (std <= 0).any():
        raise ValueError("Reference standard deviations must be greater than zero in every retained bin")


def calc_target_scores(sample, null, std, target_mask, ff_value, ratio_score=False):
    sample = pd.Series(sample, index=null.index, dtype=float)
    null = pd.Series(null, dtype=float)
    std = pd.Series(std, index=null.index, dtype=float)
    target_mask = np.asarray(target_mask, dtype=bool)
    _validate_profile(sample, null, std, ff_value)
    if target_mask.shape != (len(null),) or not target_mask.any():
        raise ValueError("Each NIPT target must contain at least one retained reference bin")

    llnull = norm.logpdf(sample.to_numpy(), loc=null.to_numpy(), scale=std.to_numpy()).sum()
    scores = {}
    for direction, change in (("gain", ff_value / 2), ("deletion", -ff_value / 2)):
        alternative = null.to_numpy(copy=True)
        alternative[target_mask] *= 1 + change
        alternative /= alternative.sum()
        llalt = norm.logpdf(sample.to_numpy(), loc=alternative, scale=std.to_numpy()).sum()
        scores[f"{direction}_LLR"] = 2 * (llalt - llnull)

    # Z-score: mean of per-bin Z-scores within the target, scaled by sqrt(n_bins)
    z_per_bin = (sample.to_numpy()[target_mask] - null.to_numpy()[target_mask]) / std.to_numpy()[target_mask]
    n_bins = target_mask.sum()
    scores["Zscore"] = z_per_bin.mean() * np.sqrt(n_bins)

    if ratio_score:
        null_mass = null.to_numpy()[target_mask].sum()
        if null_mass <= 0:
            raise ValueError("Reference abundance for an NIPT target must be greater than zero")
        relative_shift = sample.to_numpy()[target_mask].sum() / null_mass - 1
        gain_implied_ff = min(1.0, max(0.0, 2 * relative_shift))
        deletion_implied_ff = min(1.0, max(0.0, -2 * relative_shift))
        scores["gain_MR"] = gain_implied_ff / ff_value
        scores["deletion_MR"] = deletion_implied_ff / ff_value

    return scores


def _read_bed_targets(path, bins):
    targets = {}
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4 or not fields[3]:
                raise ValueError(f"BED line {line_number} must have a nonempty name in column 4")
            chrom, name = fields[0], fields[3]
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise ValueError(f"BED line {line_number} has non-integer coordinates") from exc
            if start < 0 or end <= start:
                raise ValueError(f"BED line {line_number} must be a nonempty 0-based half-open interval")
            if name in targets:
                raise ValueError(f"BED target names must be unique: {name!r}")
            mask = np.array([
                bin_chrom == chrom and bin_start < end and start < bin_end
                for bin_chrom, bin_start, bin_end in bins
            ])
            if not mask.any():
                raise ValueError(f"BED target {name!r} does not overlap any retained reference bins")
            targets[name] = mask
    if not targets:
        raise ValueError("BED file does not contain any named regions")
    return targets


def build_targets(columns, regions=None):
    bins = [_parse_bin_label(column) for column in columns]
    if regions:
        return _read_bed_targets(regions, bins)
    targets = {}
    for chrom, _, _ in bins:
        if chrom not in targets:
            targets[chrom] = np.array([bin_chrom == chrom for bin_chrom, _, _ in bins])
    return targets


def _load_reference(args):
    reference = pd.read_table(args.referencesamples, index_col=0)
    parsed = {column: _parse_bin_label(column) for column in reference.columns}
    full_columns = [
        column for column, (_, start, end) in parsed.items()
        if end - start == args.binsize
    ]
    if not full_columns:
        raise ValueError("Reference sample table has no full bins matching --binsize")
    reference = reference.loc[:, full_columns]
    no_n_columns = [
        column for column in reference.columns
        if utils.get_N_content_from_fasta(args.reference, *parsed[column]) < 0.01
    ]
    reference = reference.loc[:, no_n_columns]
    totals = reference.sum(axis=1)
    reference = reference[(totals > 5e6) & (totals < 50e6)]
    if len(reference) < 2:
        raise ValueError("At least two usable reference samples are required")
    if reference.shape[1] == 0:
        raise ValueError("No reference bins remain after N-content filtering")
    if args.gccorrect:
        gc_content = utils.get_gc_content(reference, reference=args.reference)
        reference = utils.gc_correct_counts(reference, gc_content, frac=args.frac)
    return reference.div(reference.sum(axis=1), axis=0)


def _worker_args(args):
    worker_args = copy.copy(args)
    if hasattr(worker_args, "func"):
        del worker_args.func
    return worker_args


def _get_fetal_fractions(args):
    if args.ff is not None:
        return {sample: args.ff for sample in args.samfiles}
    predictions = ff.ff(_worker_args(args), cmdline=False)
    return dict(zip(args.samfiles, predictions))


def nipt(args):
    args.samfiles = collect_bam_files(args.samfiles, getattr(args, "bamlist", None))
    args.bamlist = None
    if not args.samfiles:
        raise ValueError("No input alignment files provided")

    fetal_fractions = _get_fetal_fractions(args)
    reference = _load_reference(args)
    targets = build_targets(reference.columns, getattr(args, "regions", None))

    columns, counts = bincounts.bincounts(_worker_args(args), cmdline=False)
    samples = pd.DataFrame(counts, columns=columns, index=args.samfiles, dtype=float)
    missing = reference.columns.difference(samples.columns)
    if len(missing):
        raise ValueError(f"Sample counts are missing {len(missing)} reference bins")
    samples = samples.loc[:, reference.columns]
    samples = samples.div(samples.sum(axis=1), axis=0)

    fields = ["filename", "FF"]
    for target in targets:
        fields.extend((f"{target}_Zscore", f"{target}_gain_LLR", f"{target}_deletion_LLR"))
        if args.ratio_score:
            fields.extend((f"{target}_gain_MR", f"{target}_deletion_MR"))
    sys.stdout.write("\t".join(fields) + "\n")

    null = reference.mean()
    std = reference.std()
    for sample_name, sample in samples.iterrows():
        row = [sample_name, str(fetal_fractions[sample_name])]
        for mask in targets.values():
            scores = calc_target_scores(
                sample, null, std, mask, fetal_fractions[sample_name], args.ratio_score
            )
            row.extend((str(scores["Zscore"]), str(scores["gain_LLR"]), str(scores["deletion_LLR"])))
            if args.ratio_score:
                row.extend((str(scores["gain_MR"]), str(scores["deletion_MR"])))
        sys.stdout.write("\t".join(row) + "\n")
        sys.stdout.flush()
