from itertools import starmap
import numpy as np
import pysam
import gffutils
from scipy.fft import fft, fftfreq
from scipy.signal import lfilter, periodogram
from scipy.interpolate import interp1d
import os
import sys
from multiprocessing import Pool
import logging as log_module

from cfstats import utils


def _soft_clipped(cigar):
    return any(operation in (4, 5, 6) for operation, length in (cigar or []))


def _leuven_regions(path):
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("Leuven annotation must have five columns: gene_id chromosome start end strand")
            gene_id, chromosome, start, end, strand = fields
            yield gene_id, chromosome, int(start), int(end), strand


def _next_235_length(length):
    candidate = int(length)
    while True:
        remainder = candidate
        for factor in (2, 3, 5):
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1:
            return candidate
        candidate += 1


def fft_wps_intensity(signal, ampmin=193, ampmax=199, pmin=120, pmax=280,
                      npoints=100, args=None, taper=0.3, pad=0.3):
    """Return per-period FFT-WPS amplitudes for the configured band.

    The period band is defined by ``--amplitude-min``, ``--amplitude-max`` and
    ``--amplitude-step`` (read from ``args`` when provided, otherwise from the
    function defaults).  Callers that need a single scalar should take the
    mean of the returned list (``np.nanmean``).
    """
    # npoints is kept only for backward compatibility with direct callers.
    signal = np.asarray(signal, dtype=float)
    periods = list(range(
        int(round(getattr(args, 'ampmin', ampmin))),
        int(round(getattr(args, 'ampmax', ampmax))) + 1,
        getattr(args, 'ampstep', 1) or 1))
    n_periods = len(periods)
    if not n_periods:
        return [np.nan]

    if not getattr(args, "leuven", False):
        if (signal.size == 0 or np.all(np.isnan(signal))
                or np.all(signal == signal.flat[0])):
            return [np.nan] * n_periods
        frequencies, power_spectrum = periodogram(
            signal, fs=1, scaling='spectrum')
        if frequencies.size < 2:
            return [np.nan] * n_periods
        sample_periods = 1 / frequencies[1:]
        period_mask = (sample_periods >= pmin) & (sample_periods <= pmax)
        target_periods = sample_periods[period_mask]
        target_intensity = power_spectrum[1:][period_mask]
        if target_periods.size < 2:
            return [np.nan] * n_periods
        interpolation_function = interp1d(
            target_periods, target_intensity,
            bounds_error=False, fill_value=np.nan)
        return [float(interpolation_function(p)) for p in periods]

    recursive_filter = 1 / np.arange(5, 101, 4, dtype=float)
    prefixed = np.concatenate((signal[:300], signal))
    denominator = np.concatenate(([1.0], -recursive_filter))
    signal = lfilter([1.0], denominator, prefixed)[300:]

    trim = int(np.floor(0.1 * signal.size))
    trimmed_mean = np.mean(np.sort(signal)[trim:signal.size - trim])
    signal = signal - trimmed_mean

    n_original = signal.size
    time = np.arange(1, n_original + 1) - (n_original + 1) / 2
    sum_time_squared = n_original * (n_original ** 2 - 1) / 12
    signal = signal - np.mean(signal) - np.sum(signal * time) * time / sum_time_squared

    taper_length = int(np.floor(n_original * taper))
    weights = 0.5 * (1 - np.cos(
        np.pi * np.arange(1, 2 * taper_length, 2) / (2 * taper_length)))
    signal = signal * np.concatenate((
        weights, np.ones(n_original - 2 * taper_length), weights[::-1]))

    padded_length = n_original + int(n_original * pad)
    fft_length = _next_235_length(padded_length)
    signal = np.pad(signal, (0, fft_length - n_original))
    periodogram_values = np.abs(np.fft.fft(signal)) ** 2 / n_original
    periodogram_values[0] = 0.5 * (
        periodogram_values[1] + periodogram_values[-1])
    periodogram_values = (
        0.5 * periodogram_values
        + 0.25 * np.roll(periodogram_values, 1)
        + 0.25 * np.roll(periodogram_values, -1))

    n_spectrum = fft_length // 2
    frequencies = np.arange(1, n_spectrum + 1) / fft_length
    spectrum = periodogram_values[1:n_spectrum + 1]
    spectrum /= 1 - (5 / 8) * taper * 2
    rounded_periods = np.round(1 / frequencies)
    values = []
    for period in periods:
        matches = np.flatnonzero(rounded_periods == period)
        if matches.size == 0:
            values.append(np.nan)
        else:
            values.append(float(spectrum[matches[0]]))
    return values


def wps(bam_file, chromosome, start_query, end_query, k=120, min_len=120,
        max_len=180, args=None):
    """Calculate WPS using the standard or Leuven definition selected by args."""
    leuven = bool(getattr(args, "leuven", False))
    region_length = end_query - start_query + int(leuven)
    if region_length <= 0:
        empty = np.array([], dtype=int)
        return (empty, False) if leuven else empty

    span_diff = np.zeros(region_length + 1, dtype=int)
    subtract_diff = np.zeros(region_length + 1, dtype=int)
    half = k // 2
    fetch_start = max(0, start_query - (half + 1 if leuven else k))
    fetch_end = end_query + (half + 1 if leuven else k)
    covered = False
    reqflag = getattr(args, "reqflag", None) if args is not None else None
    exclflag = getattr(args, "exclflag", None) if args is not None else None
    mapqual = getattr(args, "mapqual", None) if args is not None else None

    try:
        for read in bam_file.fetch(chromosome, fetch_start, fetch_end):
            if reqflag is not None and read.flag & reqflag != reqflag:
                continue
            if exclflag is not None and read.flag & exclflag:
                continue
            if mapqual is not None and read.mapping_quality < mapqual:
                continue

            if leuven:
                if _soft_clipped(read.cigartuples):
                    continue
                if not read.is_paired or read.mate_is_unmapped:
                    continue
                if read.next_reference_id != read.reference_id:
                    continue
                if not (read.is_read1 or
                        (read.is_read2 and read.next_reference_start
                         + read.query_length < fetch_start)):
                    continue
                fragment_length = abs(read.template_length)
                if fragment_length == 0 or not min_len <= fragment_length <= max_len:
                    continue
                fragment_start = min(
                    read.reference_start, read.next_reference_start) + 1
                fragment_end = fragment_start + fragment_length - 1
                covered |= (fragment_end >= start_query
                            and fragment_start <= end_query)

                overlap_start = max(start_query, fragment_start - half + 1)
                overlap_end = min(end_query, fragment_end + half - 1)
                if overlap_start <= overlap_end:
                    i = overlap_start - start_query
                    j = overlap_end - start_query
                    subtract_diff[i] -= 1
                    subtract_diff[j + 1] += 1

                span_start = max(start_query, fragment_start + half)
                span_end = min(end_query, fragment_end - half)
                if span_start <= span_end:
                    i = span_start - start_query
                    j = span_end - start_query
                    span_diff[i] += 2
                    span_diff[j + 1] -= 2
            else:
                if not (read.is_paired and read.is_proper_pair and read.is_read1):
                    continue
                fragment_length = abs(read.template_length)
                if not min_len <= fragment_length <= max_len:
                    continue
                if read.template_length > 0:
                    fragment_start = read.reference_start
                    fragment_end = read.reference_start + read.template_length
                else:
                    fragment_start = read.reference_start + read.template_length
                    fragment_end = read.reference_start
                if fragment_end <= fetch_start or fragment_start >= fetch_end:
                    continue

                span_start = max(0, fragment_start + half - start_query)
                span_end = min(
                    region_length - 1, fragment_end - half - start_query)
                if span_start <= span_end:
                    span_diff[span_start] += 1
                    span_diff[span_end + 1] -= 1

                for endpoint in (fragment_start, fragment_end - 1):
                    endpoint_start = max(
                        0, endpoint - half + 1 - start_query)
                    endpoint_end = min(
                        region_length - 1, endpoint + half - start_query)
                    if endpoint_start <= endpoint_end:
                        subtract_diff[endpoint_start] += 1
                        subtract_diff[endpoint_end + 1] -= 1
    except Exception as error:
        print(f"Error processing BAM file: {error}")
        failed = np.full(region_length, np.nan)
        return (failed, False) if leuven else failed

    if leuven:
        signal = np.cumsum(span_diff[:-1]) + np.cumsum(subtract_diff[:-1])
        return signal, covered
    return np.cumsum(span_diff[:-1]) - np.cumsum(subtract_diff[:-1])

def worker_fourier_transform_samfile(pl):
    import traceback

    try:
        samfile, args = pl
        logger = log_module.getLogger("cfstats.fourier")
        logger.debug(f"Processing samfile {samfile}")
        bam = pysam.AlignmentFile(
            samfile, "rb",
            reference_filename=args.reference if args.reference is not None else None)
        references = set(bam.references)
        leuven = bool(getattr(args, "leuven", False))
        values = {}

        if leuven or args.gfffile.lower().endswith(('.tsv', '.txt')):
            regions = _leuven_regions(args.gfffile)
        else:
            db_filename = f'{args.gfffile}.db'
            if not os.path.exists(db_filename):
                logger.info("Constructing gene DB...")
                db = gffutils.create_db(
                    args.gfffile, dbfn=db_filename, force=True,
                    keep_order=True, merge_strategy='merge',
                    sort_attribute_values=True)
            else:
                db = gffutils.FeatureDB(db_filename, keep_order=True)

            def standard_regions():
                for gene in db.features_of_type('gene'):
                    if gene.strand == '-':
                        start, end = gene.end - int(args.window), gene.end
                    else:
                        start, end = gene.start, gene.start + int(args.window)
                    name = (gene.attributes["gene_name"][0]
                            if "gene_name" in gene.attributes
                            else gene.attributes["gene_id"][0])
                    yield name, gene.chrom, start, end, gene.strand
            regions = standard_regions()

        try:
            for name, chromosome, start, end, strand in regions:
                bam_chromosome = (chromosome if chromosome in references
                                  else 'chr' + chromosome)
                if bam_chromosome not in references:
                    logger.debug(f"{chromosome} not in samfile, skipping {name}.")
                    continue
                result = wps(
                    bam, bam_chromosome, start, end,
                    min_len=120, max_len=180, args=args)
                if leuven:
                    signal, covered = result
                    if not covered:
                        continue
                    if strand == '-':
                        signal = signal[::-1]
                else:
                    signal = result
                period_values = fft_wps_intensity(signal, args=args)
                return_periods = getattr(args, '_return_periods', False)
                values[name] = (period_values if return_periods
                                else float(np.nanmean(period_values)))
        finally:
            bam.close()

        return {'samfile': samfile, 'fft': values}
    except Exception as error:
        print("WORKER EXCEPTION:", error, file=sys.stderr)
        traceback.print_exc()
        raise

def _output_periods(args):
    return list(range(
        getattr(args, 'ampmin', 193),
        getattr(args, 'ampmax', 199) + 1,
        getattr(args, 'ampstep', 1)))


def _choose_output_format(args):
    leuven = bool(getattr(args, "leuven", False))
    long_flag = bool(getattr(args, "long_format", False))
    wide_flag = bool(getattr(args, "wide_format", False))
    if long_flag and wide_flag:
        raise ValueError("--long and --wide are mutually exclusive")
    if leuven:
        return 'long' if long_flag else ('wide' if wide_flag else 'long')
    return 'wide' if wide_flag else ('long' if long_flag else 'wide')


def fourier_transform_coverage(args):
    logger = log_module.getLogger("cfstats.fourier")
    args.samfiles = utils.collect_bam_files(args.samfiles, getattr(args, 'bamlist', None))
    output_format = _choose_output_format(args)
    args._return_periods = (output_format == 'long')

    if output_format == 'long' and len(args.samfiles) != 1:
        raise ValueError(
            "--long requires exactly one BAM/CRAM file; use --wide for multiple samples")

    if getattr(args, "leuven", False):
        explicit_options = getattr(args, '_explicit_options', {})
        if not explicit_options.get('--amplitude-step', False):
            args.ampstep = 3
        elif args.ampstep != 3:
            logger.warning(
                "--amplitude-step %s overrides the Leuven preset 3",
                args.ampstep)
        presets = {'-f': ('reqflag', 1), '-F': ('exclflag', 1548),
                   '-q': ('mapqual', 0)}
        explicit = getattr(args, '_explicit_filters', {})
        for flag, (attribute, preset) in presets.items():
            if explicit.get(flag, False):
                value = getattr(args, attribute)
                if value != preset:
                    logger.warning(
                        "%s %s overrides the Leuven preset %s",
                        flag, value, preset)
            else:
                setattr(args, attribute, preset)

    if output_format == 'long':
        result = worker_fourier_transform_samfile((args.samfiles[0], args))
        periods = _output_periods(args)
        sys.stdout.write("#Region\t" + "\t".join(map(str, periods)) + "\n")
        for gene, values in result['fft'].items():
            sys.stdout.write(gene + "\t" + "\t".join(map(str, values)) + "\n")
        return

    # wide format: samples as rows, genes as columns
    if getattr(args, "leuven", False):
        # Leuven wide output needs the union of all observed genes before the
        # header can be written, so results are buffered but computed in parallel.
        utils.require_sample_names(args)
        if args.nproc > 1:
            with Pool(args.nproc) as pool:
                results = list(pool.imap_unordered(
                    worker_fourier_transform_samfile,
                    zip(args.samfiles, [args] * len(args.samfiles))))
        else:
            results = [worker_fourier_transform_samfile((samfile, args))
                       for samfile in args.samfiles]

        all_genes = []
        seen = set()
        for r in results:
            for gene in r['fft'].keys():
                if gene not in seen:
                    seen.add(gene)
                    all_genes.append(gene)
        if not all_genes:
            return
        sys.stdout.write("#filename\t" + "\t".join(all_genes) + "\n")
        sys.stdout.flush()
        for result in results:
            fft_values = result['fft']
            row = [result['samfile']]
            for gene in all_genes:
                value = fft_values.get(gene)
                row.append("nan" if value is None else str(value))
            sys.stdout.write("\t".join(row) + "\n")
            sys.stdout.flush()
    else:
        utils.require_sample_names(args)
        header_written = False
        if args.nproc > 1:
            with Pool(args.nproc) as pool:
                iterator = pool.imap_unordered(
                    worker_fourier_transform_samfile,
                    zip(args.samfiles, [args] * len(args.samfiles)))
        else:
            iterator = (worker_fourier_transform_samfile((samfile, args))
                        for samfile in args.samfiles)

        for result in iterator:
            fft_values = result['fft']
            if not header_written:
                sys.stdout.write("#filename\t" + "\t".join(fft_values) + "\n")
                sys.stdout.flush()
                header_written = True
            sys.stdout.write("\t".join(
                [result['samfile']] + list(map(str, fft_values.values()))) + "\n")
            sys.stdout.flush()
