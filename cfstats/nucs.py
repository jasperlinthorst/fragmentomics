import gzip
import logging as log_module

import numpy as np
import pysam
from scipy.signal import find_peaks, peak_widths

from cfstats.ft import wps
from cfstats.utils import collect_bam_files


def _call_nucleosomes_from_wps(signal, start, chrom, min_prominence=5, min_distance=147, rel_height=0.5):
    """Call nucleosomes from a WPS signal using peak finding.

    The extent of each nucleosome is inferred from the peak itself: it spans
    the positions where the WPS stays above ``rel_height`` of the peak's
    prominence (``rel_height=0.5`` gives the full width at half prominence).

    Parameters
    ----------
    signal : 1D numpy.ndarray
        WPS values for positions [start, start + len(signal)).
    start : int
        Genomic start coordinate (0-based) corresponding to signal[0].
    chrom : str
        Chromosome name.
    min_prominence : float
        Minimum peak prominence to consider a nucleosome.
    min_distance : int
        Minimum distance (in bp) between neighboring peaks.
    rel_height : float
        Relative height (fraction of prominence, measured down from the summit)
        at which the nucleosome boundaries are determined.

    Returns
    -------
    peaks : list of (chrom, nuc_start, nuc_end, summit, score)
        Half-open [nuc_start, nuc_end) nucleosome intervals, the summit position
        and the WPS value at the summit.
    """
    if signal is None or len(signal) == 0:
        return []

    signal = np.asarray(signal, dtype=float)

    peaks_idx, properties = find_peaks(
        signal,
        prominence=min_prominence,
        distance=min_distance,
    )
    if len(peaks_idx) == 0:
        return []

    _, _, left_ips, right_ips = peak_widths(
        signal,
        peaks_idx,
        rel_height=rel_height,
        prominence_data=(properties["prominences"], properties["left_bases"], properties["right_bases"]),
    )

    peaks = []
    for idx, left, right in zip(peaks_idx, left_ips, right_ips):
        nuc_start = start + min(int(np.ceil(left)), idx)
        nuc_end = start + max(int(np.floor(right)), idx) + 1
        peaks.append((chrom, nuc_start, nuc_end, start + int(idx), float(signal[idx])))

    return peaks


class _WPSTrackWriter:
    """Write a bp-resolution WPS track as bedGraph (optionally gzipped) or bigWig."""

    def __init__(self, path, contigs, name="WPS"):
        self.bigwig = path.lower().endswith((".bw", ".bigwig"))
        if self.bigwig:
            try:
                import pyBigWig
            except ImportError:
                raise ImportError("Writing bigWig output requires pyBigWig (pip install pyBigWig); "
                                  "alternatively write a .bedGraph(.gz) file.")
            self.fh = pyBigWig.open(path, "w")
            self.fh.addHeader(list(contigs))
        else:
            self.fh = gzip.open(path, "wt") if path.endswith(".gz") else open(path, "w")
            self.fh.write(f'track type=bedGraph name="{name}" description="Window protection score" '
                          f'visibility=full autoScale=on\n')

    def write(self, chrom, start, values):
        values = np.asarray(values)
        keep = ~np.isnan(values) if values.dtype.kind == "f" else np.ones(len(values), dtype=bool)
        if not keep.any():
            return
        # Run-length encode consecutive identical values into bedGraph intervals
        breaks = np.flatnonzero((np.diff(values) != 0) | (np.diff(keep) != 0)) + 1
        run_starts = np.concatenate(([0], breaks))
        run_ends = np.concatenate((breaks, [len(values)]))
        mask = keep[run_starts]
        run_starts, run_ends = run_starts[mask], run_ends[mask]
        run_values = values[run_starts].astype(float)
        if self.bigwig:
            self.fh.addEntries([chrom] * len(run_starts), (start + run_starts).tolist(),
                               ends=(start + run_ends).tolist(), values=run_values.tolist())
        else:
            self.fh.writelines(f"{chrom}\t{start + s}\t{start + e}\t{v:g}\n"
                               for s, e, v in zip(run_starts, run_ends, run_values))

    def close(self):
        self.fh.close()


def nucs(args, cmdline=True):
    """Call nucleosomes from the aggregated WPS profile of one or more alignment files.

    WPS is computed per file and summed over all files (equivalent to computing
    WPS over the pooled reads), after which peaks are called on the aggregate.

    Behaviour:
    - If args.chrom is None: iterate genome-wide over all contigs.
    - If args.chrom is set but args.start/args.end are None: use full chromosome.
    - If args.chrom, args.start and args.end are all set: use the specified region
      (0-based, half-open).

    Regions are processed in chunks of ``args.chunk_size`` bp, padded on both
    sides so that peak calling is not affected by chunk boundaries.

    Other parameters:
    - args.samfiles : list of sam/bam/cram paths
    - args.reference : reference fasta (required for CRAM)
    - args.k : WPS window size
    - args.min_len, args.max_len : fragment length range for WPS
    - args.min_prominence : minimum WPS peak prominence
    - args.min_distance : minimum distance between peaks
    - args.rel_height : relative peak height at which nucleosome boundaries are set
    - args.wps_out : optional path for a bp-resolution WPS track
      (.bedGraph, .bedGraph.gz or .bw)

    Output (cmdline=True):
        BED8 lines written to stdout:
        chrom  start  end  name  score  strand  thickStart  thickEnd

        - start/end: inferred nucleosome extent
        - name: "."
        - score: aggregated WPS value at the summit
        - strand: "."
        - thickStart/thickEnd: summit position (summit, summit+1)
    """
    logger = getattr(args, "logger", None) or log_module.getLogger("cfstats.nucs")

    args.samfiles = collect_bam_files(args.samfiles, getattr(args, 'bamlist', None))
    if not args.samfiles:
        raise ValueError("No alignment files provided")

    k = args.k
    chunk_size = getattr(args, "chunk_size", 1000000)
    rel_height = getattr(args, "rel_height", 0.5)
    pad = max(2000, 4 * k, 4 * args.min_distance)

    pysamfiles = [pysam.AlignmentFile(samfile, "rb", reference_filename=args.reference)
                  for samfile in args.samfiles]
    contig_lengths = dict(zip(pysamfiles[0].references, pysamfiles[0].lengths))

    regions = []  # list of (chrom, start, end)
    if args.chrom is None:
        regions = [(ctg, 0, length) for ctg, length in contig_lengths.items() if length > 0]
    else:
        chrom = args.chrom
        if chrom not in contig_lengths:
            raise ValueError(f"Chromosome {chrom} not found in {args.samfiles[0]}")
        chrom_len = contig_lengths[chrom]
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None else chrom_len
        if start < 0 or end <= start or end > chrom_len:
            raise ValueError(f"Invalid region {chrom}:{start}-{end} (length {chrom_len})")
        regions.append((chrom, start, end))

    wps_out = getattr(args, "wps_out", None)
    track = _WPSTrackWriter(wps_out, contig_lengths.items()) if wps_out else None

    all_peaks = []
    try:
        for chrom, start, end in regions:
            chrom_len = contig_lengths[chrom]
            region_peaks = []
            for core_start in range(start, end, chunk_size):
                core_end = min(core_start + chunk_size, end)
                ext_start = max(0, core_start - pad)
                ext_end = min(chrom_len, core_end + pad)

                signal = np.zeros(ext_end - ext_start, dtype=float)
                for pysamfile in pysamfiles:
                    signal += wps(pysamfile, chrom, ext_start, ext_end, k=k,
                                  min_len=args.min_len, max_len=args.max_len, args=args)

                if track is not None:
                    track.write(chrom, core_start, signal[core_start - ext_start:core_end - ext_start])

                if np.isnan(signal).any():
                    logger.warning(f"WPS could not be computed for {chrom}:{ext_start}-{ext_end}, skipping")
                    continue

                peaks = [p for p in _call_nucleosomes_from_wps(
                            signal, start=ext_start, chrom=chrom,
                            min_prominence=args.min_prominence,
                            min_distance=args.min_distance,
                            rel_height=rel_height)
                         if core_start <= p[3] < core_end]

                if not cmdline:
                    region_peaks.extend(peaks)
                    continue

                for chrom_, nuc_start, nuc_end, summit, score in peaks:
                    logger.debug(f"NUC_CALL\t{chrom_}:{nuc_start}-{nuc_end}\tsummit={summit}\t{score:.2f}")
                    print(f"{chrom_}\t{nuc_start}\t{nuc_end}\t.\t{score:.2f}\t.\t{summit}\t{summit + 1}")

            if not cmdline:
                all_peaks.append({"chrom": chrom, "start": start, "end": end, "peaks": region_peaks})
    finally:
        if track is not None:
            track.close()
        for pysamfile in pysamfiles:
            pysamfile.close()

    if not cmdline:
        return all_peaks

    return None
