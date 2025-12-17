import numpy as np
import pysam
from scipy.signal import find_peaks

from cfstats.ft import wps


def _call_nucleosomes_from_wps(signal, start, chrom, min_prominence=5, min_distance=147):
    """Call nucleosome centers from a WPS signal using peak finding.

    Parameters
    ----------
    signal : 1D numpy.ndarray
        WPS values for positions [start, start + len(signal)).
    start : int
        Genomic start coordinate (0-based) corresponding to signal[0].
    chrom : str
        Chromosome name.
    min_prominence : float
        Minimum peak prominence to consider a nucleosome center.
    min_distance : int
        Minimum distance (in bp) between neighboring peaks.

    Returns
    -------
    peaks : list of (chrom, center, score)
        List of called nucleosome centers with WPS score as score.
    """
    if signal is None or len(signal) == 0:
        return []

    # Ensure numpy array
    signal = np.asarray(signal)

    # Basic peak finding on WPS profile
    peaks_idx, properties = find_peaks(
        signal,
        prominence=min_prominence,
        distance=min_distance,
    )

    peaks = []
    for idx in peaks_idx:
        center = start + idx
        score = float(signal[idx])
        peaks.append((chrom, center, score))

    return peaks


def nucs(args, cmdline=True):
    """Call nucleosomes from WPS profiles for one or more alignment files.

    Behaviour:
    - If args.chrom is None: iterate genome-wide over all contigs.
    - If args.chrom is set but args.start/args.end are None: use full chromosome.
    - If args.chrom, args.start and args.end are all set: use the specified region
      (0-based, half-open).

    Other parameters:
    - args.samfiles : list of sam/bam/cram paths
    - args.reference : reference fasta (required for CRAM)
    - args.k : WPS window size
    - args.min_len, args.max_len : fragment length range for WPS
    - args.min_prominence : minimum WPS peak prominence
    - args.min_distance : minimum distance between peaks

    Output (cmdline=True):
        BED-like lines written to stdout:
        chrom  start  end  name  score  strand

        - start/end: single-base interval around peak center (center, center+1)
        - name: input filename
        - score: WPS value at peak
        - strand: "."
    """

    k = args.k
    min_len = args.min_len
    max_len = args.max_len
    min_prominence = args.min_prominence
    min_distance = args.min_distance

    all_peaks = []

    for samfile in args.samfiles:
        mode = "rb"
        # Open alignment file; reference needed for CRAM
        pysamfile = pysam.AlignmentFile(
            samfile,
            mode,
            reference_filename=args.reference,
        )

        # Determine regions to scan
        regions = []  # list of (chrom, start, end)

        if args.chrom is None:
            # Genome-wide: all contigs
            for ctg in pysamfile.references:
                length = pysamfile.get_reference_length(ctg)
                if length is None or length <= 0:
                    continue
                regions.append((ctg, 0, length))
        else:
            # Single chromosome, maybe full or sub-region
            chrom = args.chrom
            try:
                chrom_len = pysamfile.get_reference_length(chrom)
            except ValueError:
                raise ValueError(f"Chromosome {chrom} not found in {samfile}")

            start = args.start if args.start is not None else 0
            end = args.end if args.end is not None else chrom_len

            if start < 0 or end <= start or end > chrom_len:
                raise ValueError(
                    f"Invalid region {chrom}:{start}-{end} for {samfile} (length {chrom_len})"
                )

            regions.append((chrom, start, end))

        # For each region, compute WPS and call peaks
        for chrom, start, end in regions:
            signal = wps(
                pysamfile,
                chrom,
                start,
                end,
                k=k,
                min_len=min_len,
                max_len=max_len,
            )

            peaks = _call_nucleosomes_from_wps(
                signal,
                start=start,
                chrom=chrom,
                min_prominence=min_prominence,
                min_distance=min_distance,
            )

            if not cmdline:
                all_peaks.append({"samfile": samfile, "chrom": chrom, "start": start, "end": end, "peaks": peaks})
                continue

            # Write BED-like output
            for chrom_, center, score in peaks:
                # single-base interval; could be widened downstream
                bed_start = center
                bed_end = center + 1
                name = samfile
                strand = "."
                # tab-separated: chrom start end name score strand
                if getattr(args, "logger", None) is not None:
                    args.logger.info(
                        f"NUC_CALL\t{samfile}\t{chrom_}:{bed_start}-{bed_end}\t{score:.2f}"
                    )
                print(
                    f"{chrom_}\t{bed_start}\t{bed_end}\t{name}\t{score:.2f}\t{strand}"
                )

        pysamfile.close()

    if not cmdline:
        return all_peaks

    return None
