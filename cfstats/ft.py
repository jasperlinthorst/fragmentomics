import numpy as np
import pysam
import gffutils
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
from scipy.interpolate import interp1d
import os
from logging import log

def wps(bam_file, chromosome, start_query, end_query, k=120, min_len=120, max_len=180):
    """Calculate the Windowed Protection Score (WPS) for a genomic region.

    This implementation uses difference arrays (prefix sums) so that WPS is
    accumulated in a *single pass* over fragments, avoiding an explicit
    fragments×positions nested loop.

    The semantics are unchanged relative to the previous version:
    - For each genomic position `g` in [start_query, end_query), we consider a
      window `[g - k//2, g + k//2)`.
    - `spanning_count(g)` is the number of fragments that *fully span* this
      window.
    - `endpoint_within_count(g)` is the number of fragment endpoints that fall
      inside this window (counting at most two endpoints per fragment and
      avoiding double counting single‑base fragments).
    - `WPS(g) = spanning_count(g) - endpoint_within_count(g)`.

    Args:
        bam_file: pysam.AlignmentFile
        chromosome (str): Chromosome name (e.g., 'chr1').
        start_query (int): Start coordinate of the query region (0-based, inclusive).
        end_query (int): End coordinate of the query region (0-based, exclusive).
        k (int): Window size (120 for L-WPS, 16 for S-WPS, etc.).
        min_len (int): Minimum fragment length to consider.
        max_len (int): Maximum fragment length to consider.

    Returns:
        numpy.ndarray: Array of WPS scores (length = end_query - start_query).
    """

    # Length of the query region
    region_length = end_query - start_query
    if region_length <= 0:
        return np.array([], dtype=int)

    # Difference arrays for efficient accumulation
    # span_diff: prefix-sum gives number of fragments that fully span the window
    # end_diff:  prefix-sum gives number of endpoints that fall within the window
    span_diff = np.zeros(region_length + 1, dtype=int)
    end_diff = np.zeros(region_length + 1, dtype=int)

    # We need to fetch reads in an extended region to allow windows near the
    # edges to be fully evaluated.
    fetch_start = max(0, start_query - k)
    fetch_end = end_query + k

    try:
        for read in bam_file.fetch(chromosome, fetch_start, fetch_end):
            # Only use properly paired read1 to represent a fragment
            if not (read.is_paired and read.is_proper_pair and read.is_read1):
                continue

            template_length = abs(read.template_length)
            if template_length < min_len or template_length > max_len:
                continue

            # Derive fragment coordinates [frag_start, frag_end) in reference
            if read.template_length > 0:
                frag_start = read.reference_start
                frag_end = read.reference_start + read.template_length
            else:
                frag_start = read.reference_start + read.template_length
                frag_end = read.reference_start

            # Skip fragments that do not overlap the broader region at all
            if frag_end <= fetch_start or frag_start >= fetch_end:
                continue

            # --- 1. Fragments that COMPLETELY SPAN the window ---
            # For a fragment [frag_start, frag_end), a window centered at g
            # with bounds [g - k//2, g + k//2) is fully spanned when:
            #   frag_start <= g - k//2  and  frag_end >= g + k//2
            # => g >= frag_start + k//2  and  g <= frag_end - k//2
            g_start_span = frag_start + k // 2
            g_end_span = frag_end - k // 2

            # Convert to index space of the query region (0..region_length-1)
            i_start_span = g_start_span - start_query
            i_end_span = g_end_span - start_query

            # Clip to valid index range
            if i_end_span < 0 or i_start_span > region_length - 1:
                pass  # No contribution inside query region
            else:
                i_start_span = max(0, i_start_span)
                i_end_span = min(region_length - 1, i_end_span)
                if i_start_span <= i_end_span:
                    span_diff[i_start_span] += 1
                    span_diff[i_end_span + 1] -= 1

            # --- 2. Fragments with an ENDPOINT WITHIN the window ---
            # For an endpoint at position p (0-based), we count windows for
            # which [g - k//2, g + k//2) contains p:
            #   g - k//2 <= p < g + k//2
            # => g > p - k//2  and  g <= p + k//2
            # Integer g satisfy: g in [p - k//2 + 1, p + k//2].

            def _add_endpoint(p):
                g_min = p - k // 2 + 1
                g_max = p + k // 2
                i_start = g_min - start_query
                i_end = g_max - start_query

                if i_end < 0 or i_start > region_length - 1:
                    return
                i_start = max(0, i_start)
                i_end = min(region_length - 1, i_end)
                if i_start <= i_end:
                    end_diff[i_start] += 1
                    end_diff[i_end + 1] -= 1

            # Fragment start endpoint
            _add_endpoint(frag_start)

            # Fragment end endpoint (frag_end - 1), avoiding double counting
            if frag_start != frag_end - 1:
                _add_endpoint(frag_end - 1)

    except Exception as e:
        print(f"Error processing BAM file: {e}")
        return np.full(region_length, np.nan)

    # Accumulate counts via prefix sums
    spanning_count = np.cumsum(span_diff[:-1])
    endpoint_count = np.cumsum(end_diff[:-1])

    # WPS = fragments that span the window minus endpoints within the window
    wps_scores = spanning_count - endpoint_count
    return wps_scores

def fourier_transform_coverage(args):

    for samfile in args.samfiles:
        # Open the SAM/BAM/CRAM file
        pysamfile = pysam.AlignmentFile(samfile, "rb", reference_filename=args.reference if args.reference!=None else None)
        samctgs=set([ctg for ctg in pysamfile.references])
        # pysamfile.close()

        # Load the GFF file
        db_filename = f'{args.gfffile}.db'
        if not os.path.exists(db_filename):
            args.logger.info("Constructing gene DB...")
            db = gffutils.create_db(args.gfffile, dbfn=db_filename, force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)
            args.logger.info("Done.")
        else:
            args.logger.debug("Loading gene DB...")
            db = gffutils.FeatureDB(db_filename, keep_order=True)
            args.logger.debug("Done.")
        
        # Iterate over each gene in the GFF file
        for gene in db.features_of_type('gene'):
            # Get the coverage profile for the gene
            # coverage = np.zeros(gene.end - gene.start + 1)
            # coverage = np.zeros(10000+1)

            if gene.chrom not in samctgs:
                if 'chr'+gene.chrom in samctgs:
                    chrom='chr'+gene.chrom
                else:
                    args.logger.debug(f'{gene.chrom} not in samfile, skipping {gene}.')
                    continue
            else:
                chrom=gene.chrom
            
            if gene.strand=='-':
                start=gene.end-10000
                end=gene.end
            else:
                start=gene.start
                end=gene.start+10000

            args.logger.debug(f"Start WPS calculation for: {chrom}:{start}-{end}")
            signal = wps(pysamfile, chrom, start, end)

            args.logger.debug("WPS done.")

            # for read in pysamfile.fetch(chrom, start, end):
            #     for ref_pos in read.get_reference_positions():
            #         pos = ref_pos + 1  # 1-based
            #         if start <= pos <= end:
            #             coverage[pos - start] += 1

            # Perform Fourier transform on the coverage profile
            # fourier_transformed = fft(coverage)
            
            # signal = coverage
            # print(signal)
            args.logger.debug(f"Doing fft...")
            frequencies, power_spectrum = periodogram(signal, fs=1, scaling='spectrum')
            args.logger.debug("Done.")

            periods = 1 / frequencies[1:]
            intensity = power_spectrum[1:]

            # --- 3. Focus on the Target Frequency Range (e.g., 193-199 bp) ---
            # Filter for periods described in the source (e.g., 120 bp to 280 bp)
            period_mask = (periods >= 120) & (periods <= 280)
            target_periods = periods[period_mask]
            target_intensity = intensity[period_mask]

            # The sources mention using "smooth FFT periodograms" [7]. 
            # We'll use interpolation (a form of smoothing) to find intensity at precise periods.
            # NOTE: The intensity value is the amplitude/power at a given frequency.

            # Define the specific range for correlation analysis (193-199 bp)
            P_MIN = 193
            P_MAX = 199

            # Interpolate the periodogram to get a smoother curve (as required by sources)
            # and evaluate intensity at points between P_MIN and P_MAX.
            # We invert the periods for consistent indexing if needed, but here we interpolate directly.
            args.logger.debug("Interpolating...")
            interpolation_function = interp1d(target_periods, target_intensity)
            args.logger.debug("Done.")
            
            # Create a fine array of periods in the target range
            args.logger.debug("Creating fine periods...")
            fine_periods = np.linspace(P_MIN, P_MAX, 100)
            # Calculate the intensity at these fine points
            intensity_at_fine_periods = interpolation_function(fine_periods)
            args.logger.debug("Done.")

            # --- 4. Calculate Mean Intensity (The value that corresponds to 'intensity') ---
            # The mean intensity in the 193-199 bp range is the core value used for correlation.
            mean_intensity_193_199 = np.mean(intensity_at_fine_periods)

            # Print or save the Fourier transformed coverage profile
            name=gene.attributes["gene_name"][0] if "gene_name" in gene.attributes else gene.attributes["gene_id"][0]
            print(f"{name}\t{mean_intensity_193_199}")
        
        pysamfile.close()