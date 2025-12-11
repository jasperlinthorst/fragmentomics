import numpy as np
import pysam
import gffutils
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
from scipy.interpolate import interp1d
import os
from logging import log

def wps(bam_file, chromosome, start_query, end_query, k=120, min_len=120, max_len=180):
    """
    Calculates the Windowed Protection Score (WPS) for every position in the query region
    using a sliding window approach with aligned reads from a BAM file.

    The window size k defaults to 120 bp (L-WPS), appropriate for nucleosome analysis 
    using fragments in the 120–180 bp range [2].

    Args:
        bam_file_path (str): Path to the alignment file (BAM).
        chromosome (str): Chromosome name (e.g., 'chr1').
        start_query (int): Start coordinate of the query region (1-based inclusive).
        end_query (int): End coordinate of the query region (1-based exclusive).
        k (int): The window size k (120 for L-WPS, 16 for S-WPS) [2].
        min_len (int): Minimum fragment length to consider (e.g., 120 for L-WPS).
        max_len (int): Maximum fragment length to consider (e.g., 180 for L-WPS).

    Returns:
        numpy.ndarray: Array of WPS scores for each position in the query region.
    """
    
    # Calculate the length of the query region
    region_length = end_query - start_query
    
    # Initialize result array
    wps_scores = np.zeros(region_length, dtype=int)
    
    # Pre-fetch all relevant reads in the extended region
    fetch_start = max(0, start_query - k)
    fetch_end = end_query + k
    
    # Store fragment information for efficient processing
    fragments = []
    
    try:
        # Iterate over all reads overlapping the extended region
        for read in bam_file.fetch(chromosome, fetch_start, fetch_end):
            
            # We only process the first read in a pair to avoid double counting fragments
            # and must ensure the read is properly paired and mapped.
            if read.is_paired and read.is_proper_pair and read.is_read1:
                
                # Get fragment length (template length, tlen)
                template_length = abs(read.template_length)
                
                # Filter reads based on length criteria (e.g., 120-180 bp for L-WPS)
                if template_length < min_len or template_length > max_len:
                    continue
                
                # Determine fragment coordinates for paired reads
                # Use template_length which represents the insert size (distance from start of read1 to end of read2)
                # For properly paired reads, the fragment spans from the leftmost alignment to the rightmost alignment
                if read.template_length > 0:
                    # Read1 is leftmost, read2 is rightmost
                    frag_start = read.reference_start
                    frag_end = read.reference_start + read.template_length
                else:
                    # Read1 is rightmost, read2 is leftmost
                    frag_start = read.reference_start + read.template_length  # template_length is negative
                    frag_end = read.reference_start
                
                fragments.append((frag_start, frag_end))
                        
    except Exception as e:
        print(f"Error processing BAM file: {e}")
        return np.full(region_length, np.nan)
    
    # Calculate WPS for each position using sliding window
    for pos in range(region_length):
        # Convert to 0-based genomic coordinates
        genomic_pos = start_query + pos
        
        # Define window boundaries (0-based inclusive, exclusive)
        start_window = genomic_pos - k // 2
        end_window = genomic_pos + k // 2
        
        spanning_count = 0
        endpoint_within_count = 0
        
        # Count fragments for this window
        for frag_start, frag_end in fragments:
            # Check for Fragments COMPLETELY SPANNING the window
            if (frag_start <= start_window) and (frag_end >= end_window):
                spanning_count += 1
            
            # Check for Fragments with an ENDPOINT WITHIN the window
            # Check fragment start endpoint
            if (frag_start >= start_window) and (frag_start < end_window):
                endpoint_within_count += 1
            
            # Check fragment end endpoint
            if (frag_end - 1 >= start_window) and (frag_end - 1 < end_window):
                # Ensure we don't double count single-base fragments
                if frag_start != (frag_end - 1):
                    endpoint_within_count += 1
        
        # Calculate WPS for this position
        wps_scores[pos] = spanning_count - endpoint_within_count
    
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