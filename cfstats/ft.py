import numpy as np
import pysam
import gffutils
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
from scipy.interpolate import interp1d
import os
from logging import log


def compute_protection(bam, chrom, start, end):
    """
    Compute protection score in [start, end] (1-based, inclusive).
    Replace logic with your definition.
    """
    # Get depth array
    depth = np.zeros(end - start + 1, dtype=int)

    for pileupcol in bam.pileup(
        chrom, start-1, end, truncate=True
    ):
        pos = pileupcol.reference_pos
        if start <= pos+1 <= end:
            depth[pos - start] = pileupcol.nsegments

    # example: protection = negative slope of depth or low coverage region
    return depth.mean()  # placeholder


def fourier_transform_coverage(args):

    for samfile in args.samfiles:
        # Open the SAM/BAM/CRAM file
        pysamfile = pysam.AlignmentFile(samfile, "rb", reference_filename=args.reference if args.reference!=None else None)
        samctgs=set([ctg for ctg in pysamfile.references])
        # pysamfile.close()

        # Load the GFF file
        db_filename = f'{args.gfffile}.db'
        if not os.path.exists(db_filename):
            log(20,"Constructing gene DB...")
            db = gffutils.create_db(args.gfffile, dbfn=db_filename, force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)
            log(20,"Done.")
        else:
            db = gffutils.FeatureDB(db_filename, keep_order=True)
        
        

        # Iterate over each gene in the GFF file
        for gene in db.features_of_type('gene'):
            # Get the coverage profile for the gene
            # coverage = np.zeros(gene.end - gene.start + 1)
            coverage = np.zeros(10000+1)

            if gene.chrom not in samctgs:
                if 'chr'+gene.chrom in samctgs:
                    chrom='chr'+gene.chrom
                else:
                    log(20,f'{gene.chrom} not in samfile, skipping {gene}.')
                    continue
            else:
                chrom=gene.chrom
            
            if gene.strand=='-':
                start=gene.end-10000
                end=gene.end
            else:
                start=gene.start
                end=gene.start+10000

            # print(f'{gene.chrom} {gene.start} {gene.end} {start} {end}.')

            # pysamfile = pysam.AlignmentFile(samfile, "rb", reference_filename=args.reference if args.reference!=None else None)

            # for pileupcolumn in pysamfile.pileup(chrom, start, end):
            #     if pileupcolumn.pos < start or pileupcolumn.pos > end:
            #         continue
            #     coverage[pileupcolumn.pos - start] = pileupcolumn.nsegments

            for read in pysamfile.fetch(chrom, start, end):
                for ref_pos in read.get_reference_positions():
                    pos = ref_pos + 1  # 1-based
                    if start <= pos <= end:
                        coverage[pos - start] += 1

            # Perform Fourier transform on the coverage profile
            # fourier_transformed = fft(coverage)
            
            signal = coverage

            frequencies, power_spectrum = periodogram(signal, fs=1, scaling='spectrum')

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
            interpolation_function = interp1d(target_periods, target_intensity)

            # Create a fine array of periods in the target range
            fine_periods = np.linspace(P_MIN, P_MAX, 100)
            # Calculate the intensity at these fine points
            intensity_at_fine_periods = interpolation_function(fine_periods)

            # --- 4. Calculate Mean Intensity (The value that corresponds to 'intensity') ---
            # The mean intensity in the 193-199 bp range is the core value used for correlation.
            mean_intensity_193_199 = np.mean(intensity_at_fine_periods)
 
            # Print or save the Fourier transformed coverage profile
            name=gene.attributes["gene_name"][0] if "gene_name" in gene.attributes else gene.attributes["gene_id"][0]
            print(f"{name}\t{mean_intensity_193_199}")
        
        pysamfile.close()