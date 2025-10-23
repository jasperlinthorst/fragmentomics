
def fourier_transform_coverage(args):
    import gffutils
    from scipy.fft import fft

    for samfile in args.samfiles:
        # Open the SAM/BAM/CRAM file
        pysamfile = pysam.AlignmentFile(samfile, "rb")
        
        # Load the GFF file
        db_filename = f'{args.gfffile}.db'
        if not os.path.exists(db_filename):
            db = gffutils.create_db(args.gfffile, dbfn=db_filename, force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)
        else:
            db = gffutils.FeatureDB(db_filename, keep_order=True)
        
        # Iterate over each gene in the GFF file
        for gene in db.features_of_type('gene'):
            # Get the coverage profile for the gene
            coverage = np.zeros(gene.end - gene.start + 1)
            for pileupcolumn in pysamfile.pileup(gene.chrom, gene.start, gene.end):
                if pileupcolumn.pos < gene.start or pileupcolumn.pos > gene.end:
                    continue
                coverage[pileupcolumn.pos - gene.start] = pileupcolumn.nsegments
            
            # Perform Fourier transform on the coverage profile
            fourier_transformed = fft(coverage)
            
            # Print or save the Fourier transformed coverage profile
            print(f"Gene: {gene.id}, Fourier Transformed Coverage: {fourier_transformed}")
        
        pysamfile.close()
