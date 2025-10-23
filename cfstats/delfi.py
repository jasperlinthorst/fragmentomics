import pysam
import sys

def delfi(args, cmdline=True):

    for samfile in args.samfiles:
        cram=pysam.AlignmentFile(samfile,reference_filename=args.reference)
        
        if args.reference==None:
            parser.error("Reference file is required.")
        
        fasta=pysam.FastaFile(args.reference)

        bins_short,bins_long={},{}    
        refl={}
        
        for ref in fasta.references:
            refl[ref]=fasta.get_reference_length(ref)
            bins_short[ref]=np.zeros(int(refl[ref]/args.binsize)+1)
            bins_long[ref]=np.zeros(int(refl[ref]/args.binsize)+1)
        
        if args.maxo!=None:
            # Obtain total reads in CRAM from index
            #print(pysam.idxstats(cram.filename))
            total_mapped_reads = sum([int(l.split("\t")[2]) for l in pysam.idxstats(cram.filename).split("\n")[:-1]])
            samplefrac=args.maxo/total_mapped_reads
        
        for read in cram:
            if read.is_unmapped:
                continue
            
            if args.maxo!=None: #restrict to sample approximately maxo reads
                if random.random() > samplefrac:
                    continue
            
            if read.mapping_quality>=args.mapqual:
                if args.insertissize:
                    if not read.is_unmapped and not read.is_duplicate and read.is_read2:
                        if (abs(read.template_length)>args.shortlow and abs(read.template_length)<args.shortup):
                            bins_short[read.reference_name][int(read.pos/args.binsize)]+=1
                        elif (abs(read.template_length)>args.longlow and abs(read.template_length)<args.longup):
                            bins_long[read.reference_name][int(read.pos/args.binsize)]+=1
                else:
                    if not read.is_unmapped and not read.is_duplicate:
                        if (abs(read.query_length)>args.shortlow and abs(read.query_length)<args.shortup):
                            bins_short[read.reference_name][int(read.pos/args.binsize)]+=1
                        elif (abs(read.query_length)>args.longlow and abs(read.query_length)<args.longup):
                            bins_long[read.reference_name][int(read.pos/args.binsize)]+=1
        
        if args.header and samfile==args.samfiles[0]:
            h=["filename"] if args.name else []
            for ref in bins_short:
                h+=[ref+'_delfi_'+str(x) for x in range(len(bins_short[ref]))]
            sys.stdout.write("\t".join(h)+"\n")  
        
        vshort,vlong=[],[]
        for ref in bins_short:
            vshort+=list(bins_short[ref])
            vlong+=list(bins_long[ref])
        
        if args.name:
            sys.stdout.write(samfile+"\t")
        
        sys.stdout.write("\t".join( map(str,(np.array(vshort)+1)/(np.array(vlong)+1) ) ) +"\n")
