
def worker_cleavesitemotifs(pl):
    samfile,args=pl
    cram = pysam.AlignmentFile(samfile, reference_filename=args.reference)

    if args.reference==None:
        parser.error("Reference file is required.")

    cram=pysam.AlignmentFile(samfile,reference_filename=args.reference)
    fasta=pysam.FastaFile(args.reference)
    
    k=args.k
    
    if args.purpyr:
        d={k:0 for k in allkp(k,onlylexsmallest=True)}
    else:
        d={k:0 for k in allk(k,onlylexsmallest=True)}        
    
    if args.maxo!=None:
        total_mapped_reads = sum([int(l.split("\t")[2]) for l in pysam.idxstats(cram.filename).split("\n")[:-1]])
        samplefrac=args.maxo/total_mapped_reads

    i=0
    for read in cram.fetch():
        if args.maxo!=None: #restrict to sample approximately maxo reads
            if random.random() > samplefrac:
                continue
    
        if args.reqflag != None:
            if read.flag & args.reqflag != args.reqflag:
                continue
        
        if args.exclflag != 0:
            if read.flag & args.exclflag != 0:
                continue
                    
        if read.mapping_quality<args.mapqual:
            continue
        
        if read.reference_start>int(k/2) and read.reference_end<cram.get_reference_length(read.reference_name)-int(k/2):
            if read.is_reverse:
                s=fasta.fetch(read.reference_name,int(read.reference_end-k/2),int(read.reference_end+k/2)).upper()
            else:
                s=fasta.fetch(read.reference_name,int(read.reference_start-k/2),int(read.reference_start+k/2)).upper()
            if 'N' not in s:
                try:
                    rcs=s.translate(revcomptable)[::-1]

                    if args.purpyr:
                        s=nuc2purpyr(s)
                        rcs=nuc2purpyr(rcs)
                    
                    d[s if s<rcs else rcs]+=1
                    i+=1
                except KeyError: #skip when reads have other characters than ACGT
                    print("Err",s)
                    pass

    result = {
        "samfile": samfile,
        "d": d
    }
    return result

def cleavesitemotifs(args, cmdline=True):
    
    v=[]
    with Pool(args.nproc) as pool:
        results = pool.map(worker_cleavesitemotifs, zip(args.samfiles, [args]*len(args.samfiles)))

    for result in results:
        samfile = result["samfile"]
        d = result["d"]

        if args.norm == 'freq':
            c = np.array(list(d.values()))
            f = c / c.sum()
        elif args.norm == 'rpx':
            c = np.array(list(d.values()))
            f = (c / (c.sum() / args.x)).astype(int)
        else:
            f = np.array(list(d.values()))

        if not cmdline:
            v.append(f)
            continue

        if args.header and samfile == args.samfiles[0]:
            if args.name:
                sys.stdout.write("filename\t")
            sys.stdout.write("\t".join(map(str, list(d.keys()))) + "\n")

        if args.name:
            sys.stdout.write(samfile + "\t")

        sys.stdout.write("\t".join(map(str, f)) + "\n")
    
    return v

def cleavesitemotifs_old(args, cmdline=True):
    
    for samfile in args.samfiles:
        if args.reference==None:
            parser.error("Reference file is required.")

        cram=pysam.AlignmentFile(samfile,reference_filename=args.reference)
        fasta=pysam.FastaFile(args.reference)
        
        k=args.k
        
        if args.purpyr:
            d={k:0 for k in allkp(k,onlylexsmallest=True)}
        else:
            d={k:0 for k in allk(k,onlylexsmallest=True)}        
        
        if args.maxo!=None:
            total_mapped_reads = sum([int(l.split("\t")[2]) for l in pysam.idxstats(cram.filename).split("\n")[:-1]])
            samplefrac=args.maxo/total_mapped_reads

        i=0
        for read in cram.fetch():
            if args.maxo!=None: #restrict to sample approximately maxo reads
                if random.random() > samplefrac:
                    continue
        
            if args.reqflag != None:
                if read.flag & args.reqflag != args.reqflag:
                    continue
            
            if args.exclflag != 0:
                if read.flag & args.exclflag != 0:
                    continue
                        
            if read.mapping_quality<args.mapqual:
                continue
            
            if read.reference_start>int(k/2) and read.reference_end<cram.get_reference_length(read.reference_name)-int(k/2):
                if read.is_reverse:
                    s=fasta.fetch(read.reference_name,int(read.reference_end-k/2),int(read.reference_end+k/2)).upper()
                else:
                    s=fasta.fetch(read.reference_name,int(read.reference_start-k/2),int(read.reference_start+k/2)).upper()
                if 'N' not in s:
                    try:
                        rcs=s.translate(revcomptable)[::-1]

                        if args.purpyr:
                            s=nuc2purpyr(s)
                            rcs=nuc2purpyr(rcs)
                        
                        d[s if s<rcs else rcs]+=1
                        i+=1
                    except KeyError: #skip when reads have other characters than ACGT
                        print("Err",s)
                        pass
        
        c=np.array(list(d.values()))
        if args.norm=='freq':
            f=c/c.sum()
        elif args.norm=='rpx':
            f=(c/(c.sum()/args.x)).astype(int)
        else:
            f=c

        if not cmdline: #return values instead of printing
            return f

        if args.header and samfile==args.samfiles[0]:
            if args.name:
                sys.stdout.write("filename\t")
            sys.stdout.write("\t".join(map(str,list(d.keys()))) + "\n")

        if args.name:
            sys.stdout.write(samfile+"\t")

        sys.stdout.write("\t".join(map(str,f)) + "\n")

def cleavesitemotifsbysize(args, cmdline=True):
    
    for samfile in args.samfiles:
        if args.reference==None:
            parser.error("Reference file is required.")

        cram=pysam.AlignmentFile(samfile,reference_filename=args.reference)
        fasta=pysam.FastaFile(args.reference)
        
        k=args.k
        
        if args.purpyr:
            d={k:{i:0 for i in range(args.lower,args.upper)} for k in allkp(k,onlylexsmallest=True)}
        else:
            d={k:{i:0 for i in range(args.lower,args.upper)} for k in allk(k,onlylexsmallest=True)}        
        
        if args.maxo!=None:
            total_mapped_reads = sum([int(l.split("\t")[2]) for l in pysam.idxstats(cram.filename).split("\n")[:-1]])
            samplefrac=args.maxo/total_mapped_reads

        i=0
        for read in cram.fetch():
            
            if args.reqflag != None:
                if read.flag & args.reqflag != args.reqflag:
                    continue
            
            if args.exclflag != 0:
                if read.flag & args.exclflag != 0:
                    continue
                        
            if read.mapping_quality<args.mapqual:
                continue
            
            if args.maxo!=None: #restrict to sample approximately maxo reads
                if random.random() > samplefrac:
                    continue
            
            if read.reference_start>int(k/2) and read.reference_end<cram.get_reference_length(read.reference_name)-int(k/2):
                if read.is_reverse:
                    s=fasta.fetch(read.reference_name,int(read.reference_end-k/2),int(read.reference_end+k/2)).upper()
                else:
                    s=fasta.fetch(read.reference_name,int(read.reference_start-k/2),int(read.reference_start+k/2)).upper()
                if 'N' not in s:
                    try:
                        rcs=s.translate(revcomptable)[::-1]

                        if args.purpyr:
                            s=nuc2purpyr(s)
                            rcs=nuc2purpyr(rcs)

                        if args.insertissize:
                            if abs(read.template_length)>=args.lower and abs(read.template_length)<args.upper:
                                d[s if s<rcs else rcs][abs(read.template_length)]+=1
                        else:
                            if read.query_length>=args.lower and read.query_length<args.upper:
                                d[s if s<rcs else rcs][read.query_length]+=1
                        
                        i+=1
                    except KeyError: #skip when reads have other characters than ACGT
                        print("Err",s)
                        pass
                
        m=pd.DataFrame(d)
        m.T.to_csv(sys.stdout,sep="\t",index=True)
