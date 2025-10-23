import pysam
import random
import sys


from cfstats import utils

from logging import log
import numpy as np
from multiprocessing import Pool


def worker_5pends(pl):
    samfile,args=pl
    cram = pysam.AlignmentFile(samfile, reference_filename=args.reference)

    if args.reference:
        fasta = pysam.FastaFile(args.reference)

    k = args.k

    revcomptable = str.maketrans("acgtACGT", "tgcaTGCA")

    kmers = []
    d = {}
    for i in range(4**k):
        s = ""
        for j in range(k):
            s += "ACGT"[int(i / (4**(k - j - 1))) % 4]

        rcs = s.translate(revcomptable)[::-1]

        if args.uselexsmallest:
            if s <= rcs:
                kmers.append(s)
                d[s] = 0
        else:
            kmers.append(s)
            d[s] = 0

    if args.maxo is not None:
        total_mapped_reads = sum([int(l.split("\t")[2]) for l in pysam.idxstats(cram.filename).split("\n")[:-1]])
        samplefrac = args.maxo / total_mapped_reads

    i = 0
    for read in cram:
        if args.maxo is not None:
            if random.random() > samplefrac:
                continue

        if args.reqflag is not None:
            if read.flag & args.reqflag != args.reqflag:
                continue

        if args.exclflag != 0:
            if read.flag & args.exclflag != 0:
                continue

        if read.mapping_quality >= args.mapqual and len(read.query_sequence) > args.k * 2:
            if not read.is_duplicate:
                if args.useref:
                    if read.is_reverse:
                        s = fasta.fetch(read.reference_name, int(read.reference_end - k), int(read.reference_end)).translate(revcomptable)[::-1].upper()
                    else:
                        s = fasta.fetch(read.reference_name, int(read.reference_start), int(read.reference_start + k)).upper()
                else:
                    s = read.query_sequence[:k].upper() if not read.is_reverse else read.query_sequence[-k:].translate(revcomptable)[::-1].upper()

                if 'N' not in s:
                    try:
                        if args.uselexsmallest:
                            rcs = s.translate(revcomptable)[::-1]
                            d[s if s < rcs else rcs] += 1
                        else:
                            d[s] += 1

                        i += 1
                    except KeyError:
                        print("Err", s)
                        pass

    result = {
        "samfile": samfile,
        "d": d
    }
    return result


def _5pends(args, cmdline=True):
    
    v=[]
    with Pool(args.nproc) as pool:
        results = pool.map(worker_5pends, zip(args.samfiles, [args]*len(args.samfiles)))

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

def _5pendsbysize(args, cmdline=True):
    
    k=args.k
    
    if args.purpyr:
        d={k:{i:0 for i in range(args.lower,args.upper)} for k in allkp(k,onlylexsmallest=args.uselexsmallest)}
    else:
        d={k:{i:0 for i in range(args.lower,args.upper)} for k in allk(k,onlylexsmallest=args.uselexsmallest)}        

    if args.useref:
        if args.reference==None:
            parser.error("Reference file is required.")
        fasta=pysam.FastaFile(args.reference)

    for samfile in args.samfiles:
        cram=pysam.AlignmentFile(samfile,reference_filename=args.reference)
        
        
        if args.maxo!=None:
            total_mapped_reads = sum([int(l.split("\t")[2]) for l in pysam.idxstats(cram.filename).split("\n")[:-1]])
            samplefrac=args.maxo/total_mapped_reads

        i=0
        for read in cram:
            
            if args.reqflag != None:
                if read.flag & args.reqflag != args.reqflag:
                    continue
            
            if args.exclflag != 0:
                if read.flag & args.exclflag != 0:
                    continue

            if args.maxo!=None: #restrict to sample approximately maxo reads
                if random.random() > samplefrac:
                    continue

            if read.mapping_quality>=args.mapqual and len(read.query_sequence)>args.k*2: #and read.flag(args.incflag)
                
                if not read.is_duplicate:

                    if args.useref:
                        
                        if read.is_reverse:
                            s=fasta.fetch(read.reference_name,int(read.reference_end-k),int(read.reference_end)).translate(revcomptable)[::-1].upper()
                        else:
                            s=fasta.fetch(read.reference_name,int(read.reference_start),int(read.reference_start+k)).upper()
                    else:
                        s=read.query_sequence[:k].upper() if not read.is_reverse else read.query_sequence[-k:].translate(revcomptable)[::-1].upper()

                    rcs=s.translate(revcomptable)[::-1]

                    if args.purpyr:
                        s=nuc2purpyr(s)
                        rcs=nuc2purpyr(rcs)
                    
                    if 'N' not in s:
                        try:
                            if args.insertissize:
                                sz=abs(read.template_length)
                            else:
                                sz=read.query_length

                            if sz>=args.lower and sz<args.upper:
                                if args.uselexsmallest: #only count lexigraphically smallest kmer, reduce size of d
                                    d[s if s<rcs else rcs][sz]+=1
                                else:
                                    d[s][sz]+=1
                            
                            i+=1
                        except KeyError: #skip when reads have other characters than ACGT or are not in the dictionary for another reason
                            print("Err",s,sz)
                            pass
        
        m=pd.DataFrame(d)
        m.T.to_csv(sys.stdout,sep="\t",index=True)
