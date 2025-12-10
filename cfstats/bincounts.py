import pysam
import random
import sys


from cfstats import utils

from logging import log
import numpy as np
from multiprocessing import Pool
import pandas as pd

def worker_bincounts(pl):
    samfile,args=pl

    if args.reference==None:
        raise ValueError("Reference file is required.")

    cram=pysam.AlignmentFile(samfile,reference_filename=args.reference)
    fasta=pysam.FastaFile(args.reference)

    bins={}    
    refl={}
    for ref in fasta.references:
        refl[ref]=fasta.get_reference_length(ref)
        bins[ref]=np.zeros(int(refl[ref]/args.binsize)+1)
    
    if args.maxo!=None:
        total_mapped_reads = sum([int(l.split("\t")[2]) for l in pysam.idxstats(cram.filename).split("\n")[:-1]])
        samplefrac=args.maxo/total_mapped_reads

    for read in cram:
        if args.maxo!=None: #restrict to sample approximately maxo reads
            if random.random() > samplefrac:
                continue

        if args.reqflag != None:
            if read.flag & args.reqflag != args.reqflag:
                continue
        
        if args.exclflag != 0:
            if read.flag & args.exclflag != 0:
                continue
        
        if read.mapping_quality>=args.mapqual:
            if not read.is_unmapped and not read.is_duplicate:
                bins[read.reference_name][int(read.pos/args.binsize)]+=1
    
    return {'samfile':samfile, 'd':bins}

def bincounts(args, cmdline=True):
    if args.reference==None:
        raise ValueError("Reference file is required.")
    
    reflabels=[]
    #determine bin labels
    fasta=pysam.FastaFile(args.reference)
    refl={}
    for ref in fasta.references:
        refl[ref]=fasta.get_reference_length(ref)
        for bini in range(int(refl[ref]/args.binsize)+1):
            start=str(bini*args.binsize)
            end=str(((bini+1)*args.binsize if (bini+1)*args.binsize<refl[ref] else refl[ref]))
            reflabels.append("%s_%s_%s"%(ref,start,end))

    with Pool(args.nproc) as pool:
        results = pool.map(worker_bincounts, zip(args.samfiles, [args]*len(args.samfiles)))

    V=[]
    
    for i,result in enumerate(results):
        samfile = result["samfile"]
        bins = result["d"]
        
        v=[]
        for ref in bins:
            v+=list(bins[ref])
        
        v=np.array(v)

        if args.gccorrect:
            #gc correct
            dfcnt=pd.DataFrame([v], columns=reflabels)
            gc_content = utils.get_gc_content(dfcnt, args.reference)
            dfcnt_corrected = utils.gc_correct_counts(dfcnt, gc_content)
            v = dfcnt_corrected.iloc[0].values

        if not cmdline:
            V.append(v)
            continue
        else:
            if args.header and samfile==args.samfiles[0]:
                if args.name:
                    sys.stdout.write("filename\t")
                sys.stdout.write("\t".join(reflabels)+"\n")
            
            if args.name:
                sys.stdout.write(samfile+"\t")
            if args.norm=='freq':
                sys.stdout.write("\t".join(map(str,np.array(v)/v.sum()))+"\n")
            elif args.norm=='rpx':
                sys.stdout.write("\t".join(map(str,(np.array(v)/(v.sum()/args.x))).astype(int))+"\n")
            else:
                sys.stdout.write("\t".join(map(str,v))+"\n")

    if not cmdline:
        return reflabels, np.array(V)