import pysam
import random
import sys


from cfstats import utils

import logging
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

    # Opportunistically map BAM/CRAM reference names to FASTA names when they
    # differ only by a 'chr' prefix and the lengths match.
    log = logging.getLogger(__name__)
    fasta_refs = set(fasta.references)
    fasta_lengths = dict(zip(fasta.references, fasta.lengths))
    ref_alias = {}
    for cram_ref, cram_len in zip(cram.references, cram.lengths):
        if cram_ref in fasta_refs and fasta_lengths[cram_ref] == cram_len:
            continue
        candidates = ['chr' + cram_ref] if not cram_ref.startswith('chr') else [cram_ref[3:]]
        for alt in candidates:
            if alt in fasta_refs and fasta_lengths[alt] == cram_len:
                ref_alias[cram_ref] = alt
                break
    if ref_alias:
        log.warning("Reference names in %s differ from FASTA names by a 'chr' prefix; mapping %s", samfile, ref_alias)
    
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
                ref_name = ref_alias.get(read.reference_name, read.reference_name)
                if ref_name not in bins:
                    continue
                bins[ref_name][int(read.pos/args.binsize)]+=1
    
    return {'samfile':samfile, 'd':bins}

def iter_bincounts(args):
    log=logging.getLogger(__name__)
    if args.reference==None:
        raise ValueError("Reference file is required.")

    args.samfiles = utils.collect_bam_files(args.samfiles, getattr(args, 'bamlist', None))
    utils.require_sample_names(args)

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

    payload = zip(args.samfiles, [args]*len(args.samfiles))
    if args.nproc > 1:
        pool = Pool(args.nproc)
        results = pool.imap_unordered(worker_bincounts, payload)
    else:
        pool = None
        results = map(worker_bincounts, payload)

    try:
        for result in results:
            samfile = result["samfile"]
            bins = result["d"]

            v=[]
            for ref in bins:
                v+=list(bins[ref])

            v=np.array(v)

            if v.sum()<args.x:
                log.warn("Normalisation unit (x=%d) is smaller than total read count. Consider ignoring sample=%s"%(args.x,samfile))

            if args.gccorrect:
                #gc correct
                dfcnt=pd.DataFrame([v], columns=reflabels)
                gc_content = utils.get_gc_content(dfcnt, args.reference)
                dfcnt_corrected = utils.gc_correct_counts(dfcnt, gc_content, frac=args.frac)
                v = dfcnt_corrected.iloc[0].values.copy()
                v[v<0]=0 #make sure gc corrected counts can not become negative

            yield reflabels, samfile, v
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def bincounts(args, cmdline=True):
    V={}
    reflabels=[]

    for reflabels, samfile, v in iter_bincounts(args):
        if cmdline and args.header and not V:
            if args.name:
                sys.stdout.write("filename\t")
            sys.stdout.write("\t".join(reflabels)+"\n")
            sys.stdout.flush()

        V[samfile] = v
        if not cmdline:
            continue

        if args.name:
            sys.stdout.write(samfile+"\t")
        if args.norm=='freq':
            vnorm=(np.array(v)/v.sum()).astype(np.float64)
            sys.stdout.write("\t".join(map(str,vnorm))+"\n")
        elif args.norm=='rpx':
            if np.nansum(v)==0:
                vnorm=v
            else:
                vnorm=(np.array(v)/(np.nansum(v)/args.x)).astype(np.uint32)
            sys.stdout.write("\t".join(map(str,vnorm))+"\n")
        else:
            sys.stdout.write("\t".join(map(str,v))+"\n")
        sys.stdout.flush()

    if not cmdline:
        return reflabels, np.array([V[s] for s in args.samfiles])
