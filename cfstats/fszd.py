import pysam
import random
import sys


from cfstats import utils

from logging import log
import numpy as np
from multiprocessing import Pool


def worker_fszd(pl):
    samfile,args=pl
    cram = pysam.AlignmentFile(samfile, reference_filename=args.reference)
    fszd = {}
    for fsz in range(args.lower, args.upper):
        fszd[fsz] = 0
    if args.maxo != None:
        total_mapped_reads = sum([int(l.split("\t")[2]) for l in pysam.idxstats(cram.filename).split("\n")[:-1]])
        samplefrac = args.maxo / total_mapped_reads
    i = 0
    for read in cram:
        if args.maxo != None:
            if random.random() > samplefrac:
                continue
        if args.reqflag != None:
            if read.flag & args.reqflag != args.reqflag:
                continue
        if args.exclflag != 0:
            if read.flag & args.exclflag != 0:
                continue
        if read.mapping_quality >= args.mapqual:
            if args.insertissize:
                if read.template_length != None and read.is_read2 and (read.is_reverse != read.mate_is_reverse):
                    if abs(read.template_length) >= args.lower and abs(read.template_length) < args.upper:
                        fszd[abs(read.template_length)] += 1
                        i += 1
            else:
                if read.query_length >= args.lower and read.query_length < args.upper:
                    fszd[read.query_length] += 1
                i += 1

    result = {
        "samfile": samfile,
        "fszd": fszd
    }
    return result


def fszd(args, cmdline=True):

    args.samfiles = utils.collect_bam_files(args.samfiles, getattr(args, 'bamlist', None))
    utils.require_sample_names(args)

    V={}

    if cmdline and args.header:
        if args.name:
            sys.stdout.write("filename\t")
        sys.stdout.write("\t".join(map(str, range(args.lower, args.upper))) + "\n")
        sys.stdout.flush()

    def _iter_results():
        payload = zip(args.samfiles, [args]*len(args.samfiles))
        if args.nproc > 1:
            with Pool(args.nproc) as pool:
                yield from pool.imap_unordered(worker_fszd, payload)
        else:
            for pl in payload:
                yield worker_fszd(pl)

    for result in _iter_results():
        samfile = result["samfile"]
        fszd = result["fszd"]

        if args.norm == 'freq':
            c = np.array([fszd[sz] for sz in range(args.lower, args.upper, 1)])
            v = c / c.sum()
        elif args.norm == 'rpx':
            c = np.array([fszd[sz] for sz in range(args.lower, args.upper, 1)])
            v = (c / (c.sum() / args.x)).astype(int)
        else:
            v = np.array([fszd[sz] for sz in range(args.lower, args.upper, 1)])

        if not cmdline:
            V[samfile] = v
            continue

        if args.name:
            sys.stdout.write(samfile + "\t")
        sys.stdout.write("\t".join(map(str, v)) + "\n")
        sys.stdout.flush()

    if not cmdline:
        # Preserve input order for programmatic callers.
        return [V[s] for s in args.samfiles]

def fszd_old(args, cmdline=True):
    
    for samfile in args.samfiles:

        cram=pysam.AlignmentFile(samfile,reference_filename=args.reference)
        
        fszd={}
        
        for fsz in range(args.lower,args.upper):
            fszd[fsz]=0
        
        if args.maxo!=None:
            # Obtain total reads in CRAM from index
            #print(pysam.idxstats(cram.filename))
            total_mapped_reads = sum([int(l.split("\t")[2]) for l in pysam.idxstats(cram.filename).split("\n")[:-1]])
            samplefrac=args.maxo/total_mapped_reads

        i=0
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
            
        #if not read.is_proper_pair or read.is_secondary or read.is_supplementary:
            # continue

            if read.mapping_quality>=args.mapqual:
                if args.insertissize:
                    if read.template_length!=None and read.is_read2 and (read.is_reverse != read.mate_is_reverse): #only count read2, and only if it is on the reverse strand
                        if abs(read.template_length)>=args.lower and abs(read.template_length)<args.upper:
                            fszd[abs(read.template_length)]+=1
                            i+=1
                else:
                    if read.query_length>=args.lower and read.query_length<args.upper:
                        fszd[read.query_length]+=1
                    i+=1
        
        c=np.array([fszd[sz] for sz in range(args.lower,args.upper,1)])
        
        if args.norm=='freq':
            v=c/c.sum()
        elif args.norm=='rpx':
            v=(c/(c.sum()/args.x)).astype(int)
        else:
            v=c

        if not cmdline:
            return v

        if args.header and samfile==args.samfiles[0] and cmdline:
            if args.name:
                sys.stdout.write("filename\t")        
            sys.stdout.write("\t".join(map(str,range(args.lower, args.upper)))+"\n")

        if args.name:
            sys.stdout.write(samfile+"\t")
        
        # if args.norm=='freq':    
        #     sys.stdout.write("\t".join(map(str,v/v.sum()))+"\n")
        # elif args.norm=='rpx':
        #     sys.stdout.write("\t".join(map(str,(v/(v.sum()/args.x)).astype(int)))+"\n")
        # else:
        sys.stdout.write("\t".join(map(str,v))+"\n")

        sys.stderr.write(f"Processed {i} read pairs in {samfile}")
