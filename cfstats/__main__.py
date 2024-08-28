import pysam
import numpy as np
import pandas as pd
import argparse
import sys

def _5pends(args):
    pass

def cleavesitemotifs(args):
    
    if args.reference==None:
        parser.error("Reference file is required.")

    cram=pysam.AlignmentFile(args.samfile,reference_filename=args.reference)
    fasta=pysam.FastaFile(args.reference)
    
    k=args.k
    
    revcomptable = str.maketrans("acgtACGT","tgcaTGCA")
    # n=int(wildcards.samplen) if wildcards.samplen!="ALL" else None
    
    kmers=[]
    d={}
    for i in range(4**k):
        s=""
        for j in range(k):
            s+="ACGT"[int(i/(4**(k-j-1)))%4]
    
        rcs=s.translate(revcomptable)[::-1]
    
        if s <= rcs:
            kmers.append(s)
            d[s]=0
    
    i=0
    for read in cram:
        
        if read.mapping_quality>=args.mapqual: #and read.flag(args.incflag)
        
            if not read.is_unmapped and not read.is_duplicate and read.reference_start>int(k/2) and read.reference_end<cram.get_reference_length(read.reference_name)-int(k/2):
                if read.is_reverse:
                    s=fasta.fetch(read.reference_name,int(read.reference_end-k/2),int(read.reference_end+k/2)).upper()
                else:
                    s=fasta.fetch(read.reference_name,int(read.reference_start-k/2),int(read.reference_start+k/2)).upper()
                if 'N' not in s:
                    try:
                        rcs=s.translate(revcomptable)[::-1]
                        d[s if s<rcs else rcs]+=1
                        i+=1
                    except KeyError: #skip when reads have other characters than ACGT
                        print("Err",s)
                        pass
        
        if args.maxo!=None:
            if i==args.maxo:
                break
    
    if args.header:
        sys.stdout.write("\t".join(map(str,list(d.keys()))) + "\n")
    
    if args.norm:
        c=np.array(list(d.values()))
        f=c/c.sum()
        sys.stdout.write("\t".join(map(str,f)) + "\n")
    else:
        sys.stdout.write("\t".join(map(str,d.values())) + "\n") 

def bincounts(args):    

    if args.reference==None:
        parser.error("Reference file is required.")

    cram=pysam.AlignmentFile(args.samfile,reference_filename=args.reference)
    fasta=pysam.FastaFile(args.reference)

    bins={}    
    refl={}
    for ref in fasta.references:
        refl[ref]=fasta.get_reference_length(ref)
        bins[ref]=np.zeros(int(refl[ref]/args.binsize)+1)
    
    for read in cram:
        if read.mapping_quality>=args.mapqual:
            if not read.is_unmapped and not read.is_duplicate:
                bins[read.reference_name][int(read.pos/args.binsize)]+=1
    
    if args.header:
        h=[]
        for ref in bins:
            h+=[ref+'_counts_'+str(x) for x in range(len(bins[ref]))]
    
    v=[]
    for ref in bins:
        v+=list(bins[ref])
    
    if args.header:
        sys.stdout.write("\t".join(h)+"\n")    
    
    if args.norm:
        sys.stdout.write("\t".join(map(str,np.array(v)/v.sum()))+"\n")
    else:
        sys.stdout.write("\t".join(map(str,v))+"\n")

def fszd(args):
    
    cram=pysam.AlignmentFile(args.samfile,reference_filename=args.reference)
    
    fszd={}
    
    for fsz in range(args.lower,args.upper):
        fszd[fsz]=0
    
    i=0
    for read in cram:
        if read.mapping_quality>=args.mapqual and not read.is_unmapped and not read.is_duplicate and read.is_read2:
            if read.template_length!=None:
                if abs(read.template_length)>args.lower and abs(read.template_length)<args.upper:
                    if abs(read.template_length)>=args.lower and abs(read.template_length)<args.upper:
                        fszd[abs(read.template_length)]+=1
                    i+=1
        if args.maxo!=None:
            if i==args.maxo:
                break
    
    if args.header:
        sys.stdout.write("\t".join(map(str,range(args.lower, args.upper))+"\n"))
    
    v=np.array([fszd[sz] for sz in range(args.lower,args.upper,1)])
    
    if args.norm:
        sys.stdout.write("\t".join(map(str,v/v.sum()))+"\n")
    else:
        sys.stdout.write("\t".join(map(str,v))+"\n")


def delfi(args):
    cram=pysam.AlignmentFile(args.samfile,reference_filename=args.reference)
    
    if args.reference==None:
        parser.error("Reference file is required.")
    
    fasta=pysam.FastaFile(args.reference)

    bins_short,bins_long={},{}    
    refl={}
    
    for ref in fasta.references:
        refl[ref]=fasta.get_reference_length(ref)
        bins_short[ref]=np.zeros(int(refl[ref]/args.binsize)+1)
        bins_long[ref]=np.zeros(int(refl[ref]/args.binsize)+1)
    
    for read in cram:
        if read.mapping_quality>=args.mapqual:
            if not read.is_unmapped and not read.is_duplicate and read.is_read2:
                if (abs(read.template_length)>args.shortlow and abs(read.template_length)<args.shortup):
                    bins_short[read.reference_name][int(read.pos/args.binsize)]+=1
                elif (abs(read.template_length)>args.longlow and abs(read.template_length)<args.longup):
                    bins_long[read.reference_name][int(read.pos/args.binsize)]+=1
    
    if args.header:
        h=[]
        for ref in bins_short:
            h+=[ref+'_delfi_'+str(x) for x in range(len(bins_short[ref]))]
    
    vshort,vlong=[],[]
    for ref in bins_short:
        vshort+=list(bins_short[ref])
        vlong+=list(bins_long[ref])
        
    if args.header:
        sys.stdout.write("\t".join(h)+"\n")    
    
    sys.stdout.write("\t".join( map(str,(np.array(vshort)+1)/(np.array(vlong)+1) ) ) +"\n")

parser = argparse.ArgumentParser(prog="cfstats", usage="cfstats -h", description="Gather cfDNA statistics", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def main():
    
    global_parser = argparse.ArgumentParser(add_help=False) #parser for arguments that apply to all subcommands

    global_parser.add_argument("-f", dest="inclflag", default=0, type=int, help="Sam file filter flag: only include reads that conform to this flag (like samtools -f option)")
    global_parser.add_argument("-F", dest="exclflag", default=0, type=int, help="Sam file filter flag: exclude reads that conform to this flag (like samtools -F option)")
    global_parser.add_argument("-q", dest="mapqual", default=60, type=int, help="Minimal mapping quality of reads to be considered (like samtools -q option)")
    global_parser.add_argument("--no-norm", dest="norm", action="store_false", default=True, help="Don't normalize: report absolute counts instead of frequencies")
    global_parser.add_argument("-o", dest="maxo", default=None, type=int, help="Limit stats to maxo observations.")
    global_parser.add_argument("--header", dest="header", action="store_true", default=False, help="Write header for names of features")
    global_parser.add_argument("-r", dest="reference", default=None, type=str, help="Reference file for: reference depended features cleave-site motifs/binned counts/cram decoding.")
    
    subparsers = parser.add_subparsers()
    
    parser_csm = subparsers.add_parser('csm',prog="cfstats csm", description="Extract k-length cleave-site motifs using the reference sequence at the 5' start/end of cfDNA fragments.", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_csm.add_argument('samfile', help='sam/bam/cram file')
    parser_csm.add_argument("-k", dest="k", default=4, type=int, help="Length of the cleave-site motifs.")
    parser_csm.set_defaults(func=cleavesitemotifs)
    
    parser_5pends = subparsers.add_parser('5pends',prog="cfstats 5pends", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_5pends.add_argument('samfile', nargs=1, help='sam/bam/cram file')
    parser_5pends.add_argument("-k", dest="k", default=4, type=int, help="Length of the 5' ends patterns.")
    parser_5pends.set_defaults(func=_5pends)
    
    parser_bincounts = subparsers.add_parser('bincounts',prog="cfstats bincounts", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_bincounts.add_argument('samfile', help='sam/bam/cram file')
    parser_bincounts.add_argument("-b", "--binsize", dest="binsize", type=int, default=1000000, help="Size of the bins.")
    parser_bincounts.set_defaults(func=bincounts)

    parser_fszd = subparsers.add_parser('fszd',prog="cfstats fszd", description="Extract fragment size distribution (only for paired-end data)", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_fszd.add_argument('samfile', help='sam/bam/cram file')
    parser_fszd.add_argument('-l','--lower', default=60, help='Lower limit for fragments to report')
    parser_fszd.add_argument('-u','--upper', default=600, help='Upper limit for fragments to report')
    parser_fszd.set_defaults(func=fszd)
        
    parser_delfi = subparsers.add_parser('delfi',prog="cfstats delfi", description="Extract DELFI-like measure for bins of a predefined size (only for paired-end data)", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_delfi.add_argument('samfile', help='sam/bam/cram file')
    parser_delfi.add_argument("-b", "--binsize", dest="binsize", type=int, default=1000000, help="Size of the bins.")
    parser_delfi.add_argument('--short-lower', dest='shortlow', default=100, help='Definition of short fragments')
    parser_delfi.add_argument('--short-upper', dest='shortup', default=150, help='Definition of short fragments')
    parser_delfi.add_argument('--long-lower', dest='longlow', default=150, help='Definition of long fragments')
    parser_delfi.add_argument('--long-upper', dest='longup', default=200, help='Definition of short fragments')
    parser_delfi.set_defaults(func=delfi)

    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
