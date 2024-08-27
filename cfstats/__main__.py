import pysam
import numpy as np
import pandas as pd
import argparse

def _5pends(args):
    pass

def cleavesitemotifs(args):
    cramfilename=args.samfile
    reference="hg38flat.fa"
    
    cram=pysam.AlignmentFile(cramfilename,reference_filename=reference)
    fasta=pysam.FastaFile(reference)
    
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
        if not read.is_unmapped and not read.is_duplicate and read.mapq==60 and read.reference_start>int(k/2) and read.reference_end<cram.get_reference_length(read.reference_name)-int(k/2):
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
    
    print("\t".join(d.values()))

def bincounts(args):
    pass

def main():
    parser = argparse.ArgumentParser(prog="cfstats", usage="cfstats -h", description="Gather cfDNA statistics", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    global_parser = argparse.ArgumentParser(add_help=False) #parser for arguments that apply to all subcommands
    
    subparsers = parser.add_subparsers()
    
    parser_csm = subparsers.add_parser('5pends',prog="cfstats 5pends", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_csm.add_argument('samfile', nargs=1, help='')
    parser_csm.add_argument("-k", dest="k", default=4, type=int, help="Length of the 5' ends patterns.")
    parser_csm.set_defaults(func=cleavesitemotifs)
    
    parser_5pends = subparsers.add_parser('5pends',prog="cfstats 5pends", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_5pends.add_argument('samfile', nargs=1, help='')
    parser_5pends.add_argument("-k", dest="k", default=4, type=int, help="Length of the 5' ends patterns.")
    parser_5pends.set_defaults(func=_5pends)
    
    parser_bincounts = subparsers.add_parser('bincounts',prog="cfstats bincounts", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_bincounts.add_argument('samfile', nargs=1, help='')
    parser_bincounts.add_argument("-b", "--binsize", dest="binsize", default=1000000, help="Size of the bins.")
    parser_bincounts.set_defaults(func=bincounts)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
