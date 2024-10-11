from logging import log
import pysam
import numpy as np
import pandas as pd
import argparse
import sys

base64_pca=""
base64_clfb_pca=""
base64_clfgt_pca="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"



#Collapse nucleotide sequence to Purine/Pyrimidine sequence
def nuc2purpyr(s):
    n2p={'A':'R','G':'R','C':'Y','T':'Y'} #R=purine / Y=Pyrimidine
    return "".join([n2p[c] for c in s])

def _5pends(args):
        
        #if args.reference==None:
        #    parser.error("Reference file is required.")
    
        cram=pysam.AlignmentFile(args.samfile[0],reference_filename=args.reference)
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
            
            if args.uselexsmallest:
                if s <= rcs:
                    kmers.append(s)
                    d[s]=0
            else:
                kmers.append(s)
                d[s]=0
        
        i=0
        for read in cram:
            
            if read.mapping_quality>=args.mapqual: #and read.flag(args.incflag)

                if not read.is_unmapped and not read.is_duplicate:

                    if args.useref:
                        if read.is_reverse:
                            s=fasta.fetch(read.reference_name,int(read.reference_end-k),int(read.reference_end)).translate(revcomptable)[::-1].upper()
                        else:
                            s=fasta.fetch(read.reference_name,int(read.reference_start),int(read.reference_start+k)).upper()
                    else:
                        s=read.query_sequence[:k].upper() if not read.is_reverse else read.query_sequence[-k:].translate(revcomptable)[::-1].upper()

                    # print("ref",fasta.fetch(read.reference_name,int(read.reference_start),int(read.reference_end)).upper())
                    # print("qry",read.query_sequence)
                    # print("reverse",read.is_reverse)

                    if 'N' not in s:
                        try:
                            if args.uselexsmallest: #only count lexigraphically smallest kmer
                                rcs=s.translate(revcomptable)[::-1]
                                d[s if s<rcs else rcs]+=1
                            else:
                                d[s]+=1
                            
                            i+=1
                        except KeyError: #skip when reads have other characters than ACGT
                            print("Err",s)
                            pass
            
            if args.maxo!=None:
                if i==args.maxo:
                    break
        
        if args.header:
            if args.name:
                sys.stdout.write("filename\t")
            sys.stdout.write("\t".join(map(str,list(d.keys()))) + "\n")
        
        if args.name:
            sys.stdout.write(args.samfile[0]+"\t")
        
        if args.norm=='freq':
            c=np.array(list(d.values()))
            f=c/c.sum()
            sys.stdout.write("\t".join(map(str,f)) + "\n")
        elif args.norm=='rpx':
            c=np.array(list(d.values()))
            f=(c/(c.sum()/args.x)).astype(int)
            sys.stdout.write("\t".join(map(str,f)) + "\n")
        else: #counts
            sys.stdout.write("\t".join(map(str,d.values())) + "\n")

def cleavesitemotifs(args, cmdline=True):
    
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
    
    c=np.array(list(d.values()))
    if args.norm=='freq':
        f=c/c.sum()
        
    elif args.norm=='rpx':
        f=(c/(c.sum()/args.x)).astype(int)
    else:
        f=c

    if not cmdline: #return values instead of printing
        return f

    if args.header:
        if args.name:
            sys.stdout.write("filename\t")
        sys.stdout.write("\t".join(map(str,list(d.keys()))) + "\n")

    if args.name:
        sys.stdout.write(args.samfile+"\t")

    sys.stdout.write("\t".join(map(str,f)) + "\n")

def bincounts(args, cmdline=True):    

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
        if args.name:
            sys.stdout.write("filename\t")

        sys.stdout.write("\t".join(h)+"\n")    

    if args.name:
        sys.stdout.write(args.samfile+"\t")

    if args.norm=='freq':
        sys.stdout.write("\t".join(map(str,np.array(v)/v.sum()))+"\n")
    elif args.norm=='rpx':
        sys.stdout.write("\t".join(map(str,(np.array(v)/(v.sum()/args.x))).astype(int))+"\n")
    else:
        sys.stdout.write("\t".join(map(str,v))+"\n")

def fszd(args, cmdline=True):
    
    cram=pysam.AlignmentFile(args.samfile,reference_filename=args.reference)
    
    fszd={}
    
    for fsz in range(args.lower,args.upper):
        fszd[fsz]=0
        
    i=0
    for read in cram:
        if read.mapping_quality>=args.mapqual and not read.is_unmapped and not read.is_duplicate:
            if args.insertissize:
                if read.template_length!=None and read.is_read2:
                    if abs(read.template_length)>=args.lower and abs(read.template_length)<args.upper:
                        fszd[abs(read.template_length)]+=1
                        i+=1
            else:
                if read.query_length>=args.lower and read.query_length<args.upper:
                    fszd[read.query_length]+=1
                i+=1
        
        if args.maxo!=None:
            if i==args.maxo:
                break
    
    if args.header:
        if args.name:
            sys.stdout.write("filename\t")        
        sys.stdout.write("\t".join(map(str,range(args.lower, args.upper)))+"\n")
    
    v=np.array([fszd[sz] for sz in range(args.lower,args.upper,1)])
    
    if not cmdline:
        return v

    if args.name:
        sys.stdout.write(args.samfile+"\t")
    
    if args.norm=='freq':    
        sys.stdout.write("\t".join(map(str,v/v.sum()))+"\n")
    elif args.norm=='rpx':
        sys.stdout.write("\t".join(map(str,(v/(v.sum()/args.x)).astype(int)))+"\n")
    else:
        sys.stdout.write("\t".join(map(str,v))+"\n")


def delfi(args, cmdline=True):
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
    
    if args.header:
        h=["filename" if args.name else None]
        for ref in bins_short:
            h+=[ref+'_delfi_'+str(x) for x in range(len(bins_short[ref]))]
    
    vshort,vlong=[],[]
    for ref in bins_short:
        vshort+=list(bins_short[ref])
        vlong+=list(bins_long[ref])
    
    if args.header:
        if args.name:
            sys.stdout.write("filename\t")
        sys.stdout.write("\t".join(h)+"\n")    
    
    if args.name:
        sys.stdout.write(args.samfile+"\t")
    
    sys.stdout.write("\t".join( map(str,(np.array(vshort)+1)/(np.array(vlong)+1) ) ) +"\n")

import sklearn
import pickle
import base64

def R206C(args):
    (pca,clf)=pickle.load(open(args.clf, 'rb'))

    args.k=4
    args.norm='rpx'
    args.x=1000000
    f=np.array(list(cleavesitemotifs(args, cmdline=False))).reshape(1,-1)

    clfgt=clf.predict(pca.transform(f))[0]
    classp=clf.predict_proba(pca.transform(f))[0]

    sys.stdout.write(f"R206C genotype: {clfgt} (0={classp[0]:.2f},1={classp[1]:.2f},2={classp[2]:.2f})\n")

parser = argparse.ArgumentParser(prog="cfstats", usage="cfstats -h", description="Gather cfDNA statistics", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def main():
    
    global_parser = argparse.ArgumentParser(add_help=False) #parser for arguments that apply to all subcommands

    global_parser.add_argument("-f", dest="inclflag", default=0, type=int, help="Sam file filter flag: only include reads that conform to this flag (like samtools -f option)")
    global_parser.add_argument("-F", dest="exclflag", default=0, type=int, help="Sam file filter flag: exclude reads that conform to this flag (like samtools -F option)")
    global_parser.add_argument("-q", dest="mapqual", default=60, type=int, help="Minimal mapping quality of reads to be considered (like samtools -q option)")
    global_parser.add_argument("-x", dest="x", default=1000000, type=int, help="Normalisation unit, see norm")
    global_parser.add_argument("--norm", dest="norm", choices=['counts','freq','rpx'], default='counts', help="Normalize: report counts, frequencies or reads per X reads (default x=1000000, set X with -x option).")

    global_parser.add_argument("-o", dest="maxo", default=None, type=int, help="Limit stats to maxo observations.")
    global_parser.add_argument("--header", dest="header", action="store_true", default=False, help="Write header for names of features")
    global_parser.add_argument("--name", dest="name", action="store_true", default=False, help="Prefix tab-separated values with the name of the file")
    global_parser.add_argument("-r", dest="reference", default=None, type=str, help="Reference file for: reference depended features cleave-site motifs/binned counts/cram decoding.")
    
    subparsers = parser.add_subparsers()
    
    parser_csm = subparsers.add_parser('csm',prog="cfstats csm", description="Extract k-length cleave-site motifs using the reference sequence at the 5' start/end of cfDNA fragments.", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_csm.add_argument('samfile', help='sam/bam/cram file')
    parser_csm.add_argument("-k", dest="k", default=4, type=int, help="Length of the cleave-site motifs.")
    parser_csm.set_defaults(func=cleavesitemotifs)
    
    parser_5pends = subparsers.add_parser('5pends',prog="cfstats 5pends", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_5pends.add_argument('samfile', nargs=1, help='sam/bam/cram file')
    parser_5pends.add_argument("-k", dest="k", default=4, type=int, help="Length of the 5' ends patterns.")
    parser_5pends.add_argument("--useref", action="store_true", dest="useref", default=False, help="Use reference sequence instead of read sequence.")
    parser_5pends.add_argument("--uselexsmallest", action="store_true", dest="uselexsmallest", default=False, help="Only count lexigraphically smallest kmer.")
    parser_5pends.set_defaults(func=_5pends)
    
    parser_bincounts = subparsers.add_parser('bincounts',prog="cfstats bincounts", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_bincounts.add_argument('samfile', help='sam/bam/cram file')
    parser_bincounts.add_argument("-b", "--binsize", dest="binsize", type=int, default=1000000, help="Size of the bins.")
    parser_bincounts.set_defaults(func=bincounts)

    parser_fszd = subparsers.add_parser('fszd',prog="cfstats fszd", description="Extract fragment size distribution (only for paired-end data)", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_fszd.add_argument('samfile', help='sam/bam/cram file')
    parser_fszd.add_argument('-l','--lower', default=60, type=int, help='Lower limit for fragments to report')
    parser_fszd.add_argument('-u','--upper', default=600, type=int, help='Upper limit for fragments to report')
    parser_fszd.add_argument("--noinsert", dest="insertissize", action="store_false", default=True, help="In case of long-read/unpaired sequencing infer fragmentsize from sequence instead of insert.")
    parser_fszd.set_defaults(func=fszd)
        
    parser_delfi = subparsers.add_parser('delfi',prog="cfstats delfi", description="Extract DELFI-like measure for bins of a predefined size (only for paired-end data)", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_delfi.add_argument('samfile', help='sam/bam/cram file')
    parser_delfi.add_argument("-b", "--binsize", dest="binsize", type=int, default=1000000, help="Size of the bins.")
    parser_delfi.add_argument('--short-lower', dest='shortlow', default=100, help='Definition of short fragments')
    parser_delfi.add_argument('--short-upper', dest='shortup', default=150, help='Definition of short fragments')
    parser_delfi.add_argument('--long-lower', dest='longlow', default=150, help='Definition of long fragments')
    parser_delfi.add_argument('--long-upper', dest='longup', default=200, help='Definition of short fragments')
    parser_delfi.add_argument("--noinsert", dest="insertissize", action="store_false", default=True, help="In case of long-read/unpaired sequencing infer fragmentsize from sequence instead of insert.")
    parser_delfi.set_defaults(func=delfi)


    parser_R206C = subparsers.add_parser('R206C',prog="cfstats R206C", description="Predict R206C genotype using fragmentomics", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_R206C.add_argument('samfile', help='sam/bam/cram file')
    parser_R206C.add_argument('clf', help='classifier file')
    parser_R206C.set_defaults(func=R206C)

    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
