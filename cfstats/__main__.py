from logging import log
import pysam
import numpy as np
import pandas as pd
import argparse
import sys

#Collapse nucleotide sequence to Purine/Pyrimidine sequence
def nuc2purpyr(s):
    n2p={'A':'R','G':'R','C':'Y','T':'Y'} #R=purine / Y=Pyrimidine
    return "".join([n2p[c] for c in s])

def _5pends(args):
        
        if args.reference==None:
            parser.error("Reference file is required.")
    
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
        
        if args.norm:
            c=np.array(list(d.values()))
            f=c/c.sum()
            sys.stdout.write("\t".join(map(str,f)) + "\n")
        else:
            sys.stdout.write("\t".join(map(str,d.values())) + "\n")

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
        if args.name:
            sys.stdout.write("filename\t")
        sys.stdout.write("\t".join(map(str,list(d.keys()))) + "\n")
    
    if args.name:
        sys.stdout.write(args.samfile+"\t")
    
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
        if args.name:
            sys.stdout.write("filename\t")

        sys.stdout.write("\t".join(h)+"\n")    

    if args.name:
        sys.stdout.write(args.samfile+"\t")

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
        if read.mapping_quality>=args.mapqual and not read.is_unmapped and not read.is_duplicate:
            
            if args.insertissize:
                if read.template_length!=None and read.is_read2:
                    if abs(read.template_length)>args.lower and abs(read.template_length)<args.upper:
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
    
    v=np.array([fszd[sz]+1 for sz in range(args.lower,args.upper,1)])
    
    if args.name:
        sys.stdout.write(args.samfile+"\t")
    
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

parser = argparse.ArgumentParser(prog="cfstats", usage="cfstats -h", description="Gather cfDNA statistics", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def main():
    
    global_parser = argparse.ArgumentParser(add_help=False) #parser for arguments that apply to all subcommands

    global_parser.add_argument("-f", dest="inclflag", default=0, type=int, help="Sam file filter flag: only include reads that conform to this flag (like samtools -f option)")
    global_parser.add_argument("-F", dest="exclflag", default=0, type=int, help="Sam file filter flag: exclude reads that conform to this flag (like samtools -F option)")
    global_parser.add_argument("-q", dest="mapqual", default=60, type=int, help="Minimal mapping quality of reads to be considered (like samtools -q option)")
    global_parser.add_argument("--no-norm", dest="norm", action="store_false", default=True, help="Don't normalize: report absolute counts instead of frequencies")
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

    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
