from logging import log
import pysam
import numpy as np
import pandas as pd
import argparse
import sys
import os 
import random

revcomptable = str.maketrans("acgtACGTRY","tgcaTGCAYR")
def revcomp(s):
    return s.translate(revcomptable)[::-1]

def allk(k,onlylexsmallest=False):
    kmers=[]
    for i in range(4**k):
        s=""
        for j in range(k):
            s+="ACGT"[int(i/(4**(k-j-1)))%4]
        if onlylexsmallest:
            if s<=revcomp(s):
                kmers.append(s)
        else:
            kmers.append(s)
    return kmers

def allkp(k,onlylexsmallest=False):
    kpmers=[]
    for i in range(2**k):
        s=""
        for j in range(k):
            s+="RY"[int(i/(2**(k-j-1)))%2]
        if onlylexsmallest:
            if s<=revcomp(s):
                kpmers.append(s)
        else:
            kpmers.append(s)
    return kpmers

#Collapse nucleotide sequence to Purine/Pyrimidine sequence
def nuc2purpyr(s):
    n2p={'A':'R','G':'R','C':'Y','T':'Y'} #R=purine / Y=Pyrimidine
    return "".join([n2p[c] for c in s])

def fourier_transform_coverage(args):
    import gffutils
    from scipy.fft import fft

    for samfile in args.samfiles:
        # Open the SAM/BAM/CRAM file
        pysamfile = pysam.AlignmentFile(samfile, "rb")
        
        # Load the GFF file
        db_filename = f'{args.gfffile}.db'
        if not os.path.exists(db_filename):
            db = gffutils.create_db(args.gfffile, dbfn=db_filename, force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)
        else:
            db = gffutils.FeatureDB(db_filename, keep_order=True)
        
        # Iterate over each gene in the GFF file
        for gene in db.features_of_type('gene'):
            # Get the coverage profile for the gene
            coverage = np.zeros(gene.end - gene.start + 1)
            for pileupcolumn in pysamfile.pileup(gene.chrom, gene.start, gene.end):
                if pileupcolumn.pos < gene.start or pileupcolumn.pos > gene.end:
                    continue
                coverage[pileupcolumn.pos - gene.start] = pileupcolumn.nsegments
            
            # Perform Fourier transform on the coverage profile
            fourier_transformed = fft(coverage)
            
            # Print or save the Fourier transformed coverage profile
            print(f"Gene: {gene.id}, Fourier Transformed Coverage: {fourier_transformed}")
        
        pysamfile.close()

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

        if not cmdline:
            v.append(list(d.values()))
            continue

        if args.header and samfile == args.samfiles[0]:
            if args.name:
                sys.stdout.write("filename\t")
            sys.stdout.write("\t".join(map(str, list(d.keys()))) + "\n")

        if args.name:
            sys.stdout.write(samfile + "\t")

        if args.norm == 'freq':
            c = np.array(list(d.values()))
            f = c / c.sum()
            sys.stdout.write("\t".join(map(str, f)) + "\n")
        elif args.norm == 'rpx':
            c = np.array(list(d.values()))
            f = (c / (c.sum() / args.x)).astype(int)
            sys.stdout.write("\t".join(map(str, f)) + "\n")
        else:
            sys.stdout.write("\t".join(map(str, d.values())) + "\n")
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

        # c=np.array(list(d.values()))
        # if args.norm=='freq':
        #     f=c/c.sum()
            
        # elif args.norm=='rpx':
        #     f=(c/(c.sum()/args.x)).astype(int)
        # else:
        #     f=c

        # if not cmdline: #return values instead of printing
        #     return f

        # if args.header and samfile==args.samfiles[0]:
        #     if args.name:
        #         sys.stdout.write("filename\t")
        #     sys.stdout.write("\t".join(map(str,list(d.keys()))) + "\n")

        # if args.name:
        #     sys.stdout.write(samfile+"\t")

        # sys.stdout.write("\t".join(map(str,f)) + "\n")

def cleavesitemotifs(args, cmdline=True):
    
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

def bincounts(args, cmdline=True):    

    for samfile in args.samfiles:
        if args.reference==None:
            parser.error("Reference file is required.")

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
        
        if args.header:
            h=[]
            for ref in bins:
                h+=[ref+'_counts_'+str(x) for x in range(len(bins[ref]))]
        
        v=[]
        for ref in bins:
            v+=list(bins[ref])
        
        if not cmdline:
            return v

        if args.header and samfile==args.samfiles[0]:
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
            
            if read.mapping_quality>=args.mapqual and not read.is_duplicate:
                if args.insertissize:
                    if read.template_length!=None and read.is_read2:
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
        
        if args.norm=='freq':    
            sys.stdout.write("\t".join(map(str,v/v.sum()))+"\n")
        elif args.norm=='rpx':
            sys.stdout.write("\t".join(map(str,(v/(v.sum()/args.x)).astype(int)))+"\n")
        else:
            sys.stdout.write("\t".join(map(str,v))+"\n")

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

import sklearn
import pickle
import base64
from multiprocessing import Pool

def R206C(args):
    (pca,clfb,clfgt,reg)=pickle.load(open(args.clf, 'rb'))

    args.k=8
    args.norm='rpx'
    args.x=1000000
    args.purpyr=False
    
    f=np.array(list(cleavesitemotifs(args, cmdline=False, ))).reshape(1,-1)

    fp=pca.transform(f)

    b=clfb.predict(fp)[0]
    gt=clfgt.predict(fp)[0]
    classp=clfgt.predict_proba(fp)[0]

    actreg=reg.predict(fp)[0]

    sys.stdout.write(f"R206C genotype prediction: {gt} (0={classp[0]:.2f},1={classp[1]:.2f},2={classp[2]:.2f})\tR206C homozygous vs WT: {b}\tDNASE1L3 plasma activity regression: {actreg:.3f}\n")

def plot_fragmentome(args):
    
    from matplotlib import pyplot as plt

    (reducer,embedding,k)=pickle.load(open(args.mapping, 'rb'))

    args.k=4
    args.norm='rpx'
    args.x=1000000
    args.purpyr=False
    args.uselexsmallest=False
    args.useref=False
    args.insertissize=True
    args.lower=60
    args.upper=600

    Xfszd=np.array(fszd(args, cmdline=False, )).reshape(1,-1)
    # print(Xfszd)
    Xcsm=np.array(cleavesitemotifs(args, cmdline=False, )).reshape(1,-1)
    # print(Xcsm)
    Xsem=np.array(_5pends(args, cmdline=False, )).reshape(1,-1)
    # print(Xsem)

    f=np.concatenate((Xfszd,Xcsm,Xsem),axis=1)

    print(f.shape)
    # print(f)

    fp=reducer.transform(f)

    plt.scatter(embedding[:,0],embedding[:,1],c='blue',s=5,alpha=0.5)
    plt.scatter(fp[:,0],fp[:,1],c='red',s=5,alpha=1)
    plt.show()
    if args.outfile is None:
        args.outfile="fragmentome.png"
    plt.savefig(args.outfile)
    plt.close()

parser = argparse.ArgumentParser(prog="cfstats", usage="cfstats -h", description="Gather cfDNA statistics", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def main():
    
    global_parser = argparse.ArgumentParser(add_help=False) #parser for arguments that apply to all subcommands

    global_parser.add_argument("-f", dest="reqflag", default=None, type=int, help="Sam file filter flag: have all of the FLAGs present (like samtools -f option)")
    global_parser.add_argument("-F", dest="exclflag", default=3852, type=int, help="Sam file filter flag: have none of the FLAGs present (like samtools -F option, but exclude duplicates and unmapped read by default)")
    global_parser.add_argument("-q", dest="mapqual", default=60, type=int, help="Minimal mapping quality of reads to be considered (like samtools -q option)")
    global_parser.add_argument("-x", dest="x", default=1000000, type=int, help="Normalisation unit, see norm")
    global_parser.add_argument("--nproc", dest="nproc", default=1, type=int, help="Number of parallel processes to use.")
    global_parser.add_argument("--norm", dest="norm", choices=['counts','freq','rpx'], default='counts', help="Normalize: report counts, frequencies or reads per X reads (default x=1000000, set X with -x option).")

    global_parser.add_argument("-o", dest="maxo", default=None, type=int, help="Limit stats to maxo observations.")
    global_parser.add_argument("--header", dest="header", action="store_true", default=False, help="Write header for names of features")
    global_parser.add_argument("--name", dest="name", action="store_true", default=False, help="Prefix tab-separated values with the name of the file")
    global_parser.add_argument("-r", dest="reference", default=None, type=str, help="Reference file for: reference depended features cleave-site motifs/binned counts/cram decoding.")
    global_parser.add_argument("--seed", dest="seed", default=42, type=int, help="Seed for random number generator.")

    subparsers = parser.add_subparsers()
    
    parser_csmbsz = subparsers.add_parser('csmbsz',prog="cfstats csmbsz", description="Extract k-length cleave-site motifs using the reference sequence at the 5' start/end of cfDNA fragments and stratify by size of the cfDNA fragment.", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_csmbsz.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_csmbsz.add_argument("-k", dest="k", default=4, type=int, help="Length of the cleave-site motifs.")
    parser_csmbsz.add_argument("--pp", dest="purpyr", action="store_true", default=False, help="Collapse nucleotide sequence to Purine/Pyrimidine sequence.")
    parser_csmbsz.add_argument('-l','--lower', default=60, type=int, help='Lower limit for fragments to report')
    parser_csmbsz.add_argument('-u','--upper', default=600, type=int, help='Upper limit for fragments to report')
    parser_csmbsz.add_argument("--noinsert", dest="insertissize", action="store_false", default=True, help="In case of long-read/unpaired sequencing infer fragmentsize from sequence instead of insert.")
    parser_csmbsz.set_defaults(func=cleavesitemotifsbysize)

    parser_csm = subparsers.add_parser('csm',prog="cfstats csm", description="Extract k-length cleave-site motifs using the reference sequence at the 5' start/end of cfDNA fragments.", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_csm.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_csm.add_argument("-k", dest="k", default=4, type=int, help="Length of the cleave-site motifs.")
    parser_csm.add_argument("--pp", dest="purpyr", action="store_true", default=False, help="Collapse nucleotide sequence to Purine/Pyrimidine sequence.")
    parser_csm.set_defaults(func=cleavesitemotifs)
    
    parser_5pends = subparsers.add_parser('5pends',prog="cfstats 5pends", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_5pends.add_argument('samfiles', nargs='+', help='sam/bam/cram file(s)')
    parser_5pends.add_argument("-k", dest="k", default=4, type=int, help="Length of the 5' ends patterns.")
    parser_5pends.add_argument("--useref", action="store_true", dest="useref", default=False, help="Use reference sequence instead of read sequence.")
    parser_5pends.add_argument("--uselexsmallest", action="store_true", dest="uselexsmallest", default=False, help="Only count lexigraphically smallest kmer.")
    parser_5pends.set_defaults(func=_5pends)

    parser_5pendsbsz = subparsers.add_parser('5pendsbsz',prog="cfstats 5pendsbsz", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_5pendsbsz.add_argument('samfiles', nargs='+', help='sam/bam/cram file(s)')
    parser_5pendsbsz.add_argument("-k", dest="k", default=4, type=int, help="Length of the 5' ends patterns.")
    parser_5pendsbsz.add_argument("--useref", action="store_true", dest="useref", default=False, help="Use reference sequence instead of read sequence.")
    parser_5pendsbsz.add_argument("--uselexsmallest", action="store_true", dest="uselexsmallest", default=False, help="Only count lexigraphically smallest kmer.")
    parser_5pendsbsz.add_argument("--pp", dest="purpyr", action="store_true", default=False, help="Collapse nucleotide sequence to Purine/Pyrimidine sequence.")
    parser_5pendsbsz.add_argument('-l','--lower', default=60, type=int, help='Lower limit for fragments to report')
    parser_5pendsbsz.add_argument('-u','--upper', default=600, type=int, help='Upper limit for fragments to report')
    parser_5pendsbsz.add_argument("--noinsert", dest="insertissize", action="store_false", default=True, help="In case of long-read/unpaired sequencing infer fragmentsize from sequence instead of insert.")
    parser_5pendsbsz.set_defaults(func=_5pendsbysize)

    parser_bincounts = subparsers.add_parser('bincounts',prog="cfstats bincounts", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_bincounts.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_bincounts.add_argument("-b", "--binsize", dest="binsize", type=int, default=1000000, help="Size of the bins.")
    parser_bincounts.set_defaults(func=bincounts)

    parser_fszd = subparsers.add_parser('fszd',prog="cfstats fszd", description="Extract fragment size distribution (only for paired-end data)", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_fszd.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_fszd.add_argument('-l','--lower', default=60, type=int, help='Lower limit for fragments to report')
    parser_fszd.add_argument('-u','--upper', default=600, type=int, help='Upper limit for fragments to report')
    parser_fszd.add_argument("--noinsert", dest="insertissize", action="store_false", default=True, help="In case of long-read/unpaired sequencing infer fragmentsize from sequence instead of insert.")
    parser_fszd.set_defaults(func=fszd)
        
    parser_delfi = subparsers.add_parser('delfi',prog="cfstats delfi", description="Extract DELFI-like measure for bins of a predefined size (only for paired-end data)", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_delfi.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_delfi.add_argument("-b", "--binsize", dest="binsize", type=int, default=1000000, help="Size of the bins.")
    parser_delfi.add_argument('--short-lower', dest='shortlow', default=100, help='Definition of short fragments')
    parser_delfi.add_argument('--short-upper', dest='shortup', default=150, help='Definition of short fragments')
    parser_delfi.add_argument('--long-lower', dest='longlow', default=150, help='Definition of long fragments')
    parser_delfi.add_argument('--long-upper', dest='longup', default=200, help='Definition of short fragments')
    parser_delfi.add_argument("--noinsert", dest="insertissize", action="store_false", default=True, help="In case of long-read/unpaired sequencing infer fragmentsize from sequence instead of insert.")
    parser_delfi.set_defaults(func=delfi)

    parser_R206C = subparsers.add_parser('R206C',prog="cfstats R206C", description="Predict R206C genotype using fragmentomics", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_R206C.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_R206C.add_argument('clf', help='Pickled pca/classifier/regressor model')
    parser_R206C.set_defaults(func=R206C)

    parser_plot = subparsers.add_parser('plot',prog="cfstats R206C", description="Plot points in fragmentome embedding", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_plot.add_argument("--outfile", dest="outfile", default=None, help="Name of the file to store the plot.")
    parser_plot.add_argument('mapping', help='Pickled embedding')
    parser_plot.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_plot.set_defaults(func=plot_fragmentome)

    parser_fourier = subparsers.add_parser('fourier', prog="cfstats fourier", description="Extract Fourier transformed coverage profile for each gene", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_fourier.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_fourier.add_argument('gfffile', help='GFF file with gene annotations')
    parser_fourier.set_defaults(func=fourier_transform_coverage)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        random.seed(args.seed)
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
    
