import numpy as np
from cfstats import bincounts, ff, utils
from scipy.stats import norm
import pandas as pd

#Use population of non-trisomic samples to derive ff and llr
def calc_llr_ff_t21(tup, ff=0.01): #initialize ff to 1%, then update if positive LLR

    sample, null, std=tup
    
    alt=[]
    for bin,p in null.items():
        if bin.startswith("chr21"): #but increment the chr21 bins with the expected in increase according to a fixed ff
            p += p * (ff/2) # trisomy 21
        alt.append(p)
    alt=np.array(alt)
    alt= alt / np.sum(alt) #make sum to 1

    llnull = np.log(norm.pdf(sample, loc=null, scale=std)).sum()
    llt21 = np.log(norm.pdf(sample, loc=alt, scale=std)).sum()

    llrt21=2 * (llt21 - llnull)

    llrt21_=None
    if llrt21 > 0:
        ff_=(((sample[null.index[null.index.str.startswith('chr21')]] - null[null.index[null.index.str.startswith('chr21')]]).mean() * 2) / null[null.index[null.index.str.startswith('chr21')]]).mean()
    
    #     ff=(((gcbindf.loc[sample,null.index[null.index.str.startswith('chr21')]] - null[null.index[null.index.str.startswith('chr21')]]).mean() * 2) / null[null.index[null.index.str.startswith('chr21')]]).mean()
    #     ff_std=(((gcbindf.loc[sample,null.index[null.index.str.startswith('chr21')]] - null[null.index[null.index.str.startswith('chr21')]]).mean() * 2) / null[null.index[null.index.str.startswith('chr21')]]).std()

    #     # ff=((gcbindf.loc[sample,gcbindf.columns[gcbindf.columns.str.startswith('chr21')]]-gcbindf.loc[:,gcbindf.columns[gcbindf.columns.str.startswith('chr21')]].mean()).mean() * 2) / gcbindf.loc[:,gcbindf.columns[gcbindf.columns.str.startswith('chr21')]].mean().mean()
    
        #update llrt21
        alt=[]
        for bin,p in null.items():
            if bin.startswith("chr21"): #but increment the chr21 bins with the expected in increase according to a fixed ff
                p += p * (ff_/2) # trisomy 21
            alt.append(p)
        alt=np.array(alt)
        alt= alt / np.sum(alt) #make sum to 1

        llnull = np.log(norm.pdf(sample, loc=null, scale=std)).sum()
        llt21 = np.log(norm.pdf(sample, loc=alt, scale=std)).sum()
        llrt21_= 2 * (llt21 - llnull)

        return (llrt21, ff, llrt21_, ff_, ff_/ff) #updated likelihood ratio and inferred ff, and ratio of trisomy inferred vs globally derived ff
    
    return (llrt21, ff)

def nipt(args):

    #load reference dataset

    # refbindf=pd.read_table("/net/beegfs/hgn/niptres/allnipt/analysis/bincounts/reference.w1000000.q1.F1024.tsv", index_col=0) #TODO: use bincounts in 50kb windows?
    refbindf=pd.read_table(args.referencesamples, index_col=0) #TODO: use bincounts in 50kb windows?
    fullcols=refbindf.columns[(refbindf.columns.str.split('_').str[-1].astype(int) % 1000000 == 0) & ~refbindf.columns.str.match(r'chrM') & ~refbindf.columns.str.match(r'chrX') & ~refbindf.columns.str.match(r'chrY')] #only full 1MB bins and veriseq samples
    refbindf=refbindf.loc[:,fullcols]

    noNcols=[col for col in refbindf.columns if utils.get_N_content_from_fasta(args.reference,col.split("_")[0],col.split("_")[1],col.split("_")[2]) < 0.01]
    refbindf=refbindf.loc[:,noNcols] #only cols with less than 1% N content
    dfcnttot=refbindf.sum(axis=1)
    refbindf=refbindf[(dfcnttot>5e6) & (dfcnttot < 50e6)]

    if args.gccorrect:
        gc_content = utils.get_gc_content(refbindf, reference=args.reference)
        refbindf=utils.gc_correct_counts(refbindf,gc_content) #gc correct

    refbindf=refbindf.div(refbindf.sum(axis=1),axis=0) #norm

    columns, counts = bincounts.bincounts(args,cmdline=False)
    
    bindf=pd.DataFrame(counts, columns=columns, index=args.samfiles).astype(float)
    if args.gccorrect:
        bindf=utils.gc_correct_counts(bindf,gc_content) #gc correct

    bindf=bindf.loc[:,refbindf.columns] #subset columns to reference, then norm
    bindf=bindf.div(bindf.sum(axis=1),axis=0) #norm
    
    #TODO: remove xy chromosomes from normalisation

    # print(bindf)
    # print(refbindf.head())

    for samfile in args.samfiles:
        # print(bindf.loc[samfile,:].sum())
        # print(bindf.loc[samfile,bindf.columns[:10]])
        # print(refbindf.iloc[0,:10])
        # print(refbindf.iloc[0,:].sum())
        # print(refbindf.mean()[:10])

        r=calc_llr_ff_t21((bindf.loc[samfile,:],refbindf.mean(),refbindf.std()), ff=args.ff)
        
        if r[0]>0:
            llrt21, ff, llrt21_, ff_, MR = r
            print(samfile,"t21 llr",llrt21,"ff",ff, "updated llr",llrt21_,"updated ff",ff_,"MR",MR)
        else:
            llrt21, ff = r
            print(samfile,"t21 llr",llrt21,"ff",ff)