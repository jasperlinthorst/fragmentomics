import numpy as np
from cfstats import bincounts, ff, utils

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

    llnull = np.log(norm.pdf(gcbindf.loc[sample, :], loc=null, scale=std)).sum()
    llt21 = np.log(norm.pdf(gcbindf.loc[sample, :], loc=alt, scale=std)).sum()
    llrt21=2 * (llt21 - llnull)
    ff=None

    llrt21_=None
    if llrt21 > 0:
        
        ff=(((gcbindf.loc[sample,null.index[null.index.str.startswith('chr21')]] - null[null.index[null.index.str.startswith('chr21')]]).mean() * 2) / null[null.index[null.index.str.startswith('chr21')]]).mean()
        ff_std=(((gcbindf.loc[sample,null.index[null.index.str.startswith('chr21')]] - null[null.index[null.index.str.startswith('chr21')]]).mean() * 2) / null[null.index[null.index.str.startswith('chr21')]]).std()

        # ff=((gcbindf.loc[sample,gcbindf.columns[gcbindf.columns.str.startswith('chr21')]]-gcbindf.loc[:,gcbindf.columns[gcbindf.columns.str.startswith('chr21')]].mean()).mean() * 2) / gcbindf.loc[:,gcbindf.columns[gcbindf.columns.str.startswith('chr21')]].mean().mean()
    
        #update llrt21
        alt=[]
        for bin,p in null.items():
            if bin.startswith("chr21"): #but increment the chr21 bins with the expected in increase according to a fixed ff
                p += p * (ff/2) # trisomy 21
            alt.append(p)
        alt=np.array(alt)
        alt= alt / np.sum(alt) #make sum to 1

        llnull = np.log(norm.pdf(gcbindf.loc[sample, :], loc=null, scale=std)).sum()
        llt21 = np.log(norm.pdf(gcbindf.loc[sample, :], loc=alt, scale=std)).sum()
        llrt21_=2 * (llt21 - llnull)

    return llrt21, llrt21_, ff, ff_std, ff_std/ff

def nipt(args):

    #load reference dataset

    refbindf=pd.read_table("/net/beegfs/hgn/niptres/allnipt/analysis/bincounts/reference.w1000000.q1.F1024.tsv", index_col=0) #TODO: use bincounts in 50kb windows?
    fullcols=refbindf.columns[(refbindf.columns.str.split('_').str[-1].astype(int) % 1000000 == 0) & ~refbindf.columns.str.match(r'chrM')] #only full 1MB bins and veriseq samples
    refbindf=refbindf.loc[:,fullcols] 
    noNcols=[col for col in refbindf.columns if get_N_content_from_fasta("hg38flat.fa",col.split("_")[0],col.split("_")[1],col.split("_")[2]) < 0.01]
    refbindf=refbindf.loc[:,noNcols] #only cols with less than 1% N content
    dfcnttot=refbindf.sum(axis=1)
    refbindf=refbindf[(dfcnttot>5e6) & (dfcnttot < 50e6)]
    gc_content = get_gc_content(refbindf)

    refbindf=utils.gc_correct_counts(refbindf,gc_content) #gc correct

    columns, counts = bincounts.bincounts(args,cmdline=False)

    countsdf=pd.DataFrame(counts, columns=columns)

    bindf=utils.gc_correct_counts(countsdf,gc_content) #gc correct

    ffs=ff(args,cmdline=False)

    for samfile in args.samfiles:
        #make calls for each chrom
        pass
