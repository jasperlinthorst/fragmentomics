import pandas as pd
import sklearn
import pickle
import numpy as np
from cfstats import bincounts
import sys
from logging import log

import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)
from glmnet import ElasticNet as glmElasticNet


#cfstats ff /net/beegfs/users/P051809/notebooks/notebooks/ffpredictor_ridge_50kautosomalbins.pickle /net/beegfs/hgn/niptres/allnipt/crams/2019/4/N190307837/N190307837.cram /net/beegfs/hgn/niptres/allnipt/crams/2017/4/N170331413/N170331413.cram /net/beegfs/hgn/niptres/allnipt/crams/2020/1/N200100049/N200100049.cram -r /net/beegfs/hgn/niptres/allnipt/lib/hg38flat.fa --nproc 5

def ff(args, cmdline=True):
    tup=pickle.load(open(args.predictor, 'rb'))
    #TODO: determine number and type of features based on model
    clf=tup[0]
    feats=tup[1]

    #for now use hardcoded match with how our model was trained
    args.binsize=50000
    args.header=True #we need to header to select the right features
    args.exclflag=1024
    args.mapqual=1
    args.gccorrect=False

    log(0,"Binning read counts...")
    columns, counts = bincounts.bincounts(args,cmdline=False)
    log(0,"Binning done.")
    
    X=pd.DataFrame(counts,columns=columns)
    
    #norm and select bins
    X=X.div(X.sum(axis=1),axis=0).loc[:,feats]

    ffs=clf.predict(X)
    if cmdline:
        for smp,ff in zip(args.samfiles, ffs):
            sys.stdout.write("%s\t%s\n"%(smp,ff))
    else:
        return ffs