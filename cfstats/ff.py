import pandas as pd
import sklearn
import pickle
from glmnet import ElasticNet as glmElasticNet
import numpy as np
from cfstats import bincounts

def ff(args):
    tup=pickle.load(open(args.reg, 'rb'))
    #TODO: determine number and type of features based on model
    clf=tup[0]
    feats=tup[1]

    #for now use hardcoded match with working model
    args.binsize=50000
    args.header=True #we need to header to select the right features

    columns, counts = bincounts.bincounts(args,cmdline=False)
    
    X=pd.DataFrame(counts,columns=columns)
    
    #norm
    X=X.loc[:,feats].div(X.sum(axis=1),axis=0)

    return clf.predict(X)
    # X=bincounts.loc[:,tup[1]].values/bincounts.values.sum()

    # X=bincounts.bincounts(args,cmdline=False)

    # ff=reg.predict(X)

    # for samfile in args.samfiles:
    #     sys.stdout.write("%s\t%.6f"%(samfile,ff))