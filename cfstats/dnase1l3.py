
import pandas as pd
import sklearn
import pickle
import numpy as np
import sys

from cfstats import csm, fszd, fpends
import os

def dnase1l3(args):
    (pca,clfb,clfgt,reg)=pickle.load(open(args.clf, 'rb'))

    args.k=4
    args.norm='rpx'
    args.x=1000000
    args.purpyr=False
    
    f=np.array(list(csm.cleavesitemotifs(args, cmdline=False, ))).reshape(1,-1)

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
    args.norm='freq'
    
    args.exclflag=3852
    args.mapqual=60

    #args.x=1000000
    args.purpyr=False
    args.uselexsmallest=False
    args.useref=False
    args.insertissize=True
    args.lower=0
    args.upper=1000
    args.bamlist=None

    Xfszd=np.array(fszd.fszd(args, cmdline=False, ))#.reshape(1,-1)
    
    args.mapqual=60
    args.exclflag=3852
    Xcsm=np.array(csm.cleavesitemotifs(args, cmdline=False, ))#.reshape(1,-1)
    #print("csm",Xcsm,Xcsm.sum())

    Xsem=np.array(fpends._5pends(args, cmdline=False, ))#.reshape(1,-1)
    #print("sem",Xsem,Xsem.sum())

    f=np.concatenate((Xfszd,Xcsm,Xsem),axis=1)

    print("Calculated feature set for cramfiles: ",f.shape)
    # print(f)

    fp=reducer.transform(f)
    print(fp)

    plt.scatter(embedding[:,0],embedding[:,1],c='blue',s=5,alpha=0.5)
    plt.scatter(fp[:,0],fp[:,1],c='red',s=10,alpha=1)
    # plt.show()
    if args.outfile is None:
        if len(args.samfiles)==1:
            args.outfile=os.path.basename(args.samfiles[0]).rstrip('cram').rstrip('bam').rstrip('sam')+"fragmentome.png"
        else:
            import uuid
            args.outfile=uuid.uuid4().hex[:8]+"fragmentome.png"

    sys.stderr.write(f"Writing fragmentome plot to: {args.outfile}")
    plt.savefig(args.outfile)
    plt.close()
