
import pandas as pd
import sklearn
import pickle
import numpy as np
import sys
import logging

from cfstats import csm, fszd, fpends
from cfstats.utils import collect_bam_files
import os

import joblib

log = logging.getLogger(__name__)

def dnase1l3(args, cmdline=True):

    hf_token = getattr(args, 'hf_token', None)

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
    args.samfiles = collect_bam_files(args.samfiles, getattr(args, 'bamlist', None))
    args.bamlist=None

    Xfszd=np.array(fszd.fszd(args, cmdline=False, ))#.reshape(1,-1)
    args.mapqual=60
    args.exclflag=3852

    Xcsm=np.array(csm.cleavesitemotifs(args, cmdline=False, ))#.reshape(1,-1)
    #print("csm",Xcsm,Xcsm.sum())

    Xsem=np.array(fpends._5pends(args, cmdline=False, ))#.reshape(1,-1)
    #print("sem",Xsem,Xsem.sum())

    f=np.concatenate((Xfszd,Xcsm,Xsem),axis=1)

    if hf_token:
        from cfstats.models import remote_dnase1l3_predict
        log.info('Using remote DNASE1L3 API')
        preds, probs = remote_dnase1l3_predict(f, hf_token)
    else:
        clf_svc=joblib.load(args.model)
        if hasattr(clf_svc, 'feature_names_in_'):
            f = pd.DataFrame(f, columns=clf_svc.feature_names_in_)
        preds = clf_svc.predict(f)
        probs = clf_svc.predict_proba(f)

    if cmdline:
        for smp, pred, prob in zip(args.samfiles, preds, probs):
            sys.stdout.write(f"{smp}\t{pred}\t{prob[1]:.4f}\n")
    else:
        return preds, probs

def plot_fragmentome(args):

    from matplotlib import pyplot as plt

    if args.mapping is None:
        sys.stderr.write("Error: --mapping is required for plot_fragmentome.\n")
        sys.exit(1)

    mapping = joblib.load(args.mapping)
    if isinstance(mapping, (tuple, list)):
        reducer = mapping[0]
        xlim = mapping[1] if len(mapping) > 1 else None
        ylim = mapping[2] if len(mapping) > 2 else None
    else:
        reducer, xlim, ylim = mapping, None, None
    embedding = reducer.embedding_

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
    args.samfiles = collect_bam_files(args.samfiles, getattr(args, 'bamlist', None))
    args.bamlist=None

    Xfszd=np.array(fszd.fszd(args, cmdline=False, ))#.reshape(1,-1)

    args.mapqual=60
    args.exclflag=3852
    Xcsm=np.array(csm.cleavesitemotifs(args, cmdline=False, ))#.reshape(1,-1)

    Xsem=np.array(fpends._5pends(args, cmdline=False, ))#.reshape(1,-1)

    f=np.concatenate((Xfszd,Xcsm,Xsem),axis=1)

    log.info("Calculated feature set for %d sample(s): %s", f.shape[0], f.shape)

    fp=reducer.transform(f)

    # Write per-sample UMAP coordinates to TSV in input order, traceable by filename.
    coords_path = getattr(args, 'coords', '-')
    columns = ['x', 'y']
    if getattr(args, 'name', True):
        columns = ['filename'] + columns
    df = pd.DataFrame(fp, columns=['x', 'y'])
    df['filename'] = args.samfiles
    df = df[columns]

    if coords_path is None or coords_path == '-':
        df.to_csv(sys.stdout, sep='\t', index=False, header=getattr(args, 'header', False))
    else:
        df.to_csv(coords_path, sep='\t', index=False, header=getattr(args, 'header', False))
        log.info("Wrote UMAP coordinates to %s", coords_path)

    #import matplotlib.image as mpimg
    #img=mpimg.imread("UMAP_background_scatter.png")
    #(reducer,xlim,ylim)=pickle.load(open("UMAP.pll", 'rb'))
    #plt.imshow(img,extent=[xlim[0],xlim[1],ylim[0],ylim[1]],aspect='auto',origin='upper')

    plt.scatter(embedding[:,0],embedding[:,1],c='blue',s=5,alpha=0.5)
    plt.scatter(fp[:,0],fp[:,1],c='red',s=10,alpha=1)
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    # plt.show()
    if args.outfile is None:
        if len(args.samfiles)==1:
            args.outfile=os.path.basename(args.samfiles[0]).rstrip('cram').rstrip('bam').rstrip('sam')+"fragmentome.png"
        else:
            import uuid
            args.outfile=uuid.uuid4().hex[:8]+".fragmentome.png"

    sys.stderr.write(f"Writing fragmentome plot to: {args.outfile}\n")
    plt.savefig(args.outfile)
    plt.close()
