import pandas as pd
import sklearn
import pickle
import numpy as np
from cfstats import bincounts
from cfstats.utils import collect_bam_files
import sys
import logging

import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)
from glmnet import ElasticNet as glmElasticNet

log = logging.getLogger(__name__)

#cfstats ff /net/beegfs/users/P051809/notebooks/notebooks/ffpredictor_ridge_50kautosomalbins.pickle /net/beegfs/hgn/niptres/allnipt/crams/2019/4/N190307837/N190307837.cram /net/beegfs/hgn/niptres/allnipt/crams/2017/4/N170331413/N170331413.cram /net/beegfs/hgn/niptres/allnipt/crams/2020/1/N200100049/N200100049.cram -r /net/beegfs/hgn/niptres/allnipt/lib/hg38flat.fa --nproc 5

def _maybe_prefix_chr(columns, feats=None, log=None):
    """Opportunistically prefix 'chr' to bin labels when the reference uses
    non-prefixed contig names but the model expects 'chr' prefixes.

    If *feats* is provided, the decision is based on the presence/absence of
    the 'chr' prefix in the model feature names.  Otherwise we fall back to a
    simple heuristic on the bincount column names.
    """
    if columns is None or len(columns) == 0:
        return list(columns) if columns is not None else []

    has_chr_col = any(str(c).startswith('chr') for c in columns)
    if has_chr_col:
        return list(columns)

    if feats is not None and len(feats) > 0:
        has_chr_feat = all(str(f).startswith('chr') for f in feats)
    else:
        # Heuristic: bin labels look like 1_0_50000 / X_0_50000 / MT_0_50000
        has_chr_feat = all(
            str(c).split('_')[0].isdigit() or str(c).split('_')[0] in {'X', 'Y', 'MT'}
            for c in columns if '_' in str(c)
        )

    if has_chr_feat:
        if log is not None:
            log.warning(
                "Reference contigs are not 'chr'-prefixed, but the model expects "
                "'chr' prefixes. Prefixing 'chr' to bin labels opportunistically."
            )
        return ['chr' + str(c) for c in columns]

    return list(columns)


def ff(args, cmdline=True):

    hf_token = getattr(args, 'hf_token', None)

    args.samfiles = collect_bam_files(args.samfiles, getattr(args, 'bamlist', None))
    args.bamlist = None

    #for now use hardcoded match with how our model was trained
    args.binsize=50000
    args.header=True #we need to header to select the right features
    args.exclflag=1024
    args.mapqual=1
    args.gccorrect=False

    log.info("Binning read counts...")
    columns, counts = bincounts.bincounts(args,cmdline=False)
    log.info("Binning done.")

    if hf_token:
        from cfstats.models import remote_ff_predict
        log.info('Using remote FF API')
        columns = _maybe_prefix_chr(columns, log=log)
        ffs = remote_ff_predict(columns, np.array(counts, dtype=np.float64), hf_token)
    else:
        tup=pickle.load(open(args.model, 'rb'))
        #TODO: determine number and type of features based on model
        clf=tup[0]
        feats=tup[1]

        columns = _maybe_prefix_chr(columns, feats=feats, log=log)

        X=pd.DataFrame(counts,columns=columns)
        
        #norm and select bins
        X=X.div(X.sum(axis=1),axis=0).loc[:,feats]

        ffs=clf.predict(X)

    if cmdline:
        for smp,ffval in zip(args.samfiles, ffs):
            sys.stdout.write("%s\t%s\n"%(smp,ffval))
    else:
        return ffs