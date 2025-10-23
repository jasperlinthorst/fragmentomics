from logging import log
import numpy as np
import pandas as pd
import argparse
import sys
import os 
import random
import sklearn
import pickle
import base64
from multiprocessing import Pool

from cfstats import utils
from cfstats import nipt, ff, bincounts, fszd, csm, delfi, fpends

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
    global_parser.add_argument("--noname", dest="name", action="store_false", default=True, help="Do not prefix tab-separated values with the name of the file")
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
    parser_csmbsz.set_defaults(func=csm.cleavesitemotifsbysize)

    parser_csm = subparsers.add_parser('csm',prog="cfstats csm", description="Extract k-length cleave-site motifs using the reference sequence at the 5' start/end of cfDNA fragments.", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_csm.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_csm.add_argument("-k", dest="k", default=4, type=int, help="Length of the cleave-site motifs.")
    parser_csm.add_argument("--pp", dest="purpyr", action="store_true", default=False, help="Collapse nucleotide sequence to Purine/Pyrimidine sequence.")
    parser_csm.set_defaults(func=csm.cleavesitemotifs)
    
    parser_5pends = subparsers.add_parser('5pends',prog="cfstats 5pends", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_5pends.add_argument('samfiles', nargs='+', help='sam/bam/cram file(s)')
    parser_5pends.add_argument("-k", dest="k", default=4, type=int, help="Length of the 5' ends patterns.")
    parser_5pends.add_argument("--useref", action="store_true", dest="useref", default=False, help="Use reference sequence instead of read sequence.")
    parser_5pends.add_argument("--uselexsmallest", action="store_true", dest="uselexsmallest", default=False, help="Only count lexigraphically smallest kmer.")
    parser_5pends.set_defaults(func=fpends._5pends)

    parser_5pendsbsz = subparsers.add_parser('5pendsbsz',prog="cfstats 5pendsbsz", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_5pendsbsz.add_argument('samfiles', nargs='+', help='sam/bam/cram file(s)')
    parser_5pendsbsz.add_argument("-k", dest="k", default=4, type=int, help="Length of the 5' ends patterns.")
    parser_5pendsbsz.add_argument("--useref", action="store_true", dest="useref", default=False, help="Use reference sequence instead of read sequence.")
    parser_5pendsbsz.add_argument("--uselexsmallest", action="store_true", dest="uselexsmallest", default=False, help="Only count lexigraphically smallest kmer.")
    parser_5pendsbsz.add_argument("--pp", dest="purpyr", action="store_true", default=False, help="Collapse nucleotide sequence to Purine/Pyrimidine sequence.")
    parser_5pendsbsz.add_argument('-l','--lower', default=60, type=int, help='Lower limit for fragments to report')
    parser_5pendsbsz.add_argument('-u','--upper', default=600, type=int, help='Upper limit for fragments to report')
    parser_5pendsbsz.add_argument("--noinsert", dest="insertissize", action="store_false", default=True, help="In case of long-read/unpaired sequencing infer fragmentsize from sequence instead of insert.")
    parser_5pendsbsz.set_defaults(func=fpends._5pendsbysize)

    parser_bincounts = subparsers.add_parser('bincounts',prog="cfstats bincounts", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_bincounts.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_bincounts.add_argument("-b", "--binsize", dest="binsize", type=int, default=1000000, help="Size of the bins.")
    parser_bincounts.set_defaults(func=bincounts.bincounts)

    parser_fszd = subparsers.add_parser('fszd',prog="cfstats fszd", description="Extract fragment size distribution (only for paired-end data)", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_fszd.add_argument('samfiles', nargs='*', help='sam/bam/cram file')
    parser_fszd.add_argument("--bamlist", dest="bamlist", type=str, default=None, help="File containing a list of sam/bam/cram files (one per line).")
    parser_fszd.add_argument('-l','--lower', default=60, type=int, help='Lower limit for fragments to report')
    parser_fszd.add_argument('-u','--upper', default=600, type=int, help='Upper limit for fragments to report')
    parser_fszd.add_argument("--noinsert", dest="insertissize", action="store_false", default=True, help="In case of long-read/unpaired sequencing infer fragmentsize from sequence instead of insert.")
    parser_fszd.set_defaults(func=fszd.fszd)
        
    parser_delfi = subparsers.add_parser('delfi',prog="cfstats delfi", description="Extract DELFI-like measure for bins of a predefined size (only for paired-end data)", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_delfi.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_delfi.add_argument("-b", "--binsize", dest="binsize", type=int, default=1000000, help="Size of the bins.")
    parser_delfi.add_argument('--short-lower', dest='shortlow', default=100, help='Definition of short fragments')
    parser_delfi.add_argument('--short-upper', dest='shortup', default=150, help='Definition of short fragments')
    parser_delfi.add_argument('--long-lower', dest='longlow', default=150, help='Definition of long fragments')
    parser_delfi.add_argument('--long-upper', dest='longup', default=200, help='Definition of short fragments')
    parser_delfi.add_argument("--noinsert", dest="insertissize", action="store_false", default=True, help="In case of long-read/unpaired sequencing infer fragmentsize from sequence instead of insert.")
    parser_delfi.set_defaults(func=delfi.delfi)

    parser_R206C = subparsers.add_parser('dnase1l3',prog="cfstats dnase1l3", description="Predict dnase1l3 activity using fragmentomics", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_R206C.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_R206C.add_argument('clf', help='Pickled pca/classifier/regressor model')
    parser_R206C.set_defaults(func=dnase1l3.dnase1l3)

    parser_plot = subparsers.add_parser('plot',prog="cfstats R206C", description="Plot points in fragmentome embedding", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_plot.add_argument("--outfile", dest="outfile", default=None, help="Name of the file to store the plot.")
    parser_plot.add_argument('mapping', help='Pickled embedding')
    parser_plot.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_plot.set_defaults(func=dnase1l3.plot_fragmentome)

    parser_fourier = subparsers.add_parser('fourier', prog="cfstats fourier", description="Extract Fourier transformed coverage profile for each gene", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_fourier.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_fourier.add_argument('gfffile', help='GFF file with gene annotations')
    parser_fourier.set_defaults(func=fourier_transform_coverage)

    parser_ff = subparsers.add_parser('ff', prog="cfstats ff", description="Estimate ff", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_ff.add_argument('predictor', nargs='+', help='Regression model that can be used to predict the fetal fraction.')
    parser_ff.add_argument('samfiles', nargs='+', help='sam/bam/cram files for which ff should be predicted')
    parser_ff.add_argument("--ff", dest="ff", type=float, default=0.10, help="Fetal fraction")
    parser_ff.set_defaults(func=ff.ff)

    parser_nipt = subparsers.add_parser('nipt', prog="cfstats nipt", description="Perform typical NIPT analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_nipt.add_argument('reference_folder', nargs='+', help='folder containing reference sam/bam/cram files')
    parser_nipt.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_nipt.add_argument("--ff", dest="ff", type=float, default=0.10, help="Fetal fraction")
    parser_nipt.set_defaults(func=nipt.nipt)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        random.seed(args.seed)
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
    
