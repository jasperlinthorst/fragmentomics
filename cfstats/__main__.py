from logging import log
import argparse
import sys
import os 
import random
import logging as log_module

from cfstats.models import get_model_path

parser = argparse.ArgumentParser(prog="cfstats", usage="cfstats -h", description="Gather cfDNA statistics", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

MODEL_LICENCE_NOTICE = """
================================================================================
  MODEL LICENCE NOTICE
================================================================================
  The model(s) used by this command are provided strictly for non-commercial,
  academic, and research purposes only. They may NOT be used for any commercial
  or clinical purpose. See the models/LICENSE file for full terms.

  By proceeding you confirm that you accept these terms.

  Tip: use --confirm-licence to bypass this prompt in automated pipelines.
================================================================================
"""

def confirm_model_licence(args):
    """Check that the user has accepted the model licence."""
    if getattr(args, 'confirm_licence', False):
        return  # already confirmed via CLI flag
    sys.stderr.write(MODEL_LICENCE_NOTICE)
    try:
        answer = input("Do you accept the model licence terms? [yes/no]: ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("yes", "y"):
        sys.stderr.write("Licence not accepted. Exiting.\n")
        sys.exit(1)

class lazy_cmd:
    """Picklable callable that lazily imports cfstats.<module_name> and calls <func_name>."""
    def __init__(self, module_name, func_name):
        self.module_name = module_name
        self.func_name = func_name
    def __call__(self, args):
        mod = __import__(f"cfstats.{self.module_name}", fromlist=[self.func_name])
        return getattr(mod, self.func_name)(args)

def main():
    global_parser = argparse.ArgumentParser(add_help=False) #parser for arguments that apply to all subcommands
    
    log = log_module
    
    global_parser.add_argument("--loglevel", dest="loglevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='WARNING', help="Log level")
        
    args, unknown_args = global_parser.parse_known_args()
    log.basicConfig(level=getattr(log, args.loglevel))

    global_parser.add_argument("-f", dest="reqflag", default=None, type=int, help="Sam file filter flag: have all of the FLAGs present (like samtools -f option)")
    global_parser.add_argument("-F", dest="exclflag", default=3852, type=int, help="Sam file filter flag: have none of the FLAGs present (like samtools -F option, but exclude duplicates and unmapped read by default)")
    global_parser.add_argument("-q", dest="mapqual", default=60, type=int, help="Minimal mapping quality of reads to be considered (like samtools -q option)")
    global_parser.add_argument("--min-base-quality", dest="min_base_quality", default=17, type=int, help="Minimum base quality for a SNP in a read (default 17)")
    global_parser.add_argument("-x", dest="x", default=1000000, type=int, help="Normalisation unit, see norm")
    global_parser.add_argument("--nproc", dest="nproc", default=1, type=int, help="Number of parallel processes to use.")
    global_parser.add_argument("--norm", dest="norm", choices=['counts','freq','rpx'], default='counts', help="Normalize: report counts, frequencies or reads per X reads (default x=1000000, set X with -x option).")
    global_parser.add_argument("-o", dest="maxo", default=None, type=int, help="Limit stats to maxo observations.")
    global_parser.add_argument("--header", dest="header", action="store_true", default=False, help="Write header for names of features")
    global_parser.add_argument("--noname", dest="name", action="store_false", default=True, help="Do not prefix tab-separated values with the name of the file")
    global_parser.add_argument("-r", "--reference", dest="reference", default=None, type=str, help="Reference file for: reference depended features cleave-site motifs/binned counts/cram decoding.")
    global_parser.add_argument("--seed", dest="seed", default=42, type=int, help="Seed for random number generator.")

    parser = argparse.ArgumentParser(prog="cfstats", usage="cfstats -h", description="Gather cfDNA statistics", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()
    
    parser_csmbsz = subparsers.add_parser('csmbsz',prog="cfstats csmbsz", description="Extract k-length cleave-site motifs using the reference sequence at the 5' start/end of cfDNA fragments and stratify by size of the cfDNA fragment.", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_csmbsz.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_csmbsz.add_argument("-k", dest="k", default=4, type=int, help="Length of the cleave-site motifs.")
    parser_csmbsz.add_argument("--pp", dest="purpyr", action="store_true", default=False, help="Collapse nucleotide sequence to Purine/Pyrimidine sequence.")
    parser_csmbsz.add_argument('-l','--lower', default=60, type=int, help='Lower limit for fragments to report')
    parser_csmbsz.add_argument('-u','--upper', default=600, type=int, help='Upper limit for fragments to report')
    parser_csmbsz.add_argument("--noinsert", dest="insertissize", action="store_false", default=True, help="In case of long-read/unpaired sequencing infer fragmentsize from sequence instead of insert.")
    parser_csmbsz.set_defaults(func=lazy_cmd('csm', 'cleavesitemotifsbysize'))

    parser_csm = subparsers.add_parser('csm',prog="cfstats csm", description="Extract k-length cleave-site motifs using the reference sequence at the 5' start/end of cfDNA fragments.", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_csm.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_csm.add_argument("-k", dest="k", default=4, type=int, help="Length of the cleave-site motifs.")
    parser_csm.add_argument("--pp", dest="purpyr", action="store_true", default=False, help="Collapse nucleotide sequence to Purine/Pyrimidine sequence.")
    parser_csm.set_defaults(func=lazy_cmd('csm', 'cleavesitemotifs'))
    
    parser_5pends = subparsers.add_parser('5pends',prog="cfstats 5pends", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_5pends.add_argument('samfiles', nargs='+', help='sam/bam/cram file(s)')
    parser_5pends.add_argument("-k", dest="k", default=4, type=int, help="Length of the 5' ends patterns.")
    parser_5pends.add_argument("--useref", action="store_true", dest="useref", default=False, help="Use reference sequence instead of read sequence.")
    parser_5pends.add_argument("--uselexsmallest", action="store_true", dest="uselexsmallest", default=False, help="Only count lexigraphically smallest kmer.")
    parser_5pends.set_defaults(func=lazy_cmd('fpends', '_5pends'))

    parser_5pendsbsz = subparsers.add_parser('5pendsbsz',prog="cfstats 5pendsbsz", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_5pendsbsz.add_argument('samfiles', nargs='+', help='sam/bam/cram file(s)')
    parser_5pendsbsz.add_argument("-k", dest="k", default=4, type=int, help="Length of the 5' ends patterns.")
    parser_5pendsbsz.add_argument("--useref", action="store_true", dest="useref", default=False, help="Use reference sequence instead of read sequence.")
    parser_5pendsbsz.add_argument("--uselexsmallest", action="store_true", dest="uselexsmallest", default=False, help="Only count lexigraphically smallest kmer.")
    parser_5pendsbsz.add_argument("--pp", dest="purpyr", action="store_true", default=False, help="Collapse nucleotide sequence to Purine/Pyrimidine sequence.")
    parser_5pendsbsz.add_argument('-l','--lower', default=60, type=int, help='Lower limit for fragments to report')
    parser_5pendsbsz.add_argument('-u','--upper', default=600, type=int, help='Upper limit for fragments to report')
    parser_5pendsbsz.add_argument("--noinsert", dest="insertissize", action="store_false", default=True, help="In case of long-read/unpaired sequencing infer fragmentsize from sequence instead of insert.")
    parser_5pendsbsz.set_defaults(func=lazy_cmd('fpends', '_5pendsbysize'))

    parser_bincounts = subparsers.add_parser('bincounts',prog="cfstats bincounts", description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_bincounts.add_argument('samfiles', nargs='*', help='sam/bam/cram file')
    parser_bincounts.add_argument("--bamlist", dest="bamlist", type=str, default=None, help="File containing a list of sam/bam/cram files (one per line).")
    parser_bincounts.add_argument("-b", "--binsize", dest="binsize", type=int, default=1000000, help="Size of the bins.")
    parser_bincounts.add_argument("--gccorrect", dest="gccorrect", action="store_true", default=False, help="Apply GC content correction.")
    parser_bincounts.add_argument("--frac", dest="frac", type=float, default=0.5, help="GC smoothing parameter, fraction of the data used for fitting.")
    parser_bincounts.set_defaults(func=lazy_cmd('bincounts', 'bincounts'))

    parser_fszd = subparsers.add_parser('fszd',prog="cfstats fszd", description="Extract fragment size distribution (only for paired-end data)", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_fszd.add_argument('samfiles', nargs='*', help='sam/bam/cram file')
    parser_fszd.add_argument("--bamlist", dest="bamlist", type=str, default=None, help="File containing a list of sam/bam/cram files (one per line).")
    parser_fszd.add_argument('-l','--lower', default=60, type=int, help='Lower limit for fragments to report')
    parser_fszd.add_argument('-u','--upper', default=1000, type=int, help='Upper limit for fragments to report')
    parser_fszd.add_argument("--noinsert", dest="insertissize", action="store_false", default=True, help="In case of long-read/unpaired sequencing infer fragmentsize from sequence instead of insert.")
    parser_fszd.set_defaults(func=lazy_cmd('fszd', 'fszd'))
        
    parser_delfi = subparsers.add_parser('delfi',prog="cfstats delfi", description="Extract DELFI-like measure for bins of a predefined size (only for paired-end data)", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_delfi.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_delfi.add_argument("-b", "--binsize", dest="binsize", type=int, default=1000000, help="Size of the bins.")
    parser_delfi.add_argument('--short-lower', dest='shortlow', default=100, help='Definition of short fragments')
    parser_delfi.add_argument('--short-upper', dest='shortup', default=150, help='Definition of short fragments')
    parser_delfi.add_argument('--long-lower', dest='longlow', default=150, help='Definition of long fragments')
    parser_delfi.add_argument('--long-upper', dest='longup', default=200, help='Definition of short fragments')
    parser_delfi.add_argument("--noinsert", dest="insertissize", action="store_false", default=True, help="In case of long-read/unpaired sequencing infer fragmentsize from sequence instead of insert.")
    parser_delfi.set_defaults(func=lazy_cmd('delfi', 'delfi'))

    parser_R206C = subparsers.add_parser('dnase1l3',prog="cfstats dnase1l3", description="Predict dnase1l3 activity using fragmentomics", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_R206C.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_R206C.add_argument('--model', dest='model', default=get_model_path('SVC_all_k4.joblib'), help='Pickled pca/classifier/regressor model')
    parser_R206C.add_argument('--confirm-licence', dest='confirm_licence', action='store_true', default=False, help='Confirm acceptance of the model licence (non-commercial, research-only use). Bypasses the interactive licence prompt.')
    parser_R206C.add_argument('--hf-token', dest='hf_token', default=None, help='Hugging Face token for remote DNASE1L3 API. When set, uses the remote cfstats-umap-api Space instead of the local model.')
    parser_R206C.set_defaults(func=lazy_cmd('dnase1l3', 'dnase1l3'), requires_licence=True)

    parser_plot = subparsers.add_parser('plot',prog="cfstats R206C", description="Plot points in fragmentome embedding", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_plot.add_argument("--outfile", dest="outfile", default=None, help="Name of the file to store the plot.")
    parser_plot.add_argument('--mapping', dest='mapping', default=None, help='Pickled embedding')
    parser_plot.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_plot.set_defaults(func=lazy_cmd('dnase1l3', 'plot_fragmentome'))

    parser_fourier = subparsers.add_parser('fourier', prog="cfstats fourier", description="Extract Fourier transformed coverage profile for each gene", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_fourier.add_argument('samfiles', nargs='+', help='sam/bam/cram file')
    parser_fourier.add_argument('gfffile', help='GFF file with gene annotations')
    parser_fourier.add_argument('-w', dest='window', default=10000, help='Size of the gene body which whould be transformed')
    parser_fourier.add_argument('--amplitude-min', dest='ampmin', default=193, help='Amplitude range over which mean is calculated')
    parser_fourier.add_argument('--amplitude-max', dest='ampmax', default=199, help='Amplitude range over which mean is calculated')
    parser_fourier.set_defaults(func=lazy_cmd('ft', 'fourier_transform_coverage'))

    parser_deconv = subparsers.add_parser('deconv', prog="cfstats deconv", description="Deconvolute fractional cell-type contributions of cfDNA from per-gene FFT-WPS profiles using a single-cell transcriptomic reference atlas (downloaded/cached under the hood).", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_deconv.add_argument('gfffile', help='GFF file with gene annotations (gene ids should be ENSG).')
    parser_deconv.add_argument('samfiles', nargs='+', help='sam/bam/cram file(s) to deconvolve.')
    parser_deconv.add_argument('-w', dest='window', default=10000, type=int, help='Size of the gene body window that is Fourier transformed.')
    parser_deconv.add_argument('--amplitude-min', dest='ampmin', default=193, type=float, help='Lower bound (bp) of the nucleosome-spacing period band.')
    parser_deconv.add_argument('--amplitude-max', dest='ampmax', default=199, type=float, help='Upper bound (bp) of the nucleosome-spacing period band.')
    parser_deconv.add_argument('--reference-atlas', dest='reference_atlas', default=None, help='Local reference: an .h5ad single-cell atlas, or a pre-built genes x cell-types matrix (.parquet/.tsv/.csv). If omitted, the atlas is downloaded/cached under the hood.')
    parser_deconv.add_argument('--atlas-url', dest='atlas_url', default=None, help='URL to download the single-cell atlas .h5ad from (default: Tabula Sapiens on cellxgene).')
    parser_deconv.add_argument('--cell-type-col', dest='cell_type_col', default=None, help='Name of the .obs column holding cell-type labels (auto-detected if omitted).')
    parser_deconv.add_argument('--min-cells', dest='min_cells', default=10, type=int, help='Minimum number of cells for a cell type to be included in the reference.')
    parser_deconv.add_argument('--rebuild-reference', dest='rebuild_reference', action='store_true', default=False, help='Force rebuilding the cached pseudobulk reference matrix.')
    parser_deconv.add_argument('--no-standardize', dest='no_standardize', action='store_true', default=False, help='Do not z-score the FFT-WPS signal and reference columns before NNLS.')
    parser_deconv.add_argument('--relationship', dest='relationship', choices=['auto', 'negative', 'positive'], default='auto', help="Orientation of the FFT-WPS vs expression relationship. FFT-WPS intensity typically decreases with expression; 'auto' detects the sign from the data.")
    parser_deconv.add_argument('--bootstrap', dest='n_bootstrap', default=0, type=int, metavar='N', help='Number of gene-level bootstrap iterations for uncertainty estimation. Writes a second <output>.bootstrap_std.tsv with per-cell-type STD. 0 disables (default).')
    parser_deconv.add_argument('--output', '-O', dest='output', default='-', help="Output TSV path (rows=samples, columns=cell types). '-' writes to stdout.")
    parser_deconv.set_defaults(func=lazy_cmd('deconv', 'deconv'))

    parser_nucs = subparsers.add_parser('nucs', prog="cfstats nucs", description="Call nucleosomes from WPS profiles (region or genome-wide)", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_nucs.add_argument('samfiles', nargs='+', help='sam/bam/cram file(s)')
    parser_nucs.add_argument('--chrom', dest='chrom', default=None, help='Chromosome name (e.g. chr1). If omitted, scan all contigs.')
    parser_nucs.add_argument('--start', dest='start', type=int, default=None, help='Start coordinate (0-based, inclusive). If omitted, start at 0 for the chromosome.')
    parser_nucs.add_argument('--end', dest='end', type=int, default=None, help='End coordinate (0-based, exclusive). If omitted, use end of chromosome.')
    parser_nucs.add_argument('-k', dest='k', type=int, default=120, help='WPS window size (bp)')
    parser_nucs.add_argument('--min-len', dest='min_len', type=int, default=120, help='Minimum fragment length to include')
    parser_nucs.add_argument('--max-len', dest='max_len', type=int, default=180, help='Maximum fragment length to include')
    parser_nucs.add_argument('--min-prominence', dest='min_prominence', type=float, default=5.0, help='Minimum WPS peak prominence for nucleosome calling')
    parser_nucs.add_argument('--min-distance', dest='min_distance', type=int, default=147, help='Minimum distance between nucleosome peaks (bp)')
    parser_nucs.set_defaults(func=lazy_cmd('nucs', 'nucs'))

    parser_ff = subparsers.add_parser('ff', prog="cfstats ff", description="Estimate ff", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_ff.add_argument('samfiles', nargs='+', help='sam/bam/cram files for which ff should be predicted')
    parser_ff.add_argument('--model', dest='model', default=get_model_path('ffpredictor_50kautosomalbins.pickle'), help='Regression model that can be used to predict the fetal fraction.')
    parser_ff.add_argument('--confirm-licence', dest='confirm_licence', action='store_true', default=False, help='Confirm acceptance of the model licence (non-commercial, research-only use). Bypasses the interactive licence prompt.')
    parser_ff.add_argument('--hf-token', dest='hf_token', default=None, help='Hugging Face token for remote FF API. When set, uses the remote cfstats-umap-api Space instead of the local model.')
    parser_ff.set_defaults(func=lazy_cmd('ff', 'ff'), requires_licence=True)

    parser_nipt = subparsers.add_parser('nipt', prog="cfstats nipt", description="Perform typical NIPT analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_nipt.add_argument('referencesamples', help='Tab-separated value list in which rows are samples and columns are bincounts (matched with specified bin size)')
    parser_nipt.add_argument('samfiles', nargs='+', help='sam/bam/cram files for which gains or deletion should be called')
    parser_nipt.add_argument("--ff", dest="ff", type=float, default=0.10, help="Global fetal fraction to use")
    parser_nipt.add_argument("-b", "--binsize", dest="binsize", type=int, default=1000000, help="Size of the bins.")
    parser_nipt.add_argument("--gccorrect", dest="gccorrect", action="store_true", default=False, help="Apply GC content correction before normalisation and calling")
    parser_nipt.set_defaults(func=lazy_cmd('nipt', 'nipt'))

    parser_fragmentome = subparsers.add_parser('fragmentome', prog="cfstats fragmentome", description="Start interactive fragmentome explorer web application (data is loaded from ClickHouse)", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_fragmentome.add_argument('--mapping', dest='mapping', default=None, help='Pickled (reducer, embedding, k) mapping as produced/used by cfstats dnase1l3 plot. Required for upload-to-embedding functionality.')
    parser_fragmentome.add_argument('--hf-token', dest='hf_token', default=None, help='Hugging Face token for remote UMAP API. When set, uses the remote cfstats-umap-api Space instead of downloading the 3.5 GB model locally.')
    parser_fragmentome.add_argument('--ch-host', dest='ch_host', default='localhost', help='ClickHouse server host.')
    parser_fragmentome.add_argument('--ch-port', dest='ch_port', default=8123, type=int, help='ClickHouse HTTP port.')
    parser_fragmentome.add_argument('--admin-password', dest='admin_password', default=None, help='Password for the /admin upload page. If omitted, uses FRAGMENTOME_ADMIN_PASSWORD env var.')
    parser_fragmentome.set_defaults(func=lazy_cmd('fragmentome', 'explore'))

    # --- imputeref: build and train reference model in one step ------------
    parser_imputeref = subparsers.add_parser(
        'imputeref', prog="cfstats imputeref",
        description="Build reference panel from BAM/VCF files and train HMM model in one step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_imputeref.add_argument('targetfile', help='Target VCF (sites only) or .pos/.txt with chrom pos ref alt.')
    parser_imputeref.add_argument('ifiles', nargs='+', help='.vcf(.gz)/.bcf/.bam/.sam/.cram or .txt manifests.')
    parser_imputeref.add_argument('-k', dest='k', default=4, type=int, help='Number of HMM states.')
    parser_imputeref.add_argument('--maxiter', dest='maxiter', default=40, type=int, help='EM iterations.')
    parser_imputeref.add_argument('--warm-start', dest='warm_start', default=None, type=str,
                                    help='Warm-start from a previously trained model VCF.')
    parser_imputeref.add_argument('--maxvar', dest='maxvar', default=int(10e6), type=int, help='Max number of variants.')
    parser_imputeref.add_argument('--region', dest='region', default=None, type=str, help='Restrict to region.')
    parser_imputeref.add_argument('--outputprefix', dest='outputprefix', default=None, type=str, help='Output file prefix.')
    parser_imputeref.add_argument('--addchr', dest='addchr', action='store_true', default=False)
    parser_imputeref.add_argument('--rmchr', dest='rmchr', action='store_true', default=False)
    parser_imputeref.add_argument('--filterflag', dest='filterflag', default=3840, type=int, help='Alignments to exclude (samtools -F).')
    parser_imputeref.add_argument("--cram-reference", dest="cramref", default=None, type=str, help="Reference FASTA for CRAM decoding.")
    parser_imputeref.add_argument('--ngen', dest='ngen', type=int, default=100, help='Generations since founding.')
    parser_imputeref.set_defaults(func=lazy_cmd('impute.cli', 'imputeref'))

    # --- impute: genotype imputation (diploid / NIPT triploid) -----------
    # Direct imputation using a reference panel (no subcommand needed)
    parser_impute = subparsers.add_parser(
        'impute', prog="cfstats impute",
        description="Impute genotypes from a BAM/CRAM using a phased population reference panel (trained model VCF, or standard hap/legend).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[global_parser])
    parser_impute.add_argument('reference', help='Phased reference VCF (.vcf.gz), trained model VCF from imputeref, or prefix of hap.gz/legend.gz pair.')
    parser_impute.add_argument('input', help='Input sam/bam/cram with reads.')
    parser_impute.add_argument('chrom', help='Chromosome to impute (as it appears in the reference VCF).')
    parser_impute.add_argument("--start", dest="start", default=None, type=int, help="Start position (VCF reference only).")
    parser_impute.add_argument("--stop", dest="stop", default=None, type=int, help="End position (VCF reference only).")
    parser_impute.add_argument("--impute-output", dest="output", default='-', type=str, help="Output (bgzipped) VCF; '-' writes to stdout.")
    parser_impute.add_argument("--sample", dest="sample", default=None, type=str, help="Sample name in the output VCF (defaults to the input basename).")
    parser_impute.add_argument("--addchr", dest="addchr", action="store_true", default=False, help="Prefix 'chr' to contig names when fetching from the input.")
    parser_impute.add_argument("--rmchr", dest="rmchr", action="store_true", default=False, help="Strip 'chr' prefix from contig names when fetching from the input.")
    parser_impute.add_argument("--cram-reference", dest="cramref", default=None, type=str, help="Reference FASTA for CRAM decoding.")
    parser_impute.add_argument("--ngen", dest="ngen", type=int, default=100, help="Generations since founding the reference population.")
    parser_impute.add_argument("--avgr", dest="avgr", type=int, default=1, help="Average recombination rate in cM/Mb.")
    parser_impute.add_argument("--minp", dest="minp", type=float, default=1e-3, help="Minimum probability (prevents underflow in fwd/bwd).")
    parser_impute.add_argument("--diploid", dest="diploid", action="store_true", default=False, help="DEPRECATED/no-op: diploid mean-field is now the default model. Triploid (NIPT) is selected by supplying --ff (or --read-prior).")
    parser_impute.add_argument("--ff", dest="ff", default=None, type=float, help="Expected fetal fraction. Supplying this enables the mean-field triploid (NIPT) model and seeds the EM that re-estimates the fetal fraction. Omit for the default diploid model.")
    parser_impute.add_argument("--nhap", dest="nhap", default=None, type=int, help="Pre-select the best-matching N haplotypes (None = use all).")
    parser_impute.add_argument("--random-init", dest="random_init", action="store_true", default=False, help="Use random initial haplotype selection with iterative posterior-guided re-selection (no msPBWT).")
    parser_impute.add_argument("--gibbs", dest="gibbs", action="store_true", default=False, help="Diploid only: use Gibbs sampling on read labels instead of the default mean-field conditioning. Reads are randomly assigned to hap1/hap2, separate fwd/bwd passes are run per haplotype, and read labels are hard-resampled from the posterior each iteration.")
    parser_impute.add_argument("--knew", dest="knew", default=None, type=int, help="Number of new haplotypes to add per iteration when using random-init (default: nhap).")
    parser_impute.add_argument("--fulliter", dest="gibbs_fulliter", type=int, default=3, help="Diploid: mean-field iterations. Triploid: full-panel selection passes.")
    parser_impute.add_argument("--partiter", dest="gibbs_partiter", type=int, default=None, help="Max label-reassignment iterations (triploid; None = until convergence).")
    parser_impute.add_argument("--maxnreads", dest="maxnreads", type=int, default=None, help="Limit the total number of reads considered.")
    parser_impute.add_argument("--nthreads", dest="nthreads", type=int, default=None, help="OpenMP threads and per-label fwd/bwd pool size (default: --nproc).")
    parser_impute.add_argument("--genetic-map", dest="genetic_map", type=str, default=None, help="Path to a (gzipped) PLINK-format genetic map file (columns: position COMBINED_rate Genetic_Map.cM). When provided, sigma is computed from interpolated genetic distances instead of a uniform recombination rate.")
    parser_impute.add_argument("--nophase", dest="nophase", action="store_true", default=False, help="Disable the phasing iteration (default: phasing is on). When phasing is on, a final forward-backward pass warm-started from converged read assignments produces phased GT (0|0, 0|1, 1|0, 1|1) with DS/GP recomputed from recast haplotypes.")
    parser_impute.add_argument("--dump", dest="dump", type=str, default=None, help="Dump gamma/emission/sigma/hap-path matrices for each iteration to this path prefix (e.g. 'cfstats_dump'). Files: <prefix>_iter<N>_hap<1|2>.npz")
    parser_impute.add_argument("--read-prior", dest="read_prior", action="store_true", default=False, help="Implies the triploid (NIPT) model. Use a per-read fetal prior decoded from the per-read tag (default XF, written by the read classifier) to modulate the mean-field emission weights and the EM responsibilities, instead of the global --ff prior. w_i = 10**(-XF/10).")
    parser_impute.add_argument("--read-prior-tag", dest="read_prior_tag", type=str, default="XF", help="SAM tag holding the Phred-encoded per-read fetal posterior used by --read-prior.")
    parser_impute.set_defaults(func=lazy_cmd('impute.cli', 'impute_ref'))

    args = parser.parse_args()

    if hasattr(args, 'func'):
        random.seed(args.seed)
        if getattr(args, 'requires_licence', False):
            confirm_model_licence(args)
        #try:
        args.func(args)
        #except Exception as e:
        #    parser.error(str(e))
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
    
