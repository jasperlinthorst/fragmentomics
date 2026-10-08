# cfstats — cell-free DNA fragmentomics toolkit

A command-line toolkit for extracting fragmentomic features from cell-free DNA
sequencing data (BAM/CRAM). It provides subcommands for fragment size
distributions, cleavage-site motifs, 5′-end sequence patterns, genome-wide bin
counts, nucleosome positioning, cell-type deconvolution, genotype imputation,
and more. Pre-trained models for DNASE1L3 activity prediction, fetal-fraction
estimation and UMAP-based fragmentome embedding are available separately (see
[Models](#models) below).

## Installation

```bash
git clone https://github.com/jasperlinthorst/fragmentomics.git
cd fragmentomics
pip install .
```

After installation the `cfstats` command is available on your `PATH`.
A reference FASTA (e.g. `hg38flat.fa`) is required for most subcommands; pass
it with `-r / --reference`.

### Dependencies

Key dependencies (pinned in `setup.py`): numpy, scikit-learn, pandas, pysam,
biopython, scipy, statsmodels, plotly, huggingface_hub.

## Quick start

```bash
# Fragment size distribution (paired-end data)
cfstats fszd -r hg38.fa sample.cram

# Cleavage-site motifs (4-mer, normalised to frequencies)
cfstats csm -r hg38.fa --norm freq sample.cram

# 5′-end k-mer patterns
cfstats 5pends -r hg38.fa sample.cram

# Genome-wide bin counts (1 Mb bins)
cfstats bincounts -r hg38.fa -b 1000000 sample.cram

# Fourier-transformed coverage (Leuven pipeline)
cfstats fourier --leuven -r hg38.fa genes.tsv sample.cram

# Cell-type deconvolution from FFT-WPS profiles
cfstats deconv -r hg38.fa --nproc 4 sample.cram

# DNASE1L3 prediction via remote API
cfstats dnase1l3 --hf-token $HF_TOKEN -r hg38.fa sample.cram

# Fetal fraction estimation via remote API
cfstats ff --hf-token $HF_TOKEN -r hg38.fa sample.cram
```

Prediction models are available through the Hugging Face Hub and can be accessed through an access token (the `--hf-token` flag). Contact j.linthorst@amsterdamumc.nl to get access.
A demo of the models is available at https://huggingface.co/spaces/jasperlinthorst/cfstats-demo.

## Global options

These flags apply to all subcommands:

| Flag | Description | Default |
|------|-------------|---------|
| `-r, --reference` | Reference FASTA (required for CRAM and reference-dependent features) | — |
| `-F` | SAM exclusion flag (like `samtools -F`) | 3852 |
| `-f` | SAM required flag (like `samtools -f`) | — |
| `-q` | Minimum mapping quality | 60 |
| `-o` | Limit to *n* observations (subsampling) | all |
| `--norm` | Normalisation: `counts`, `freq`, or `rpx` | counts |
| `-x` | RPX normalisation unit | 1 000 000 |
| `--nproc` | Parallel processes | 1 |
| `--header` | Print a header line with feature names | off |
| `--noname` | Omit sample name prefix | off |
| `--min-base-quality` | Minimum base quality for SNP-related read filtering | 17 |
| `--bamlist` | File with one BAM/CRAM path per line | — |
| `--seed` | Random seed | 42 |
| `--loglevel` | Logging verbosity | WARNING |

## Subcommands

### `fszd` — Fragment size distribution

Extract the distribution of fragment sizes (paired-end data only).

```bash
cfstats fszd -r hg38.fa sample.cram
cfstats fszd -r hg38.fa --norm freq -l 60 -u 600 sample.cram
cfstats fszd --bamlist samples.txt -r hg38.fa --nproc 4
```

| Flag | Description | Default |
|------|-------------|---------|
| `-l, --lower` | Minimum fragment length | 60 |
| `-u, --upper` | Maximum fragment length | 1000 |
| `--noinsert` | Infer size from sequence (long-read / unpaired) | off |

### `csm` — Cleavage-site motifs

Extract *k*-length motifs at the 5′ cleavage sites using the reference sequence.

```bash
cfstats csm -r hg38.fa sample.cram
cfstats csm -r hg38.fa -k 6 --norm freq sample.cram
cfstats csm -r hg38.fa --pp sample.cram          # purine/pyrimidine collapse
```

| Flag | Description | Default |
|------|-------------|---------|
| `-k` | Motif length | 4 |
| `--pp` | Collapse to purine/pyrimidine | off |

### `csmbsz` — Cleavage-site motifs by fragment size

Same as `csm`, but stratified by fragment size.

```bash
cfstats csmbsz -r hg38.fa -l 60 -u 600 sample.cram
```

Additional flags: `-k`, `--pp`, `-l`, `-u`, `--noinsert` (see `csm` / `fszd`).

### `5pends` — 5′-end sequence patterns

Extract *k*-mer frequencies at the 5′ ends of fragments.

```bash
cfstats 5pends -r hg38.fa sample.cram
cfstats 5pends -r hg38.fa --useref -k 6 sample.cram
```

| Flag | Description | Default |
|------|-------------|---------|
| `-k` | Pattern length | 4 |
| `--useref` | Use reference instead of read sequence | off |
| `--uselexsmallest` | Count only the lexicographically smallest k-mer | off |

### `5pendsbsz` — 5′-end patterns by fragment size

Same as `5pends`, but stratified by fragment size.

```bash
cfstats 5pendsbsz -r hg38.fa -l 60 -u 600 sample.cram
```

Additional flags: `-k`, `--useref`, `--uselexsmallest`, `--pp`, `-l`, `-u`, `--noinsert`.

### `bincounts` — Genome-wide bin counts

Count reads in fixed-size genomic bins.

```bash
cfstats bincounts -r hg38.fa sample.cram
cfstats bincounts -r hg38.fa -b 50000 --gccorrect sample.cram
cfstats bincounts -r hg38.fa --bamlist samples.txt --nproc 8
```

| Flag | Description | Default |
|------|-------------|---------|
| `-b, --binsize` | Bin size (bp) | 1 000 000 |
| `--gccorrect` | Apply GC-content correction | off |
| `--frac` | LOESS smoothing fraction for GC correction | 0.5 |

### `delfi` — DELFI-like fragmentation measure

Compute the ratio of short-to-long fragments per genomic bin (inspired by
[Cristiano *et al.*, Nature 2019](https://doi.org/10.1038/s41586-019-1272-6)).

```bash
cfstats delfi -r hg38.fa -b 1000000 sample.cram
```

| Flag | Description | Default |
|------|-------------|---------|
| `--short-lower / --short-upper` | Short fragment range | 100–150 |
| `--long-lower / --long-upper` | Long fragment range | 150–200 |
| `--noinsert` | Infer size from sequence (long-read / unpaired) | off |

### `dnase1l3` — DNASE1L3 activity prediction

Predict DNASE1L3 nuclease activity from fragmentomic features (SVC classifier).

```bash
# Using the remote API (recommended — no local model needed):
cfstats dnase1l3 --hf-token $HF_TOKEN -r hg38.fa sample.cram

# Using a local model file:
cfstats dnase1l3 --model SVC_all_k4.joblib --confirm-licence -r hg38.fa sample.cram
```

| Flag | Description | Default |
|------|-------------|---------|
| `--hf-token` | HF API token → use remote API instead of local model | — |
| `--model` | Path to local model file | bundled |
| `--confirm-licence` | Accept research-only licence non-interactively | off |

### `ff` — Fetal fraction estimation

Estimate the fetal fraction from bin-count profiles.

```bash
# Using the remote API:
cfstats ff --hf-token $HF_TOKEN -r hg38.fa sample.cram

# Using a local model file:
cfstats ff --model ffpredictor_50kautosomalbins.pickle --confirm-licence -r hg38.fa sample.cram
```

| Flag | Description | Default |
|------|-------------|---------|
| `--hf-token` | HF API token → use remote API | — |
| `--model` | Path to local model file | bundled |
| `--confirm-licence` | Accept research-only licence non-interactively | off |

### `nipt` — NIPT-style aneuploidy screening

Call chromosomal gains/deletions relative to a reference cohort.

```bash
cfstats nipt reference_bincounts.tsv -r hg38.fa sample.cram --ff 0.10
```

| Flag | Description | Default |
|------|-------------|---------|
| `--ff` | Assumed fetal fraction | 0.10 |
| `-b` | Bin size | 1 000 000 |
| `--gccorrect` | GC correction | off |

### `fourier` — Fourier-transformed coverage profiles

Extract Fourier-transformed coverage across gene bodies
([Snyder *et al.*, Cell 2016](https://doi.org/10.1016/j.cell.2015.11.050)).

```bash
# Default mode (wide output: samples as rows, genes as columns)
cfstats fourier -r hg38.fa --genemodel genes.gff sample1.cram sample2.cram

# Leuven-compatible output (long: genes as rows, discrete period amplitudes as columns; single sample)
cfstats fourier --leuven -r hg38.fa --genemodel Ensemble_canonical_GRCh38.body.tsv sample.cram

# Leuven processing in wide format, allowing multiple samples / --bamlist
cfstats fourier --leuven --wide --nproc 4 --bamlist bamlist.txt \
    -r hg38.fa --genemodel Ensemble_canonical_GRCh38.body.tsv
```

| Flag | Description | Default |
|------|-------------|---------|
| `--genemodel` | Gene annotation (GFF/GTF or five-column gene body TSV) | bundled |
| `-w` | Gene body window size to transform (bp) | 10000 |
| `--amplitude-min` | Lower bound of period band (bp) | 193 |
| `--amplitude-max` | Upper bound of period band (bp) | 199 |
| `--amplitude-step` | Step between discrete period amplitudes in `--leuven` mode (bp) | 3 |
| `--leuven` | Use Leuven read selection, WPS scoring and spectral transform | off |
| `--long` | Long output: gene rows, period columns; requires one sample | `--leuven` default |
| `--wide` | Wide output: sample rows, gene columns; allows multiple samples | default otherwise |

### `deconv` — Cell-type deconvolution

Deconvolute fractional cell-type contributions from per-gene FFT-WPS profiles
using a single-cell transcriptomic reference atlas (downloaded/cached under the
hood, or supplied locally).

```bash
# Use the default Tabula Sapiens atlas (downloaded on first run)
cfstats deconv -r hg38.fa sample.cram

# Local pre-built reference matrix (genes × cell types)
cfstats deconv -r hg38.fa --reference-atlas ref.parquet sample.cram

# Bootstrap uncertainties (writes sample.bootstrap_std.tsv)
cfstats deconv -r hg38.fa --bootstrap 100 --output sample_deconv.tsv sample.cram
```

| Flag | Description | Default |
|------|-------------|---------|
| `--genemodel` | Gene annotation (GFF/GTF or five-column gene body TSV; gene ids should be ENSG) | bundled |
| `-w` | Gene body window for FFT-WPS (bp) | 10000 |
| `--amplitude-min` | Lower bound of nucleosome-spacing period band (bp) | 193 |
| `--amplitude-max` | Upper bound of nucleosome-spacing period band (bp) | 199 |
| `--reference-atlas` | Local `.h5ad` atlas or genes × cell-types matrix (.parquet/.tsv/.csv) | — |
| `--atlas-url` | URL to download a single-cell atlas `.h5ad` | Tabula Sapiens |
| `--cell-type-col` | `.obs` column holding cell-type labels | auto-detected |
| `--min-cells` | Minimum cells for a cell type to be included | 10 |
| `--chunk-size` | Cells per chunk when streaming a large `.h5ad` | 20000 |
| `--rebuild-reference` | Force rebuilding the cached pseudobulk reference | off |
| `--no-standardize` | Do not z-score signal/reference columns before NNLS | off |
| `--relationship` | FFT-WPS vs expression relationship: `auto`, `negative`, `positive` | auto |
| `--bootstrap` | Gene-level bootstrap iterations for uncertainty | 0 |
| `--amplitude-step` | Step between discrete period amplitudes (bp) | 1 (3 with `--leuven`/`--rankcorr`) |
| `--leuven` | Leuven read selection, WPS scoring, strand orientation and spectral transform | off |
| `--rankcorr` | Ranked-correlation output (Leuven/Kate method) instead of NNLS fractions; implies `--leuven` | off |
| `--output`, `-O` | Output TSV path (`-` for stdout) | `-` |

With `--rankcorr`, each reference cell type is Pearson-correlated against the
per-gene FFT-WPS signal (genes observed in both, i.e. R's
`pairwise.complete.obs`) and cell types are ranked in ascending order of
correlation: rank 1 is the most negatively correlated (strongest contributor,
since FFT-WPS intensity decreases with expression). The output TSV holds ranks;
raw coefficients are written to `<output>.correlations.tsv`. This replicates
the `cellforigin_correlations.R` step of the Leuven pipeline exactly when the
same genes × cell-types matrix is passed via `--reference-atlas`.

### `nucs` — Nucleosome calling from WPS

Call nucleosome positions from Windowed Protection Scores.

```bash
cfstats nucs -r hg38.fa sample.cram
cfstats nucs -r hg38.fa --chrom chr22 --start 0 --end 50000000 sample.cram
```

| Flag | Description | Default |
|------|-------------|---------|
| `--chrom` | Restrict to chromosome | all |
| `--start / --end` | Genomic region | full chrom |
| `-k` | WPS window size | 120 |
| `--min-len / --max-len` | Fragment length filter | 120–180 |
| `--min-prominence` | Peak prominence threshold | 5.0 |
| `--min-distance` | Minimum inter-nucleosome distance | 147 |

### `gcbias` — Per-sample GC-bias correction table

Estimate a Griffin-style GC-bias correction table (`length × num_GC`) for use by
`siteprofile`. Run once per sample.

```bash
cfstats gcbias -r hg38.fa sample.cram
cfstats gcbias -r hg38.fa --size-range 100 200 --out-dir gc_tables/ *.bam
```

| Flag | Description | Default |
|------|-------------|---------|
| `--size-range` | Fragment length range to model (two integers) | 100 200 |
| `--gc-samples` | Random genomic windows used to estimate expected GC frequency | 200000 |
| `--chroms` | Chromosomes sampled for expected GC frequency | chr1–chr22 |
| `--out-dir` | Output directory for per-sample `<sample>.GC_bias.txt` files | `.` |

### `siteprofile` — Composite nucleosome coverage around sites

Compute Griffin-style composite GC-corrected nucleosome coverage profiles and
features (`mean_coverage`, `central_coverage`, `amplitude`) around a list of
sites. Sites can be provided as a Griffin-style YAML or a BED file.

```bash
# Uncorrected profile
cfstats siteprofile -r hg38.fa sites.bed sample.cram

# GC-corrected profile (run cfstats gcbias first)
cfstats siteprofile -r hg38.fa --gc-bias sample.GC_bias.txt sites.yaml sample.cram

# Save full binned profiles to a directory
cfstats siteprofile -r hg38.fa --gc-bias sample.GC_bias.txt \
    --save-window -1000 1000 --save-profile profiles/ sites.bed sample.cram
```

| Flag | Description | Default |
|------|-------------|---------|
| `--gc-bias` | GC-bias table from `cfstats gcbias` | — |
| `--size-range` | Fragment length range to include | 100 200 |
| `--norm-window` | Window around each site used for profile normalisation | -5000 5000 |
| `--save-window` | Window used for `mean_coverage` / `amplitude` features | -1000 1000 |
| `--center-window` | Window used for the `central_coverage` feature | -30 30 |
| `--step` | Bin size (bp) for the composite profile | 15 |
| `--fft-index` | FFT component index used for the `amplitude` feature | 10 |
| `--smoothing-length` | Savitzky–Golay smoothing window (bp) | 165 |
| `--no-smoothing` | Disable Savitzky–Golay smoothing | off |
| `--chroms` | Restrict sites to these chromosomes | all |
| `--chrom-column` | Chromosome column name (YAML/tsv site lists) | Chrom |
| `--position-column` | Position column name (YAML/tsv site lists) | position |
| `--strand-column` | Strand column name (YAML/tsv site lists) | Strand |
| `--save-profile` | Directory to write full binned profiles | — |

### `imputeref` — Build reference panel and train HMM

Build a reference panel from BAM/VCF files and train an HMM imputation model in
one step. The output is a trained model VCF that can be passed to `cfstats impute`.

```bash
cfstats imputeref targets.vcf.gz -r hg38.fa cohort1.bam cohort2.bam -k 4 --outputprefix myref
```

| Flag | Description | Default |
|------|-------------|---------|
| `-k` | Number of HMM states | 4 |
| `--maxiter` | EM iterations | 40 |
| `--warm-start` | Warm-start from a previously trained model VCF | — |
| `--maxvar` | Maximum number of variants to consider | 10 000 000 |
| `--region` | Restrict to region | — |
| `--outputprefix` | Output file prefix | auto |
| `--addchr` | Prefix `chr` to contig names | off |
| `--rmchr` | Strip `chr` prefix from contig names | off |
| `--filterflag` | Alignments to exclude (samtools `-F`) | 3840 |
| `--cram-reference` | Reference FASTA for CRAM decoding | — |
| `--ngen` | Generations since founding | 100 |

### `impute` — Genotype imputation from BAM/CRAM

Impute genotypes from an alignment file using a phased population reference
panel (standard hap/legend files, a VCF/BCF, or a trained model VCF from
`cfstats imputeref`). Supports diploid samples and triploid (NIPT/maternal
plasma) samples via `--ff` or `--read-prior`.

```bash
# Diploid imputation
cfstats impute refpanel.vcf.gz sample.cram chr20 --impute-output sample.vcf.gz

# NIPT/triploid with global fetal fraction prior
cfstats impute model.vcf.gz maternal.cram chr13 --ff 0.10 --impute-output nipt.vcf.gz

# NIPT with per-read fetal posteriors from a SAM tag (e.g. XF)
cfstats impute model.vcf.gz maternal.cram chr21 --read-prior --impute-output nipt.vcf.gz
```

| Flag | Description | Default |
|------|-------------|---------|
| `--start / --stop` | Start/end positions (VCF reference only) | — |
| `--impute-output` | Output (bgzipped) VCF (`-` for stdout) | `-` |
| `--sample` | Sample name in output VCF | input basename |
| `--addchr` | Prefix `chr` to input contig names | off |
| `--rmchr` | Strip `chr` prefix from input contig names | off |
| `--cram-reference` | Reference FASTA for CRAM decoding | — |
| `--ngen` | Generations since founding the reference population | 100 |
| `--avgr` | Average recombination rate (cM/Mb) | 1 |
| `--minp` | Minimum probability (prevents underflow) | 1e-3 |
| `--ff` | Expected fetal fraction; enables triploid (NIPT) model | — |
| `--read-prior` | Use per-read fetal priors from a SAM tag | off |
| `--read-prior-tag` | SAM tag holding Phred-encoded fetal posteriors | XF |
| `--nhap` | Pre-select best-matching N haplotypes | all |
| `--random-init` | Random haplotype init + iterative re-selection | off |
| `--gibbs` | Diploid: use Gibbs sampling on read labels | off |
| `--knew` | New haplotypes added per iteration with `--random-init` | `nhap` |
| `--fulliter` | Diploid mean-field / triploid full-panel iterations | 3 |
| `--partiter` | Max label-reassignment iterations (triploid) | until convergence |
| `--maxnreads` | Limit total reads considered | — |
| `--nthreads` | OpenMP threads / per-label pool size (default: `--nproc`) | — |
| `--genetic-map` | PLINK-format genetic map (overrides uniform `--avgr`) | — |
| `--nophase` | Disable final phasing iteration | off |
| `--dump` | Dump gamma/emission/sigma/hap-path matrices per iteration | — |

### `plot` — Plot fragmentome embedding

Plot samples in the UMAP fragmentome embedding space.

```bash
cfstats plot --mapping umap_model.pkl -r hg38.fa sample.cram
cfstats plot --mapping umap_model.pkl -r hg38.fa --outfile embedding.png sample.cram
```

| Flag | Description | Default |
|------|-------------|---------|
| `--mapping` | Pickled `(reducer, xlim, ylim)` UMAP mapping | bundled |
| `--outfile` | File to save the plot | — |
| `--coords`, `-c` | File to save per-sample UMAP coordinates (`filename`, `x`, `y`); use `-` for stdout | `-` |

### `fragmentome` — Interactive fragmentome explorer

Launch a web application (Dash) for interactive exploration of a fragmentome
database stored in ClickHouse.

```bash
cfstats fragmentome --ch-host localhost --ch-port 8123
cfstats fragmentome --hf-token $HF_TOKEN   # use remote UMAP API for uploads
```

| Flag | Description | Default |
|------|-------------|---------|
| `--mapping` | Pickled UMAP mapping for upload-to-embedding | — |
| `--hf-token` | HF token for remote UMAP API | — |
| `--ch-host` | ClickHouse server host | localhost |
| `--ch-port` | ClickHouse HTTP port | 8123 |
| `--admin-password` | Password for the `/admin` upload page | env var |

## Models

Pre-trained models for `dnase1l3`, `ff`, and the UMAP fragmentome embedding are
**not included** in this repository. They are hosted privately and available
under a **research-only licence** (see `cfstats/models/LICENSE`).

There are two ways to use the models:

### 1. Remote API (recommended)

The models are served via a private Hugging Face Space. Pass `--hf-token` to any
model-dependent subcommand and inference runs server-side — no local model files
or GPU needed.

```bash
export HF_TOKEN="hf_..."
cfstats dnase1l3 --hf-token $HF_TOKEN -r hg38.fa sample.cram
cfstats ff        --hf-token $HF_TOKEN -r hg38.fa sample.cram
```

To obtain an API token, contact the author (see below).

### 2. Local model files

If you have received the model files directly from the author, point `--model`
to the file:

```bash
cfstats dnase1l3 --model /path/to/SVC_all_k4.joblib --confirm-licence -r hg38.fa sample.cram
cfstats ff       --model /path/to/ffpredictor_50kautosomalbins.pickle --confirm-licence -r hg38.fa sample.cram
```

You will be asked to accept the research-only licence interactively (use
`--confirm-licence` to skip in scripts).

## Practical examples

### Extract all features for a single sample

```bash
# Fragment size distribution (frequencies)
cfstats fszd --norm freq -r hg38.fa sample.cram > sample.fszd.tsv

# Cleavage-site motifs (frequencies)
cfstats csm --norm freq -r hg38.fa sample.cram > sample.csm.tsv

# 5′-end patterns (frequencies)
cfstats 5pends --norm freq -r hg38.fa sample.cram > sample.5pends.tsv

# Bin counts (50 kb bins, for fetal fraction)
cfstats bincounts -r hg38.fa -b 50000 -q 1 -F 1024 sample.cram > sample.bincounts.tsv

# Fourier coverage / Leuven WPS profile
cfstats fourier --leuven -r hg38.fa sample.cram > sample.fourier.tsv

# Cell-type deconvolution
cfstats deconv -r hg38.fa --output sample.deconv.tsv sample.cram

# GC-corrected site profile (run gcbias first)
cfstats gcbias -r hg38.fa sample.cram
cfstats siteprofile -r hg38.fa --gc-bias sample.GC_bias.txt sites.bed sample.cram > sample.siteprofile.tsv
```

### Batch processing

```bash
# Create a file list
ls /data/crams/*.cram > samples.txt

# Extract fragment size distributions for all samples (8 threads)
cfstats fszd --bamlist samples.txt --nproc 8 --norm freq -r hg38.fa > all.fszd.tsv
```

### Full pipeline: extract features + predict

```bash
# DNASE1L3 activity (remote API)
cfstats dnase1l3 --hf-token $HF_TOKEN -r hg38.fa sample.cram

# Fetal fraction (remote API)
cfstats ff --hf-token $HF_TOKEN -r hg38.fa sample.cram
```

## Package structure

```
cfstats/
├── __main__.py        # CLI entry point and argument parsing
├── fszd.py            # Fragment size distribution
├── csm.py             # Cleavage-site motifs
├── fpends.py          # 5′-end sequence patterns
├── bincounts.py       # Genome-wide bin counts
├── delfi.py           # DELFI-like short/long ratio
├── dnase1l3.py        # DNASE1L3 prediction logic
├── ff.py              # Fetal fraction estimation
├── nipt.py            # NIPT aneuploidy screening
├── ft.py              # Fourier-transformed coverage
├── deconv.py          # Cell-type deconvolution from FFT-WPS
├── nucs.py            # Nucleosome calling (WPS)
├── gcbias.py          # Per-sample GC-bias correction tables
├── siteprofile.py     # Composite nucleosome site profiles
├── sites.py           # Site-list loading helpers
├── fragmentome.py     # Interactive Dash explorer
├── db.py              # ClickHouse database layer
├── utils.py           # Shared utilities
├── models/            # Model loading + remote API helpers
│   ├── __init__.py
│   ├── *.joblib / *.pickle
│   └── LICENSE
├── assets/            # Bundled gene models / static files
└── impute/            # Genotype imputation HMM
    ├── __init__.py
    ├── cli.py
    ├── core.py
    └── _hmm.c
```

## License

### Code — MIT License

All source code is released under the [MIT License](LICENSE).

### Models — Research Use Only

Trained models are provided strictly for **non-commercial, academic, and
research purposes**. See [`cfstats/models/LICENSE`](cfstats/models/LICENSE) for
full terms. For commercial or clinical licensing enquiries, contact the author.

## Contact

**Jasper Linthorst** — jasper.linthorst@gmail.com

- Model files and API tokens are available on request for research purposes.
- For commercial licensing, please get in touch.
