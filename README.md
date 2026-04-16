# cfstats — cell-free DNA fragmentomics toolkit

A command-line toolkit for extracting fragmentomic features from cell-free DNA
sequencing data (BAM/CRAM). It provides subcommands for fragment size
distributions, cleavage-site motifs, 5′-end sequence patterns, genome-wide bin
counts, nucleosome positioning, and more. Pre-trained models for DNASE1L3
activity prediction, fetal-fraction estimation and UMAP-based fragmentome
embedding are available separately (see [Models](#models) below).

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

# DNASE1L3 prediction via remote API
cfstats dnase1l3 --hf-token $HF_TOKEN -r hg38.fa sample.cram

# Fetal fraction estimation via remote API
cfstats ff --hf-token $HF_TOKEN -r hg38.fa sample.cram
```

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
| `--bamlist` | File with one BAM/CRAM path per line | — |

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

Additional flags: `-l`, `-u`, `--noinsert` (see `fszd`).

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
| `--bamlist` | File with one BAM/CRAM path per line | — |

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
cfstats fourier -r hg38.fa genes.gff sample.cram
```

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

### `plot` — Plot fragmentome embedding

Plot samples in the UMAP fragmentome embedding space.

```bash
cfstats plot --mapping umap_model.pkl -r hg38.fa sample.cram
```

### `fragmentome` — Interactive fragmentome explorer

Launch a web application (Dash) for interactive exploration of a fragmentome
database stored in ClickHouse.

```bash
cfstats fragmentome --ch-host localhost --ch-port 8123
cfstats fragmentome --hf-token $HF_TOKEN   # use remote UMAP API for uploads
```

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
├── nucs.py            # Nucleosome calling (WPS)
├── fragmentome.py     # Interactive Dash explorer
├── db.py              # ClickHouse database layer
├── utils.py           # Shared utilities
└── models/
    ├── __init__.py    # Model loading + remote API helpers
    └── LICENSE        # Research-only model licence
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
