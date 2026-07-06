"""``cfstats deconv`` - cell-type deconvolution of cfDNA from FFT-WPS profiles.

Idea (following Stanley et al. 2024, *Cell type signatures in cell-free DNA
fragmentation profiles reveal disease biology*):

- Around gene bodies, nucleosome positioning imprints a periodic signal on the
  cfDNA Windowed Protection Score (WPS). The intensity of that periodicity in
  the ~193-199 bp band (the mean nucleosome spacing) correlates with the
  transcriptional activity of the gene in the cells that shed the cfDNA.
- Given a single-cell transcriptomic reference (per-cell-type pseudobulk gene
  expression), the per-gene FFT-WPS vector of a cfDNA sample can be modelled as
  a non-negative mixture of the cell-type expression profiles. Solving that
  mixture (non-negative least squares) and normalising yields the fractional
  contribution of each cell type to the cfDNA pool.

The reference single-cell atlas is downloaded and cached under the hood
(``~/.cache/cfstats/deconv/``). The download URL is configurable via
``--atlas-url`` and any local ``.h5ad`` / pre-built matrix can be supplied via
``--reference-atlas``.
"""

from __future__ import annotations

import logging
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import pysam
import gffutils

from cfstats.ft import wps, fft_wps_intensity

log = logging.getLogger("cfstats.deconv")

# Tabula Sapiens dataset used in the exploratory notebooks (cellxgene). The
# download URL is only a *default*; the user can override it with --atlas-url or
# point --reference-atlas at a local file. cellxgene asset URLs can change, so a
# clear error is raised if the download fails.
DEFAULT_ATLAS_URL = (
    "https://datasets.cellxgene.cziscience.com/"
    "b225ee37-5e06-4e49-9c25-c3d7b5008dab.h5ad"
)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _cache_dir():
    d = os.environ.get(
        "CFSTATS_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "cfstats", "deconv"),
    )
    os.makedirs(d, exist_ok=True)
    return d


def _strip_gene_version(gene_id):
    """ENSG00000123456.7 -> ENSG00000123456 (leave non-ENSG ids untouched)."""
    if isinstance(gene_id, str) and gene_id.startswith("ENSG") and "." in gene_id:
        return gene_id.split(".")[0]
    return gene_id


def _download(url, dest):
    log.info("Downloading reference atlas:\n  %s\n  -> %s", url, dest)

    def _hook(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100.0, block_num * block_size * 100.0 / total_size)
            sys.stderr.write(f"\r  progress: {pct:5.1f}%")
            sys.stderr.flush()

    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_hook)
        sys.stderr.write("\n")
        os.replace(tmp, dest)
    except Exception as e:  # pragma: no cover - network dependent
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"Failed to download reference atlas from {url}: {e}\n"
            "Provide a local atlas with --reference-atlas (an .h5ad file or a "
            "pre-built genes x cell-types matrix as .parquet/.tsv/.csv), or a "
            "working URL with --atlas-url."
        )
    return dest


# ---------------------------------------------------------------------------
# Reference matrix (genes x cell types) construction
# ---------------------------------------------------------------------------

def _aggregate_h5ad(h5ad_path, cell_type_col=None, min_cells=10):
    """Aggregate a single-cell ``.h5ad`` into per-cell-type pseudobulk expression.

    Returns a DataFrame indexed by ENSG gene id with one column per cell type
    (mean expression across cells of that type, using the ``.X`` matrix which is
    log-normalised in cellxgene/Tabula Sapiens releases).
    """
    try:
        import anndata
        adata = anndata.read_h5ad(h5ad_path)
    except ImportError:
        try:
            import scanpy as sc
            adata = sc.read_h5ad(h5ad_path)
        except ImportError:
            raise RuntimeError(
                "Reading .h5ad atlases requires 'anndata' (or 'scanpy'). "
                "Install with: pip install anndata"
            )

    log.info("Loaded atlas %s (%d cells x %d genes)",
             os.path.basename(h5ad_path), adata.n_obs, adata.n_vars)

    # --- resolve gene ids to ENSG --------------------------------------
    var = adata.var
    if str(adata.var_names[0]).startswith("ENSG"):
        gene_ids = list(adata.var_names)
    else:
        gene_ids = None
        for col in ("gene_ids", "gene_id", "ensembl_id", "ensembl",
                    "ensembl_gene_id", "feature_id"):
            if col in var.columns:
                gene_ids = list(var[col])
                log.info("Using var column '%s' for ENSG gene ids", col)
                break
        if gene_ids is None:
            log.warning("No ENSG gene id column found; falling back to var_names")
            gene_ids = list(adata.var_names)
    gene_ids = [_strip_gene_version(g) for g in gene_ids]

    # --- resolve cell-type column --------------------------------------
    if cell_type_col is None:
        log.info("Available .obs columns: %s", list(adata.obs.columns))
        for col in ("cell_type_ontology_term_id", "free_annotation",
                    "cell_ontology_class", "cell_type_assigned",
                    "annotation", "celltype", "cell_type"):
            if col in adata.obs.columns:
                cell_type_col = col
                break
    if cell_type_col is None or cell_type_col not in adata.obs.columns:
        raise RuntimeError(
            "Could not find a cell-type column in the atlas .obs. "
            f"Available columns: {list(adata.obs.columns)}. "
            "Specify one with --cell-type-col."
        )
    log.info("Using cell-type column '%s' (%d unique types)",
             cell_type_col, adata.obs[cell_type_col].nunique())

    X = adata.X
    labels = adata.obs[cell_type_col].astype(str).values

    columns = {}
    for ct in pd.unique(labels):
        mask = labels == ct
        n = int(mask.sum())
        if n < min_cells:
            continue
        sub = X[mask]
        mean = sub.mean(axis=0)
        mean = np.asarray(mean).ravel()
        columns[ct] = mean
        log.debug("  %s: %d cells", ct, n)

    if not columns:
        raise RuntimeError("No cell type had >= %d cells." % min_cells)

    ref = pd.DataFrame(columns, index=gene_ids)
    # collapse duplicate ENSG ids (mean) that arise after version stripping
    ref = ref.groupby(level=0).mean()
    ref.index.name = "ENSG"
    log.info("Built pseudobulk reference: %d genes x %d cell types",
             ref.shape[0], ref.shape[1])
    return ref


def _read_prebuilt_matrix(path):
    if path.endswith((".parquet", ".pq")):
        ref = pd.read_parquet(path)
    elif path.endswith((".tsv", ".txt", ".tsv.gz")):
        ref = pd.read_csv(path, sep="\t", index_col=0)
    elif path.endswith((".csv", ".csv.gz")):
        ref = pd.read_csv(path, index_col=0)
    else:
        raise RuntimeError(f"Unrecognised reference matrix format: {path}")
    ref.index = [_strip_gene_version(g) for g in ref.index]
    ref = ref.groupby(level=0).mean()
    ref.index.name = "ENSG"
    return ref


def load_reference(args):
    """Return the reference expression matrix (genes x cell types).

    Resolution order:
      1. ``--reference-atlas`` local path (.h5ad or pre-built matrix).
      2. cached pseudobulk parquet in the cache dir.
      3. download atlas from ``--atlas-url`` (default: Tabula Sapiens), aggregate
         and cache.
    """
    ref_arg = getattr(args, "reference_atlas", None)
    cell_type_col = getattr(args, "cell_type_col", None)
    min_cells = getattr(args, "min_cells", 10)

    if ref_arg is not None:
        if not os.path.exists(ref_arg):
            raise RuntimeError(f"--reference-atlas not found: {ref_arg}")
        if ref_arg.endswith((".h5ad",)):
            return _aggregate_h5ad(ref_arg, cell_type_col, min_cells)
        return _read_prebuilt_matrix(ref_arg)

    cache = _cache_dir()
    pseudobulk = os.path.join(cache, "reference_pseudobulk.parquet")
    if os.path.exists(pseudobulk) and not getattr(args, "rebuild_reference", False):
        log.info("Loading cached reference: %s", pseudobulk)
        return _read_prebuilt_matrix(pseudobulk)

    url = getattr(args, "atlas_url", None) or DEFAULT_ATLAS_URL
    atlas_path = os.path.join(cache, os.path.basename(url.split("?")[0]))
    if not os.path.exists(atlas_path):
        _download(url, atlas_path)

    ref = _aggregate_h5ad(atlas_path, cell_type_col, min_cells)
    try:
        ref.to_parquet(pseudobulk)
        log.info("Cached pseudobulk reference: %s", pseudobulk)
    except Exception as e:  # pragma: no cover - pyarrow optional
        log.warning("Could not cache reference as parquet (%s); caching as tsv", e)
        ref.to_csv(pseudobulk.replace(".parquet", ".tsv"), sep="\t")
    return ref


# ---------------------------------------------------------------------------
# Per-sample FFT-WPS signal (keyed by ENSG)
# ---------------------------------------------------------------------------

def _open_gene_db(gfffile):
    db_filename = f"{gfffile}.db"
    if not os.path.exists(db_filename):
        log.info("Constructing gene DB (%s)...", db_filename)
        db = gffutils.create_db(
            gfffile, dbfn=db_filename, force=True, keep_order=True,
            merge_strategy="merge", sort_attribute_values=True)
        log.info("Done.")
    else:
        log.debug("Loading gene DB: %s", db_filename)
        db = gffutils.FeatureDB(db_filename, keep_order=True)
    return db


def compute_sample_fftwps(samfile, args, db=None):
    """Compute per-gene FFT-WPS intensity for one alignment file.

    Returns a ``pandas.Series`` indexed by ENSG gene id.
    """
    if db is None:
        db = _open_gene_db(args.gfffile)

    window = int(args.window)
    ampmin = float(args.ampmin)
    ampmax = float(args.ampmax)

    reference = args.reference if getattr(args, "reference", None) else None

    # WPS needs random access; make sure an index exists (auto-create if missing).
    has_index = (
        os.path.exists(samfile + ".bai")
        or os.path.exists(samfile + ".crai")
        or os.path.exists(os.path.splitext(samfile)[0] + ".bai")
        or os.path.exists(os.path.splitext(samfile)[0] + ".crai")
    )
    if not has_index:
        try:
            log.info("No index for %s; creating one...", samfile)
            pysam.index(samfile)
        except Exception as e:
            raise RuntimeError(
                f"{samfile} is not indexed and indexing failed ({e}). "
                "Index it first (e.g. 'samtools index')."
            )

    pysamfile = pysam.AlignmentFile(samfile, "rb", reference_filename=reference)
    samctgs = set(pysamfile.references)

    intensities = {}
    for gene in db.features_of_type("gene"):
        gene_id = None
        if "gene_id" in gene.attributes:
            gene_id = gene.attributes["gene_id"][0]
        elif "ID" in gene.attributes:
            gene_id = gene.attributes["ID"][0]
        if gene_id is None:
            continue
        gene_id = _strip_gene_version(gene_id)

        # resolve contig naming (chr prefix mismatch)
        if gene.chrom in samctgs:
            chrom = gene.chrom
        elif "chr" + gene.chrom in samctgs:
            chrom = "chr" + gene.chrom
        else:
            continue

        if gene.strand == "-":
            start = gene.end - window
            end = gene.end
        else:
            start = gene.start
            end = gene.start + window
        if start < 0:
            start = 0

        signal = wps(pysamfile, chrom, start, end)
        intensity = fft_wps_intensity(signal, ampmin=ampmin, ampmax=ampmax)
        if not np.isnan(intensity):
            # average duplicate gene ids
            if gene_id in intensities:
                intensities[gene_id] = 0.5 * (intensities[gene_id] + intensity)
            else:
                intensities[gene_id] = intensity

    pysamfile.close()
    log.info("%s: computed FFT-WPS for %d genes", samfile, len(intensities))
    return pd.Series(intensities, name=samfile)


def _worker(pl):
    samfile, args = pl
    try:
        return samfile, compute_sample_fftwps(samfile, args)
    except Exception as e:  # pragma: no cover
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed FFT-WPS for {samfile}: {e}")


# ---------------------------------------------------------------------------
# Deconvolution
# ---------------------------------------------------------------------------

def _zscore(a):
    """z-score, treating a constant/NaN vector as all-zeros."""
    a = np.asarray(a, dtype=float)
    sd = np.nanstd(a)
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(a)
    return (a - np.nanmean(a)) / sd


def _solve_nnls(Rmat, fvec):
    """Solve NNLS with a soft sum-to-one constraint; return (weights, residual)."""
    from scipy.optimize import nnls as _nnls
    n_types = Rmat.shape[1]
    row_scale = np.linalg.norm(Rmat) / np.sqrt(max(n_types, 1))
    penalty = max(row_scale, 1.0) * 10.0
    Aug = np.vstack([Rmat, penalty * np.ones((1, n_types))])
    bug = np.concatenate([fvec, [penalty]])
    weights, _ = _nnls(Aug, bug)
    residual = float(np.linalg.norm(Rmat @ weights - fvec))
    return weights, residual


def deconvolve(signal, ref, standardize=True, relationship="auto", n_bootstrap=0, rng=None):
    """Deconvolve a per-gene FFT-WPS ``signal`` against reference ``ref``.

    The per-gene FFT-WPS intensity is modelled as a non-negative mixture of the
    cell-type expression profiles (non-negative least squares) with a
    sum-to-one constraint, so the returned fractions are always a valid
    composition.

    Orientation: in the exploratory notebooks the FFT-WPS intensity correlates
    *negatively* with gene expression (highly transcribed gene bodies lose
    regular nucleosome phasing). Standard NNLS against the raw (positively
    oriented) expression reference therefore collapses to all-zero weights.
    ``relationship`` controls how the reference is oriented before solving:

    - ``"auto"`` (default): flip the reference sign when the aggregate
      signal/expression relationship is negative (data-driven, robust).
    - ``"negative"``: always flip (assume FFT-WPS decreases with expression).
    - ``"positive"``: never flip (assume FFT-WPS increases with expression).

    Args:
        signal (pd.Series): per-gene FFT-WPS intensity, indexed by ENSG.
        ref (pd.DataFrame): genes (ENSG, index) x cell-types (columns) expression.
        standardize (bool): z-score the signal and each reference column across
            the shared gene set before solving (puts FFT-WPS and expression on a
            common scale).
        relationship (str): one of ``"auto"``, ``"negative"``, ``"positive"``.
        n_bootstrap (int): number of bootstrap iterations (gene-level resampling)
            for estimating uncertainty. 0 disables bootstrapping.
        rng: numpy RandomState or None (uses np.random.default_rng()).

    Returns:
        tuple: (fractions, fit_residual, bootstrap_std)
            - fractions (pd.Series): non-negative fractional contribution per
              cell type (sums to 1), indexed by cell-type.
            - fit_residual (float): L2 norm of the model residual on the full
              gene set (lower = better fit).
            - bootstrap_std (pd.Series or None): per-cell-type standard deviation
              across bootstrap iterations, or None if n_bootstrap == 0.
    """
    common = ref.index.intersection(signal.dropna().index)
    if len(common) < 10:
        raise RuntimeError(
            f"Only {len(common)} genes shared between sample and reference; "
            "cannot deconvolve. Check that the GFF gene ids are ENSG."
        )

    R = ref.loc[common].astype(float)
    f = signal.loc[common].astype(float).values

    # drop reference columns that are entirely missing
    R = R.dropna(axis=1, how="all")

    if standardize:
        f = _zscore(f)
        R = R.apply(_zscore, axis=0, result_type="broadcast")

    Rmat = np.nan_to_num(R.values, nan=0.0)
    fvec = np.nan_to_num(f, nan=0.0)

    # --- orient the reference to the empirical signal/expression relationship ---
    # projections[c] is proportional to corr(expression_c, signal) when standardized.
    projections = Rmat.T @ fvec
    if relationship == "positive":
        sign = 1.0
    elif relationship == "negative":
        sign = -1.0
    else:  # auto
        sign = -1.0 if np.nansum(projections) < 0 else 1.0
    Rmat = Rmat * sign

    # --- solve NNLS on the full gene set -------------------------------------
    n_types = Rmat.shape[1]
    weights, residual = _solve_nnls(Rmat, fvec)
    total = weights.sum()
    if total <= 0:
        log.warning(
            "NNLS returned all-zero weights for %s; falling back to uniform "
            "fractions. Consider --relationship or --no-standardize.",
            signal.name,
        )
        weights = np.ones(n_types)
        total = weights.sum()
    fractions = pd.Series(weights / total, index=R.columns, name=signal.name)

    # --- bootstrap (gene-level resampling) -----------------------------------
    bootstrap_std = None
    if n_bootstrap > 0:
        _rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        n_genes = Rmat.shape[0]
        boot_fracs = np.empty((n_bootstrap, n_types))
        for i in range(n_bootstrap):
            idx = _rng.integers(0, n_genes, size=n_genes)
            w_b, _ = _solve_nnls(Rmat[idx], fvec[idx])
            t_b = w_b.sum()
            boot_fracs[i] = w_b / t_b if t_b > 0 else (np.ones(n_types) / n_types)
        bootstrap_std = pd.Series(
            boot_fracs.std(axis=0), index=R.columns, name=signal.name)

    return fractions, residual, bootstrap_std


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def deconv(args):
    """``cfstats deconv`` entry point.

    Computes per-gene FFT-WPS profiles for each input alignment file, loads (and
    caches) a single-cell reference atlas, deconvolves each sample into
    fractional cell-type contributions, and writes a TSV where rows are samples
    and columns are cell types.
    """
    ref = load_reference(args)

    # compute per-sample signals
    signals = {}
    db = _open_gene_db(args.gfffile)
    nproc = getattr(args, "nproc", 1) or 1
    if nproc > 1:
        from multiprocessing import Pool
        with Pool(nproc) as pool:
            for samfile, sig in pool.imap_unordered(
                    _worker, [(s, args) for s in args.samfiles]):
                signals[samfile] = sig
    else:
        for samfile in args.samfiles:
            signals[samfile] = compute_sample_fftwps(samfile, args, db=db)

    n_bootstrap = getattr(args, "n_bootstrap", 0) or 0

    # deconvolve each sample
    rows = {}
    residuals = {}
    boot_stds = {}
    for samfile in args.samfiles:
        sig = signals[samfile]
        try:
            fractions, residual, bootstrap_std = deconvolve(
                sig, ref,
                standardize=not args.no_standardize,
                relationship=getattr(args, "relationship", "auto"),
                n_bootstrap=n_bootstrap,
            )
        except RuntimeError as e:
            log.error("Deconvolution failed for %s: %s", samfile, e)
            continue
        rows[samfile] = fractions
        residuals[samfile] = residual
        if bootstrap_std is not None:
            boot_stds[samfile] = bootstrap_std

    if not rows:
        raise RuntimeError("No samples could be deconvolved.")

    result = pd.DataFrame(rows).T
    result.index.name = "sample"
    result = result.sort_index(axis=1)
    result.insert(0, "fit_residual", pd.Series(residuals))

    out = getattr(args, "output", "-") or "-"
    if out == "-":
        result.to_csv(sys.stdout, sep="\t")
    else:
        result.to_csv(out, sep="\t")
        log.info("Wrote deconvolution results to %s", out)

    if boot_stds:
        std_result = pd.DataFrame(boot_stds).T
        std_result.index.name = "sample"
        std_result = std_result.sort_index(axis=1)
        std_result.insert(0, "fit_residual", pd.Series(residuals))
        if out == "-":
            log.info("Bootstrap std written to stderr only (use -O to write to file).")
        else:
            std_path = out.replace(".tsv", "") + ".bootstrap_std.tsv"
            std_result.to_csv(std_path, sep="\t")
            log.info("Wrote bootstrap std to %s", std_path)

    return result
