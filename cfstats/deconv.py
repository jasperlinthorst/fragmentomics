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
from cfstats.utils import collect_bam_files

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

def _decode(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _h5_read_strings(node):
    """Read an h5py string dataset into a list of python str."""
    return [_decode(x) for x in node[:]]


def _h5_column_strings(group, key):
    """Read an anndata dataframe column (dataset or categorical group) as str list."""
    import h5py
    node = group[key]
    if isinstance(node, h5py.Group) and "categories" in node:
        cats = _h5_read_strings(node["categories"])
        codes = np.asarray(node["codes"][:])
        return [cats[i] if i >= 0 else "nan" for i in codes]
    return _h5_read_strings(node)


def _h5_column_categorical(group, key):
    """Return (categories:list[str], codes:np.ndarray[int]) for an obs column.

    Handles both anndata categorical groups and plain string/label datasets.
    """
    import h5py
    node = group[key]
    if isinstance(node, h5py.Group) and "categories" in node:
        cats = _h5_read_strings(node["categories"])
        codes = np.asarray(node["codes"][:]).astype(np.int64)
        return cats, codes
    vals = pd.Series(_h5_read_strings(node)).astype("category")
    return list(vals.cat.categories), np.asarray(vals.cat.codes, dtype=np.int64)


def _h5_column_nunique(group, key):
    import h5py
    node = group[key]
    if isinstance(node, h5py.Group) and "categories" in node:
        return int(node["categories"].shape[0])
    return int(pd.Series(_h5_read_strings(node)).nunique())


def _aggregate_h5ad(h5ad_path, cell_type_col=None, min_cells=10, chunk_size=20000):
    """Aggregate a single-cell ``.h5ad`` into per-cell-type pseudobulk expression.

    Returns a DataFrame indexed by ENSG gene id with one column per cell type
    (mean expression across cells of that type, using the ``.X`` matrix which is
    log-normalised in cellxgene/Tabula Sapiens releases).

    The ``.X`` CSR matrix is streamed directly from the HDF5 file with
    ``h5py`` in row chunks of ``chunk_size`` cells: only the slice of
    ``X/data`` / ``X/indices`` for the current chunk is read from disk, so the
    full matrix (potentially millions of cells x 60k genes, tens of GB of
    non-zeros) is never held in memory at once. anndata's ``read_h5ad`` is
    intentionally avoided: even in backed mode it loads ``.obs``, ``.raw``,
    ``.obsm`` and ``.layers`` into RAM, which OOMs on large atlases. Per-cell-
    type sums are accumulated via a sparse one-hot matmul; peak memory is
    dominated by the ``n_cell_types x n_genes`` accumulator (a few hundred MB)
    plus one chunk of ``.X``.
    """
    import h5py
    import scipy.sparse as sp

    with h5py.File(h5ad_path, "r") as f:
        # --- matrix layout ---------------------------------------------
        Xg = f["X"]
        if isinstance(Xg, h5py.Group):
            enc = _decode(Xg.attrs.get("encoding-type", "csr_matrix"))
            shape = tuple(int(s) for s in Xg.attrs["shape"])
        else:
            enc = "array"
            shape = tuple(int(s) for s in Xg.shape)
        n_obs, n_genes = shape
        log.info("Atlas %s (%d cells x %d genes), X encoding=%s",
                 os.path.basename(h5ad_path), n_obs, n_genes, enc)
        if enc not in ("csr_matrix", "array"):
            raise RuntimeError(
                f"Unsupported X encoding '{enc}' (expected CSR or dense). "
                "Re-export the atlas as CSR, or pre-build a matrix and pass it "
                "via --reference-atlas."
            )

        # --- gene ids to ENSG ------------------------------------------
        var = f["var"]
        index_key = _decode(var.attrs.get("_index", "_index"))
        gene_ids = _h5_column_strings(var, index_key)
        if not str(gene_ids[0]).startswith("ENSG"):
            for col in ("ensembl_id", "gene_ids", "gene_id", "ensembl",
                        "ensembl_gene_id", "feature_id"):
                if col in var:
                    gene_ids = _h5_column_strings(var, col)
                    log.info("Using var column '%s' for ENSG gene ids", col)
                    break
            else:
                log.warning("No ENSG gene id column found; falling back to var index")
        gene_ids = [_strip_gene_version(g) for g in gene_ids]

        # --- resolve cell-type column ----------------------------------
        obs = f["obs"]
        obs_cols = list(obs.keys())
        # Candidate columns that hold human-readable cell-type labels. We never
        # use '*_ontology_term_id' columns: those hold CL: identifiers, not
        # names, and are just an encoded duplicate of the label column.
        candidates = (
            "cell_ontology_class", "free_annotation", "cell_type_assigned",
            "annotation", "celltype", "cell_type", "broad_cell_class",
        )
        if cell_type_col is None:
            log.info("Available .obs columns: %s", obs_cols)
            present = [c for c in candidates if c in obs_cols]
            if present:
                cardinalities = {c: _h5_column_nunique(obs, c) for c in present}
                log.info("Candidate cell-type columns (name: #types): %s", cardinalities)
                cell_type_col = max(cardinalities, key=cardinalities.get)
        if cell_type_col is None or cell_type_col not in obs_cols:
            raise RuntimeError(
                "Could not find a cell-type column in the atlas .obs. "
                f"Available columns: {obs_cols}. Specify one with --cell-type-col."
            )
        type_names, codes = _h5_column_categorical(obs, cell_type_col)
        n_all_types = len(type_names)
        log.info("Using cell-type column '%s' (%d unique types)",
                 cell_type_col, n_all_types)

        # Warn if the atlas is a single-tissue subset.
        if "tissue" in obs_cols:
            n_tissues = _h5_column_nunique(obs, "tissue")
            if n_tissues <= 1:
                log.warning(
                    "Atlas covers a single tissue with only %d cell types. "
                    "Cell-type resolution is limited to this tissue; use a full "
                    "multi-tissue atlas for hundreds of cell types.", n_all_types)

        # --- streaming per-cell-type aggregation -----------------------
        counts = np.bincount(codes[codes >= 0], minlength=n_all_types).astype(np.int64)
        sums = np.zeros((n_all_types, n_genes), dtype=np.float64)

        n_chunks = (n_obs + chunk_size - 1) // chunk_size
        if enc == "csr_matrix":
            data_ds = Xg["data"]
            indices_ds = Xg["indices"]
            indptr = np.asarray(Xg["indptr"][:])
        for ci, start in enumerate(range(0, n_obs, chunk_size)):
            end = min(start + chunk_size, n_obs)
            c = codes[start:end]
            valid = c >= 0
            if valid.any():
                if enc == "csr_matrix":
                    p0, p1 = int(indptr[start]), int(indptr[end])
                    d = np.asarray(data_ds[p0:p1], dtype=np.float64)
                    idx = np.asarray(indices_ds[p0:p1])
                    iptr = indptr[start:end + 1] - p0
                    Xc = sp.csr_matrix((d, idx, iptr), shape=(end - start, n_genes))
                else:  # dense
                    Xc = sp.csr_matrix(np.asarray(Xg[start:end], dtype=np.float64))
                rows = c[valid]
                cols = np.nonzero(valid)[0]
                onehot = sp.csr_matrix(
                    (np.ones(rows.size, dtype=np.float64), (rows, cols)),
                    shape=(n_all_types, end - start),
                )
                contrib = onehot @ Xc
                if sp.issparse(contrib):
                    contrib = contrib.toarray()
                sums += np.asarray(contrib, dtype=np.float64)
            if (ci + 1) % 5 == 0 or ci + 1 == n_chunks:
                log.info("  aggregated chunk %d/%d (through cell %d)",
                         ci + 1, n_chunks, end)

    # --- means, min-cells filter, DataFrame ----------------------------
    keep = counts >= min_cells
    if not keep.any():
        raise RuntimeError("No cell type had >= %d cells." % min_cells)
    means = sums[keep] / counts[keep][:, None]  # (n_keep x n_genes)
    kept_names = [type_names[i] for i in np.nonzero(keep)[0]]
    log.info("Kept %d/%d cell types (>= %d cells)",
             int(keep.sum()), n_all_types, min_cells)

    ref = pd.DataFrame(means.T, index=gene_ids, columns=kept_names)
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
    chunk_size = getattr(args, "chunk_size", 20000) or 20000
    rebuild = getattr(args, "rebuild_reference", False)
    cache = _cache_dir()

    if ref_arg is not None:
        if not os.path.exists(ref_arg):
            raise RuntimeError(f"--reference-atlas not found: {ref_arg}")
        if ref_arg.endswith((".h5ad",)):
            return _build_or_load_pseudobulk(
                ref_arg, cache, cell_type_col, min_cells, chunk_size, rebuild)
        return _read_prebuilt_matrix(ref_arg)

    url = getattr(args, "atlas_url", None) or DEFAULT_ATLAS_URL
    atlas_path = os.path.join(cache, os.path.basename(url.split("?")[0]))
    if not os.path.exists(atlas_path):
        _download(url, atlas_path)

    return _build_or_load_pseudobulk(
        atlas_path, cache, cell_type_col, min_cells, chunk_size, rebuild)


def _build_or_load_pseudobulk(atlas_path, cache, cell_type_col, min_cells,
                              chunk_size, rebuild):
    """Aggregate an .h5ad atlas to a pseudobulk matrix, caching the result.

    The cache key includes the atlas filename, the requested cell-type column
    and min-cells so different atlases / settings don't collide.
    """
    key = "%s.ct-%s.min-%d.pseudobulk" % (
        os.path.basename(atlas_path), cell_type_col or "auto", min_cells)
    pseudobulk = os.path.join(cache, key + ".parquet")
    if os.path.exists(pseudobulk) and not rebuild:
        log.info("Loading cached pseudobulk reference: %s", pseudobulk)
        return _read_prebuilt_matrix(pseudobulk)

    ref = _aggregate_h5ad(atlas_path, cell_type_col, min_cells, chunk_size)
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
    if gfffile.lower().endswith((".tsv", ".txt")):
        with open(gfffile) as handle:
            regions = []
            for line in handle:
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(
                        "Gene body annotation must have five columns: "
                        "gene_id chromosome start end strand")
                gene_id, chromosome, start, end, strand = fields
                regions.append((gene_id, chromosome, int(start), int(end), strand))
        return regions

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
    if isinstance(db, list):
        regions = db
    else:
        def gff_regions():
            for gene in db.features_of_type("gene"):
                if "gene_id" in gene.attributes:
                    gene_id = gene.attributes["gene_id"][0]
                elif "ID" in gene.attributes:
                    gene_id = gene.attributes["ID"][0]
                else:
                    continue
                if gene.strand == "-":
                    start, end = gene.end - window, gene.end
                else:
                    start, end = gene.start, gene.start + window
                yield gene_id, gene.chrom, start, end, gene.strand
        regions = gff_regions()

    for gene_id, gene_chrom, start, end, strand in regions:
        gene_id = _strip_gene_version(gene_id)

        # resolve contig naming (chr prefix mismatch)
        if gene_chrom in samctgs:
            chrom = gene_chrom
        elif "chr" + gene_chrom in samctgs:
            chrom = "chr" + gene_chrom
        else:
            continue
        if start < 0:
            start = 0

        signal = wps(pysamfile, chrom, start, end)
        intensity = float(np.nanmean(fft_wps_intensity(
            signal, ampmin=ampmin, ampmax=ampmax)))
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
    args.samfiles = collect_bam_files(args.samfiles, getattr(args, 'bamlist', None))

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
