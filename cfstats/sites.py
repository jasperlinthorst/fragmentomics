"""Site-list loading helpers for Griffin-style nucleosome coverage profiling.

Sites can be provided either as:

- a Griffin-style YAML mapping site-list names to per-site tab-separated files
  (optionally nested under a top-level ``site_lists`` key), or
- a single BED file (in which case the file name is used as the site-list name).

All loaders return a dict ``{site_list_name: DataFrame}`` where each DataFrame
has normalised ``chrom``, ``position`` and ``strand`` columns.
"""

import os
import logging

import pandas as pd

log = logging.getLogger(__name__)

FW_MARKERS = ("+", 1, "1")
RV_MARKERS = ("-", -1, "-1")


def _normalise_strand(value):
    if value in FW_MARKERS:
        return "+"
    if value in RV_MARKERS:
        return "-"
    return "."


def _read_site_table(path, chrom_column, position_column, strand_column, chroms):
    """Read a Griffin-style per-site tab-separated file."""
    df = pd.read_csv(path, sep="\t")

    missing = [c for c in (chrom_column, position_column) if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    strand = df[strand_column] if strand_column in df.columns else "."

    out = pd.DataFrame(
        {
            "chrom": df[chrom_column].astype(str),
            "position": df[position_column].astype(int),
            "strand": strand,
        }
    )
    out["strand"] = out["strand"].map(_normalise_strand)

    if chroms is not None:
        out = out[out["chrom"].isin(set(chroms))]

    return out.reset_index(drop=True)


def _read_bed(path, chroms):
    """Read a BED file; the site position is the interval midpoint."""
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    if df.shape[1] < 3:
        raise ValueError(f"{path}: BED file needs at least 3 columns (chrom, start, end).")

    chrom = df[0].astype(str)
    start = df[1].astype(int)
    end = df[2].astype(int)
    position = ((start + end) // 2).astype(int)

    if df.shape[1] >= 6:
        strand = df[5].map(_normalise_strand)
    else:
        strand = pd.Series(["."] * len(df))

    out = pd.DataFrame({"chrom": chrom, "position": position, "strand": strand})

    if chroms is not None:
        out = out[out["chrom"].isin(set(chroms))]

    return out.reset_index(drop=True)


def load_sites(path, chrom_column="Chrom", position_column="position",
               strand_column="Strand", chroms=None):
    """Load site lists from a YAML or BED file.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping of site-list name to a DataFrame with ``chrom``, ``position``
        and ``strand`` columns.
    """
    lower = path.lower()

    if lower.endswith((".yaml", ".yml")):
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh)

        site_lists = data.get("site_lists", data) if isinstance(data, dict) else data
        if not isinstance(site_lists, dict):
            raise ValueError(
                f"{path}: expected a mapping of site-list names to files "
                f"(optionally under a 'site_lists' key)."
            )

        out = {}
        for name, site_file in site_lists.items():
            # Resolve relative paths relative to the YAML file location.
            if not os.path.isabs(site_file):
                site_file = os.path.join(os.path.dirname(os.path.abspath(path)), site_file)
            log.info("Loading site list '%s' from %s", name, site_file)
            out[name] = _read_site_table(
                site_file, chrom_column, position_column, strand_column, chroms
            )
        return out

    # Otherwise treat as a single BED file.
    name = os.path.splitext(os.path.basename(path))[0]
    log.info("Loading site list '%s' from BED %s", name, path)
    return {name: _read_bed(path, chroms)}


from cfstats.utils import collect_bam_files
