"""Shared fixtures for cfstats tests.

Provides:
- tiny_ref: a small FASTA reference (one 10 kb chromosome "chr1")
- tiny_bam: a small indexed BAM with ~50 paired-end reads on chr1
- make_args: factory to build argparse.Namespace objects with sensible defaults
"""

import os
import tempfile
import argparse

import numpy as np
import pysam
import pytest


# ---------------------------------------------------------------------------
# Helper: create a minimal FASTA reference
# ---------------------------------------------------------------------------

def _write_fasta(path, chrom="chr1", length=10000):
    """Write a simple FASTA with a single chromosome of random ACGT sequence."""
    rng = np.random.RandomState(42)
    seq = "".join(rng.choice(list("ACGT"), size=length))
    with open(path, "w") as fh:
        fh.write(f">{chrom}\n")
        for i in range(0, len(seq), 80):
            fh.write(seq[i : i + 80] + "\n")
    pysam.faidx(path)
    return path


# ---------------------------------------------------------------------------
# Helper: create a minimal BAM with paired-end reads
# ---------------------------------------------------------------------------

def _write_bam(path, ref_path, chrom="chr1", n_pairs=50, seed=42):
    """Create a small sorted, indexed BAM with *n_pairs* proper pairs."""
    rng = np.random.RandomState(seed)

    header = pysam.AlignmentHeader.from_references(
        [chrom], [10000]
    )

    tmp_unsorted = path + ".unsorted.bam"
    with pysam.AlignmentFile(tmp_unsorted, "wb", header=header) as outf:
        for i in range(n_pairs):
            pos = int(rng.randint(100, 8000))
            isize = int(rng.randint(120, 300))
            seq_len = 100
            seq = "A" * seq_len
            qual = pysam.qualitystring_to_array("I" * seq_len)

            # Read 1 (forward)
            r1 = pysam.AlignedSegment(header)
            r1.query_name = f"read_{i}"
            r1.query_sequence = seq
            r1.flag = 99  # paired, proper pair, mate reverse, read1
            r1.reference_id = 0
            r1.reference_start = pos
            r1.mapping_quality = 60
            r1.cigar = [(0, seq_len)]  # 100M
            r1.next_reference_id = 0
            r1.next_reference_start = pos + isize - seq_len
            r1.template_length = isize
            r1.query_qualities = qual
            outf.write(r1)

            # Read 2 (reverse)
            r2 = pysam.AlignedSegment(header)
            r2.query_name = f"read_{i}"
            r2.query_sequence = seq
            r2.flag = 147  # paired, proper pair, reverse, read2
            r2.reference_id = 0
            r2.reference_start = pos + isize - seq_len
            r2.mapping_quality = 60
            r2.cigar = [(0, seq_len)]
            r2.next_reference_id = 0
            r2.next_reference_start = pos
            r2.template_length = -isize
            r2.query_qualities = qual
            outf.write(r2)

    # Sort and index
    pysam.sort("-o", path, tmp_unsorted)
    pysam.index(path)
    os.remove(tmp_unsorted)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tmp_data_dir():
    """Session-scoped temp directory that is cleaned up at the end."""
    with tempfile.TemporaryDirectory(prefix="cfstats_test_") as d:
        yield d


@pytest.fixture(scope="session")
def tiny_ref(tmp_data_dir):
    path = os.path.join(tmp_data_dir, "ref.fa")
    _write_fasta(path)
    return path


@pytest.fixture(scope="session")
def tiny_bam(tmp_data_dir, tiny_ref):
    path = os.path.join(tmp_data_dir, "reads.bam")
    _write_bam(path, tiny_ref)
    return path


@pytest.fixture()
def make_args(tiny_ref, tiny_bam):
    """Factory fixture: returns a function that creates an argparse.Namespace
    pre-filled with common defaults.  Override any attribute via kwargs."""

    def _make(**overrides):
        defaults = dict(
            samfiles=[tiny_bam],
            reference=tiny_ref,
            reqflag=None,
            exclflag=0,
            mapqual=0,
            x=1000000,
            nproc=1,
            norm="counts",
            maxo=None,
            header=False,
            name=True,
            seed=42,
            binsize=1000,
            gccorrect=False,
            frac=0.5,
            bamlist=None,
            k=4,
            purpyr=False,
            lower=60,
            upper=600,
            insertissize=True,
            useref=False,
            uselexsmallest=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    return _make
