"""Integration tests for cfstats.bincounts using a tiny synthetic BAM."""

import logging
import os

import numpy as np
import pytest

from cfstats import bincounts
from tests.conftest import _write_fasta, _write_bam


class TestBincounts:
    def test_returns_labels_and_counts(self, make_args):
        args = make_args(binsize=5000, gccorrect=False)
        labels, counts = bincounts.bincounts(args, cmdline=False)
        assert isinstance(labels, list)
        assert len(labels) > 0
        assert isinstance(counts, np.ndarray)
        assert counts.ndim == 2
        assert counts.shape[0] == 1  # one BAM file

    def test_label_format(self, make_args):
        args = make_args(binsize=5000, gccorrect=False)
        labels, _ = bincounts.bincounts(args, cmdline=False)
        # Labels should be chrom_start_end
        for lbl in labels:
            parts = lbl.split("_")
            assert len(parts) == 3
            assert parts[0] == "chr1"

    def test_total_counts_positive(self, make_args):
        args = make_args(binsize=5000, gccorrect=False)
        _, counts = bincounts.bincounts(args, cmdline=False)
        assert counts.sum() > 0

    def test_smaller_bins_more_labels(self, make_args):
        args_big = make_args(binsize=5000, gccorrect=False)
        args_small = make_args(binsize=1000, gccorrect=False)
        labels_big, _ = bincounts.bincounts(args_big, cmdline=False)
        labels_small, _ = bincounts.bincounts(args_small, cmdline=False)
        assert len(labels_small) > len(labels_big)

    def test_mapqual_filter(self, make_args):
        args_all = make_args(binsize=5000, gccorrect=False, mapqual=0)
        args_strict = make_args(binsize=5000, gccorrect=False, mapqual=255)
        _, counts_all = bincounts.bincounts(args_all, cmdline=False)
        _, counts_strict = bincounts.bincounts(args_strict, cmdline=False)
        assert counts_all.sum() >= counts_strict.sum()

    def test_missing_reference_raises(self, make_args):
        args = make_args(reference=None)
        with pytest.raises(ValueError, match="Reference file is required"):
            bincounts.bincounts(args, cmdline=False)

    def test_bamlist_none_works(self, make_args):
        """bamlist=None should not cause errors (regression for the getattr fix)."""
        args = make_args(bamlist=None, binsize=5000, gccorrect=False)
        labels, counts = bincounts.bincounts(args, cmdline=False)
        assert counts.sum() > 0

    def test_noname_with_multiprocess_raises(self, make_args):
        args = make_args(nproc=2, name=False, binsize=5000, gccorrect=False)
        with pytest.raises(ValueError, match="--noname cannot be used"):
            bincounts.bincounts(args, cmdline=True)

    def test_bam_fasta_chr_prefix_mismatch(self, make_args, tmp_path, caplog):
        """A BAM with '1' reference names mapped against a FASTA with 'chr1' should
        still produce counts under the FASTA contig names."""
        ref_path = tmp_path / "ref.fa"
        bam_path = str(tmp_path / "reads.bam")
        _write_fasta(str(ref_path), chrom="chr1", length=10000)
        _write_bam(bam_path, str(ref_path), chrom="1", n_pairs=20)

        args = make_args(
            reference=str(ref_path),
            samfiles=[bam_path],
            binsize=5000,
            gccorrect=False,
        )
        with caplog.at_level(logging.WARNING, logger="cfstats.bincounts"):
            labels, counts = bincounts.bincounts(args, cmdline=False)

        assert counts.sum() > 0
        assert all(lbl.split("_")[0] == "chr1" for lbl in labels)
        assert "differ from FASTA names by a 'chr' prefix" in caplog.text
