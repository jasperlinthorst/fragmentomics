"""Tests for cfstats.utils — pure helper functions."""

import numpy as np
import pandas as pd
import pytest

from cfstats import utils


# ---------------------------------------------------------------------------
# revcomp
# ---------------------------------------------------------------------------

class TestRevcomp:
    def test_basic(self):
        assert utils.revcomp("ACGT") == "ACGT"

    def test_palindrome(self):
        assert utils.revcomp("AATT") == "AATT"

    def test_asymmetric(self):
        assert utils.revcomp("AAAA") == "TTTT"
        assert utils.revcomp("CCCC") == "GGGG"

    def test_lowercase(self):
        assert utils.revcomp("acgt") == "acgt"

    def test_purpyr(self):
        assert utils.revcomp("RY") == "RY"


# ---------------------------------------------------------------------------
# allk
# ---------------------------------------------------------------------------

class TestAllk:
    def test_length_k1(self):
        assert utils.allk(1) == ["A", "C", "G", "T"]

    def test_length_k2(self):
        kmers = utils.allk(2)
        assert len(kmers) == 16

    def test_length_k4(self):
        kmers = utils.allk(4)
        assert len(kmers) == 256

    def test_lexsmallest_k2(self):
        kmers = utils.allk(2, onlylexsmallest=True)
        # Each kmer should be <= its reverse complement
        for km in kmers:
            assert km <= utils.revcomp(km)

    def test_lexsmallest_reduces_count(self):
        full = utils.allk(4)
        reduced = utils.allk(4, onlylexsmallest=True)
        assert len(reduced) < len(full)
        assert len(reduced) == 136  # known count for k=4


# ---------------------------------------------------------------------------
# allkp (purine/pyrimidine)
# ---------------------------------------------------------------------------

class TestAllkp:
    def test_length_k1(self):
        assert utils.allkp(1) == ["R", "Y"]

    def test_length_k2(self):
        assert len(utils.allkp(2)) == 4

    def test_lexsmallest_k2(self):
        kmers = utils.allkp(2, onlylexsmallest=True)
        for km in kmers:
            assert km <= utils.revcomp(km)


# ---------------------------------------------------------------------------
# nuc2purpyr
# ---------------------------------------------------------------------------

class TestNuc2purpyr:
    def test_purines(self):
        assert utils.nuc2purpyr("AG") == "RR"

    def test_pyrimidines(self):
        assert utils.nuc2purpyr("CT") == "YY"

    def test_mixed(self):
        assert utils.nuc2purpyr("ACGT") == "RYRY"


# ---------------------------------------------------------------------------
# gc_correct_counts (with synthetic data)
# ---------------------------------------------------------------------------

class TestGcCorrectCounts:
    def test_output_shape_matches_input(self):
        n_bins = 50
        rng = np.random.RandomState(0)
        counts = pd.DataFrame(
            rng.poisson(100, size=(3, n_bins)).astype(float),
            columns=[f"bin_{i}" for i in range(n_bins)],
        )
        gc = pd.Series(
            np.linspace(0.3, 0.7, n_bins),
            index=counts.columns,
            name="gc_content",
        )
        corrected = utils.gc_correct_counts(counts, gc, frac=0.5)
        assert corrected.shape == counts.shape

    def test_skips_low_count_rows(self):
        counts = pd.DataFrame(
            [[0.0] * 20],
            columns=[f"b{i}" for i in range(20)],
        )
        gc = pd.Series(np.linspace(0.3, 0.7, 20), index=counts.columns)
        corrected = utils.gc_correct_counts(counts, gc, frac=0.5)
        # Row with sum==0 should be returned as-is
        np.testing.assert_array_equal(corrected.values, counts.values)
