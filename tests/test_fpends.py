"""Integration tests for cfstats.fpends using a tiny synthetic BAM."""

import numpy as np
import pytest

from cfstats import fpends


class TestFivePrimeEnds:
    def test_returns_list_of_arrays(self, make_args):
        args = make_args(k=4, useref=False, uselexsmallest=False, norm="counts")
        result = fpends._5pends(args, cmdline=False)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)

    def test_output_length_k4_all_kmers(self, make_args):
        """k=4, uselexsmallest=False -> 256 kmers."""
        args = make_args(k=4, useref=False, uselexsmallest=False, norm="counts")
        result = fpends._5pends(args, cmdline=False)
        assert len(result[0]) == 256

    def test_output_length_k4_lexsmallest(self, make_args):
        """k=4, uselexsmallest=True -> 136 kmers."""
        args = make_args(k=4, useref=False, uselexsmallest=True, norm="counts")
        result = fpends._5pends(args, cmdline=False)
        assert len(result[0]) == 136

    def test_freq_normalisation(self, make_args):
        args = make_args(k=4, useref=False, uselexsmallest=False, norm="freq")
        result = fpends._5pends(args, cmdline=False)
        total = result[0].sum()
        if total > 0:
            np.testing.assert_almost_equal(total, 1.0, decimal=5)

    def test_counts_non_negative(self, make_args):
        args = make_args(k=4, useref=False, uselexsmallest=False, norm="counts")
        result = fpends._5pends(args, cmdline=False)
        assert np.all(result[0] >= 0)

    def test_useref_mode(self, make_args):
        """useref=True should use the reference sequence for k-mer extraction."""
        args = make_args(k=4, useref=True, uselexsmallest=False, norm="counts")
        result = fpends._5pends(args, cmdline=False)
        assert isinstance(result[0], np.ndarray)
        assert len(result[0]) == 256
