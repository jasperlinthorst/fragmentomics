"""Integration tests for cfstats.fszd using a tiny synthetic BAM."""

import numpy as np
import pytest

from cfstats import fszd


class TestFszd:
    def test_returns_list_of_arrays(self, make_args):
        args = make_args(lower=60, upper=600, insertissize=True, bamlist=None)
        result = fszd.fszd(args, cmdline=False)
        assert isinstance(result, list)
        assert len(result) == 1  # one BAM
        assert isinstance(result[0], np.ndarray)

    def test_output_length_matches_range(self, make_args):
        lo, hi = 100, 400
        args = make_args(lower=lo, upper=hi, insertissize=True, bamlist=None)
        result = fszd.fszd(args, cmdline=False)
        assert len(result[0]) == hi - lo

    def test_freq_normalisation_sums_to_one(self, make_args):
        args = make_args(lower=60, upper=600, norm="freq", insertissize=True, bamlist=None)
        result = fszd.fszd(args, cmdline=False)
        total = result[0].sum()
        if total > 0:
            np.testing.assert_almost_equal(total, 1.0, decimal=5)

    def test_counts_non_negative(self, make_args):
        args = make_args(lower=60, upper=600, insertissize=True, bamlist=None)
        result = fszd.fszd(args, cmdline=False)
        assert np.all(result[0] >= 0)

    def test_noinsert_mode(self, make_args):
        """insertissize=False should use query_length instead of template_length."""
        args = make_args(lower=60, upper=600, insertissize=False, bamlist=None)
        result = fszd.fszd(args, cmdline=False)
        assert isinstance(result[0], np.ndarray)
        assert len(result[0]) == 540
