"""Integration tests for cfstats.csm using a tiny synthetic BAM."""

import numpy as np
import pytest

from cfstats import csm


class TestCleaveSiteMotifs:
    def test_returns_list_of_arrays(self, make_args):
        args = make_args(k=4, purpyr=False, norm="counts")
        result = csm.cleavesitemotifs(args, cmdline=False)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)

    def test_output_length_k4(self, make_args):
        """For k=4, lexsmallest=True (hardcoded in csm), expect 136 motifs."""
        args = make_args(k=4, purpyr=False, norm="counts")
        result = csm.cleavesitemotifs(args, cmdline=False)
        assert len(result[0]) == 136

    def test_freq_normalisation(self, make_args):
        args = make_args(k=4, purpyr=False, norm="freq")
        result = csm.cleavesitemotifs(args, cmdline=False)
        total = result[0].sum()
        if total > 0:
            np.testing.assert_almost_equal(total, 1.0, decimal=5)

    def test_counts_non_negative(self, make_args):
        args = make_args(k=4, purpyr=False, norm="counts")
        result = csm.cleavesitemotifs(args, cmdline=False)
        assert np.all(result[0] >= 0)

    def test_k2_shorter_output(self, make_args):
        args = make_args(k=2, purpyr=False, norm="counts")
        result = csm.cleavesitemotifs(args, cmdline=False)
        # k=2 lexsmallest -> 10 motifs
        assert len(result[0]) == 10
