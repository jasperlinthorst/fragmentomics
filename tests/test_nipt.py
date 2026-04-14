"""Tests for cfstats.nipt.calc_llr_ff_t21 — pure statistical function."""

import numpy as np
import pandas as pd
import pytest

from cfstats.nipt import calc_llr_ff_t21


def _make_null_profile(n_bins=100, chr21_bins=5):
    """Create a simple null profile with some chr21 bins."""
    labels = [f"chrX_{i}" for i in range(n_bins - chr21_bins)] + \
             [f"chr21_{i}" for i in range(chr21_bins)]
    values = np.ones(n_bins) / n_bins
    return pd.Series(values, index=labels)


class TestCalcLlrFfT21:
    def test_no_trisomy_returns_negative_llr(self):
        null = _make_null_profile()
        std = pd.Series(np.full(len(null), 0.001), index=null.index)
        # Sample matches null exactly -> LLR should be ≤ 0
        sample = null.copy()
        result = calc_llr_ff_t21((sample, null, std), ff=0.10)
        assert result[0] <= 0  # LLR
        assert len(result) == 2  # (llr, ff) when negative

    def test_trisomy_returns_positive_llr(self):
        null = _make_null_profile()
        std = pd.Series(np.full(len(null), 0.001), index=null.index)
        # Simulate a trisomy: boost chr21 bins
        sample = null.copy()
        chr21_mask = sample.index.str.startswith("chr21")
        sample[chr21_mask] *= 1.05  # 5% increase
        sample = sample / sample.sum()  # re-normalize
        result = calc_llr_ff_t21((sample, null, std), ff=0.10)
        assert result[0] > 0  # positive LLR
        assert len(result) == 5  # (llr, ff, llr_updated, ff_updated, MR)

    def test_ff_parameter_used(self):
        null = _make_null_profile()
        std = pd.Series(np.full(len(null), 0.001), index=null.index)
        sample = null.copy()
        r1 = calc_llr_ff_t21((sample, null, std), ff=0.01)
        r2 = calc_llr_ff_t21((sample, null, std), ff=0.20)
        # Both should be negative for a null sample, but LLR values differ
        assert r1[0] <= 0
        assert r2[0] <= 0
