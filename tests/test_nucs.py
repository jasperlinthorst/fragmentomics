"""Tests for cfstats.nucs._call_nucleosomes_from_wps — pure peak-calling logic."""

import numpy as np
import pytest

from cfstats.nucs import _call_nucleosomes_from_wps


class TestCallNucleosomes:
    def test_empty_signal(self):
        assert _call_nucleosomes_from_wps(np.array([]), start=0, chrom="chr1") == []

    def test_none_signal(self):
        assert _call_nucleosomes_from_wps(None, start=0, chrom="chr1") == []

    def test_flat_signal_no_peaks(self):
        signal = np.ones(1000)
        peaks = _call_nucleosomes_from_wps(signal, start=0, chrom="chr1")
        assert len(peaks) == 0

    def test_single_peak(self):
        signal = np.zeros(1000)
        signal[500] = 100  # strong peak
        peaks = _call_nucleosomes_from_wps(
            signal, start=1000, chrom="chr1", min_prominence=5, min_distance=147
        )
        assert len(peaks) == 1
        chrom, center, score = peaks[0]
        assert chrom == "chr1"
        assert center == 1500  # start + index
        assert score == 100.0

    def test_two_peaks_far_apart(self):
        signal = np.zeros(2000)
        signal[200] = 50
        signal[800] = 60
        peaks = _call_nucleosomes_from_wps(
            signal, start=0, chrom="chr1", min_prominence=5, min_distance=147
        )
        assert len(peaks) == 2
        centers = [p[1] for p in peaks]
        assert 200 in centers
        assert 800 in centers

    def test_two_peaks_too_close_keeps_strongest(self):
        signal = np.zeros(1000)
        signal[400] = 50
        signal[450] = 80  # only 50 bp apart, below min_distance=147
        peaks = _call_nucleosomes_from_wps(
            signal, start=0, chrom="chr1", min_prominence=5, min_distance=147
        )
        # find_peaks keeps the tallest when distance constraint applies
        assert len(peaks) == 1
        assert peaks[0][1] == 450

    def test_min_prominence_filters_weak(self):
        signal = np.zeros(1000)
        signal[500] = 3  # below min_prominence=5
        peaks = _call_nucleosomes_from_wps(
            signal, start=0, chrom="chr1", min_prominence=5, min_distance=147
        )
        assert len(peaks) == 0

    def test_start_offset(self):
        """Peaks should be reported at genomic coordinate = start + array index."""
        signal = np.zeros(500)
        signal[100] = 50
        peaks = _call_nucleosomes_from_wps(
            signal, start=50000, chrom="chr2", min_prominence=5, min_distance=147
        )
        assert len(peaks) == 1
        assert peaks[0][0] == "chr2"
        assert peaks[0][1] == 50100
