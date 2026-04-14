"""Tests for cfstats.ft.wps — Windowed Protection Score calculation."""

import numpy as np
import pysam
import pytest

from cfstats.ft import wps


class TestWpsBasic:
    """Tests using the tiny BAM fixture from conftest."""

    def test_output_length_matches_region(self, tiny_bam, tiny_ref):
        bam = pysam.AlignmentFile(tiny_bam, "rb", reference_filename=tiny_ref)
        signal = wps(bam, "chr1", 1000, 2000, k=120, min_len=120, max_len=300)
        bam.close()
        assert len(signal) == 1000

    def test_empty_region_returns_empty(self, tiny_bam, tiny_ref):
        bam = pysam.AlignmentFile(tiny_bam, "rb", reference_filename=tiny_ref)
        signal = wps(bam, "chr1", 1000, 1000, k=120, min_len=120, max_len=300)
        bam.close()
        assert len(signal) == 0

    def test_returns_integer_array(self, tiny_bam, tiny_ref):
        bam = pysam.AlignmentFile(tiny_bam, "rb", reference_filename=tiny_ref)
        signal = wps(bam, "chr1", 500, 1500, k=120, min_len=120, max_len=300)
        bam.close()
        assert signal.dtype in (np.int32, np.int64, int)

    def test_no_reads_region_is_zero(self, tiny_bam, tiny_ref):
        """A region far from any reads should have all-zero WPS."""
        bam = pysam.AlignmentFile(tiny_bam, "rb", reference_filename=tiny_ref)
        # Reads are placed between 100-8300; 9500-9999 should be empty
        signal = wps(bam, "chr1", 9500, 9999, k=120, min_len=120, max_len=300)
        bam.close()
        assert np.all(signal == 0)


class TestWpsEdgeCases:
    def test_negative_region_returns_empty(self, tiny_bam, tiny_ref):
        bam = pysam.AlignmentFile(tiny_bam, "rb", reference_filename=tiny_ref)
        signal = wps(bam, "chr1", 2000, 1000)  # end < start
        bam.close()
        assert len(signal) == 0

    def test_small_window(self, tiny_bam, tiny_ref):
        bam = pysam.AlignmentFile(tiny_bam, "rb", reference_filename=tiny_ref)
        signal = wps(bam, "chr1", 1000, 2000, k=16, min_len=50, max_len=500)
        bam.close()
        assert len(signal) == 1000
