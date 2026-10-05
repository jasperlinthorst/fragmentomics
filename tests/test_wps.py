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


class TestLeuvenSpectrum:
    def test_matches_r_spec_pgram_reference(self):
        from argparse import Namespace
        from cfstats.ft import fft_wps_intensity
        positions = np.arange(10001)
        signal = (np.sin(2 * np.pi * positions / 196)
                  + 0.25 * np.cos(2 * np.pi * positions / 173)
                  + (positions % 7) / 10)
        expected = [12027.2373400631, 20315.0499619015, 13265.6167207427]
        args = Namespace(leuven=True, ampmin=193, ampmax=199, ampstep=3)
        np.testing.assert_allclose(
            fft_wps_intensity(signal, args=args),
            expected, rtol=1e-12, atol=1e-10)

    def test_amplitude_step_changes_period_list(self):
        from argparse import Namespace
        from cfstats.ft import fft_wps_intensity
        positions = np.arange(10001)
        signal = np.sin(2 * np.pi * positions / 196)
        args = Namespace(leuven=True, ampmin=193, ampmax=199, ampstep=3)
        values = fft_wps_intensity(signal, args=args)
        assert len(values) == 3
        args = Namespace(leuven=True, ampmin=193, ampmax=199, ampstep=2)
        values = fft_wps_intensity(signal, args=args)
        assert len(values) == 4  # 193, 195, 197, 199
        args = Namespace(leuven=True, ampmin=195, ampmax=205, ampstep=5)
        values = fft_wps_intensity(signal, args=args)
        assert len(values) == 3  # 195, 200, 205

    def test_default_step_is_one(self):
        from argparse import Namespace
        from cfstats.ft import fft_wps_intensity
        positions = np.arange(10001)
        signal = np.sin(2 * np.pi * positions / 196)
        args = Namespace(leuven=True, ampmin=193, ampmax=199)
        values = fft_wps_intensity(signal, args=args)
        assert len(values) == 7  # 193, 194, 195, 196, 197, 198, 199


class TestLeuvenWps:
    def test_output_includes_both_region_endpoints(self, tiny_bam, tiny_ref):
        from argparse import Namespace
        bam = pysam.AlignmentFile(tiny_bam, "rb", reference_filename=tiny_ref)
        args = Namespace(leuven=True, reqflag=1, exclflag=1548, mapqual=0)
        signal, covered = wps(
            bam, "chr1", 1000, 2000, k=120, min_len=120,
            max_len=300, args=args)
        bam.close()
        assert len(signal) == 1001
        assert isinstance(covered, (bool, np.bool_))

    def test_no_reads_region_is_zero_and_uncovered(self, tiny_bam, tiny_ref):
        from argparse import Namespace
        bam = pysam.AlignmentFile(tiny_bam, "rb", reference_filename=tiny_ref)
        args = Namespace(leuven=True, reqflag=1, exclflag=1548, mapqual=0)
        signal, covered = wps(
            bam, "chr1", 9500, 9999, k=120, min_len=120,
            max_len=300, args=args)
        bam.close()
        assert np.all(signal == 0)
        assert not covered
