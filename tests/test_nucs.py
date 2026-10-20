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
        chrom, nuc_start, nuc_end, center, score = peaks[0]
        assert chrom == "chr1"
        assert center == 1500  # start + index
        assert (nuc_start, nuc_end) == (1500, 1501)
        assert score == 100.0

    def test_two_peaks_far_apart(self):
        signal = np.zeros(2000)
        signal[200] = 50
        signal[800] = 60
        peaks = _call_nucleosomes_from_wps(
            signal, start=0, chrom="chr1", min_prominence=5, min_distance=147
        )
        assert len(peaks) == 2
        centers = [p[3] for p in peaks]
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
        assert peaks[0][3] == 450

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
        assert peaks[0][3] == 50100

    def test_width_inferred_from_peak(self):
        """Nucleosome extent should follow the peak shape (half prominence by default)."""
        x = np.arange(2000)
        signal = 100 * np.exp(-0.5 * ((x - 1000) / 40.0) ** 2)
        (chrom, nuc_start, nuc_end, summit, score), = _call_nucleosomes_from_wps(
            signal, start=0, chrom="chr1")
        assert summit == 1000
        fwhm = 2 * np.sqrt(2 * np.log(2)) * 40
        assert abs((nuc_end - nuc_start) - fwhm) <= 2
        assert nuc_start < summit < nuc_end
        # Wider rel_height gives a wider nucleosome
        (_, s2, e2, _, _), = _call_nucleosomes_from_wps(
            signal, start=0, chrom="chr1", rel_height=0.9)
        assert s2 < nuc_start and e2 > nuc_end


class TestNucsCommand:
    def _args(self, make_args, **kw):
        return make_args(chrom="chr1", start=None, end=None, k=120, min_len=120, max_len=300,
                         min_prominence=1, min_distance=147, rel_height=0.5,
                         chunk_size=3000, **kw)

    def test_aggregates_files_and_writes_track(self, make_args, tiny_bam, tmp_path):
        from cfstats.ft import wps
        from cfstats.nucs import nucs

        track = tmp_path / "wps.bedGraph.gz"
        args = self._args(make_args, samfiles=[tiny_bam, tiny_bam], wps_out=str(track))
        result = nucs(args, cmdline=False)
        assert len(result) == 1
        for chrom, s, e, summit, score in result[0]["peaks"]:
            assert s <= summit < e

        import gzip
        with gzip.open(track, "rt") as fh:
            lines = fh.read().splitlines()
        assert lines[0].startswith("track type=bedGraph")
        dense = np.zeros(10000)
        prev_end = 0
        for line in lines[1:]:
            c, s, e, v = line.split("\t")
            s, e = int(s), int(e)
            assert c == "chr1" and s >= prev_end and e > s
            dense[s:e] = float(v)
            prev_end = e
        assert prev_end == 10000

        import pysam
        with pysam.AlignmentFile(tiny_bam) as bam:
            single = wps(bam, "chr1", 0, 10000, k=120, min_len=120, max_len=300, args=args)
        np.testing.assert_array_equal(dense, 2 * single)
