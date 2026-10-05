"""Tests for cfstats.dnase1l3 using mock models and mocked csm."""

import numpy as np
import pytest
from unittest import mock


class _FakePCA:
    def transform(self, X):
        return X[:, :2]


class _FakeClfBinary:
    def predict(self, X):
        return np.array([0])


class _FakeClfGT:
    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.1, 0.7, 0.2]])


class _FakeReg:
    def predict(self, X):
        return np.array([0.456])


class TestDnase1l3:
    def test_dnase1l3_writes_output(self, make_args, capsys):
        """dnase1l3() should write prediction results to stdout."""
        import pickle as real_pickle

        fake_motifs = [np.random.rand(136)]
        fake_model = (_FakePCA(), _FakeClfBinary(), _FakeClfGT(), _FakeReg())
        model_bytes = real_pickle.dumps(fake_model)

        with mock.patch("cfstats.dnase1l3.csm") as mock_csm, \
             mock.patch("builtins.open", mock.mock_open(read_data=model_bytes)):
            mock_csm.cleavesitemotifs.return_value = fake_motifs

            from cfstats.dnase1l3 import dnase1l3
            args = make_args(clf="dummy.pickle")
            dnase1l3(args)

        captured = capsys.readouterr()
        assert "R206C genotype prediction" in captured.out
        assert "DNASE1L3 plasma activity regression" in captured.out
        assert "0.456" in captured.out


class TestPlotFragmentome:
    def test_plot_fragmentome_writes_coords_to_stdout(self, make_args, capsys, tmp_path):
        """plot_fragmentome() should write per-sample UMAP coordinates to stdout."""
        from cfstats.dnase1l3 import plot_fragmentome

        class FakeReducer:
            embedding_ = np.array([[1.0, 2.0], [3.0, 4.0]])
            def transform(self, X):
                return X[:, :2]

        fake_model = (FakeReducer(), None, None)

        args = make_args(
            samfiles=["/path/to/sample1.bam", "/path/to/sample2.bam"],
            mapping="dummy.pkl",
            coords="-",
            outfile=str(tmp_path / "plot.png"),
        )

        with mock.patch("cfstats.dnase1l3.joblib.load", return_value=fake_model), \
             mock.patch("cfstats.dnase1l3.fszd.fszd") as mock_fszd, \
             mock.patch("cfstats.dnase1l3.csm.cleavesitemotifs") as mock_csm, \
             mock.patch("cfstats.dnase1l3.fpends._5pends") as mock_fpends:
            mock_fszd.return_value = [np.zeros(10), np.zeros(10)]
            mock_csm.return_value = [np.zeros(20), np.zeros(20)]
            mock_fpends.return_value = [np.zeros(30), np.zeros(30)]

            plot_fragmentome(args)

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("/path/to/sample1.bam")
        assert lines[1].startswith("/path/to/sample2.bam")
        assert (tmp_path / "plot.png").exists()

    def test_plot_fragmentome_writes_coords_to_file(self, make_args, tmp_path):
        """plot_fragmentome() should write per-sample UMAP coordinates to a file."""
        from cfstats.dnase1l3 import plot_fragmentome

        class FakeReducer:
            embedding_ = np.array([[1.0, 2.0], [3.0, 4.0]])
            def transform(self, X):
                return X[:, :2]

        fake_model = (FakeReducer(), None, None)
        coords_path = tmp_path / "coords.tsv"

        args = make_args(
            samfiles=["/path/to/sample1.bam", "/path/to/sample2.bam"],
            mapping="dummy.pkl",
            coords=str(coords_path),
            outfile=str(tmp_path / "plot.png"),
            header=True,
        )

        with mock.patch("cfstats.dnase1l3.joblib.load", return_value=fake_model), \
             mock.patch("cfstats.dnase1l3.fszd.fszd") as mock_fszd, \
             mock.patch("cfstats.dnase1l3.csm.cleavesitemotifs") as mock_csm, \
             mock.patch("cfstats.dnase1l3.fpends._5pends") as mock_fpends:
            mock_fszd.return_value = [np.zeros(10), np.zeros(10)]
            mock_csm.return_value = [np.zeros(20), np.zeros(20)]
            mock_fpends.return_value = [np.zeros(30), np.zeros(30)]

            plot_fragmentome(args)

        content = coords_path.read_text()
        lines = content.strip().split("\n")
        assert lines[0].split("\t") == ["filename", "x", "y"]
        assert lines[1].startswith("/path/to/sample1.bam")
        assert lines[2].startswith("/path/to/sample2.bam")
