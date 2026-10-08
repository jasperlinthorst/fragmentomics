"""Tests for cfstats.ff using mock models and mocked bincounts."""

import io
import logging
import pickle as real_pickle

import numpy as np
import pytest
from unittest import mock


class _FakeClf:
    """Module-level fake classifier so it is picklable."""
    def predict(self, X):
        return np.array([0.12] * len(X))


class TestFf:
    def _feats(self, n=3):
        return [f"chr1_{i*50000}_{(i+1)*50000}" for i in range(n)]

    def _mock_context(self, feats, fake_columns, fake_counts):
        """Context manager that mocks bincounts and pickle.load(open(...))."""
        import pickle as real_pickle
        model_bytes = real_pickle.dumps((_FakeClf(), feats))

        m_open = mock.mock_open(read_data=model_bytes)
        return mock.patch("cfstats.ff.bincounts") , \
               mock.patch("builtins.open", m_open)

    def test_ff_returns_predictions(self, make_args):
        """ff() with cmdline=False should return an array of predicted fetal fractions."""
        feats = self._feats(3)
        fake_columns = feats + ["chr1_extra_0"]
        fake_counts = np.array([[100, 200, 150, 50]])

        import pickle as real_pickle
        model_bytes = real_pickle.dumps((_FakeClf(), feats))

        with mock.patch("cfstats.ff.bincounts") as mock_bc, \
             mock.patch("builtins.open", mock.mock_open(read_data=model_bytes)):
            mock_bc.bincounts.return_value = (fake_columns, fake_counts)

            from cfstats.ff import ff
            args = make_args(model="dummy.pickle")
            result = ff(args, cmdline=False)

        assert result is not None
        assert len(result) == 1
        assert result[0] == pytest.approx(0.12)

    def test_ff_cmdline_writes_stdout(self, make_args, capsys):
        """ff() with cmdline=True should write predictions to stdout."""
        feats = self._feats(3)
        fake_columns = feats + ["chr1_extra_0"]
        fake_counts = np.array([[100, 200, 150, 50]])

        import pickle as real_pickle
        model_bytes = real_pickle.dumps((_FakeClf(), feats))

        with mock.patch("cfstats.ff.bincounts") as mock_bc, \
             mock.patch("builtins.open", mock.mock_open(read_data=model_bytes)):
            mock_bc.bincounts.return_value = (fake_columns, fake_counts)

            from cfstats.ff import ff
            args = make_args(model="dummy.pickle")
            ff(args, cmdline=True)

        captured = capsys.readouterr()
        assert "0.12" in captured.out

    def test_ff_warns_and_prefixes_chr_when_reference_lacks_prefix(self, make_args, caplog):
        """If the reference uses non-prefixed contig names, 'chr' should be
        prefixed to the bincount columns so the model features still match."""
        feats = self._feats(2)
        fake_columns = ["1_0_50000", "1_50000_100000"]
        fake_counts = np.array([[100, 200]])
        model_bytes = real_pickle.dumps((_FakeClf(), feats))

        with mock.patch("cfstats.ff.bincounts") as mock_bc, \
             mock.patch("builtins.open", mock.mock_open(read_data=model_bytes)):
            mock_bc.bincounts.return_value = (fake_columns, fake_counts)

            from cfstats.ff import ff
            args = make_args(model="dummy.pickle")
            with caplog.at_level(logging.WARNING, logger="cfstats.ff"):
                result = ff(args, cmdline=False)

        assert result is not None
        assert len(result) == 1
        assert result[0] == pytest.approx(0.12)
        assert "not 'chr'-prefixed" in caplog.text

    def test_ff_does_not_warn_when_columns_already_prefixed(self, make_args, caplog):
        """No opportunistic prefixing/warning is needed when everything is already
        'chr'-prefixed."""
        feats = self._feats(2)
        fake_columns = list(feats)
        fake_counts = np.array([[100, 200]])
        model_bytes = real_pickle.dumps((_FakeClf(), feats))

        with mock.patch("cfstats.ff.bincounts") as mock_bc, \
             mock.patch("builtins.open", mock.mock_open(read_data=model_bytes)):
            mock_bc.bincounts.return_value = (fake_columns, fake_counts)

            from cfstats.ff import ff
            args = make_args(model="dummy.pickle")
            with caplog.at_level(logging.WARNING, logger="cfstats.ff"):
                result = ff(args, cmdline=False)

        assert result is not None
        assert result[0] == pytest.approx(0.12)
        assert "not 'chr'-prefixed" not in caplog.text
