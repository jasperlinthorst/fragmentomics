"""Tests for cfstats.ff using mock models and mocked bincounts."""

import io

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
            args = make_args(predictor="dummy.pickle")
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
            args = make_args(predictor="dummy.pickle")
            ff(args, cmdline=True)

        captured = capsys.readouterr()
        assert "0.12" in captured.out
