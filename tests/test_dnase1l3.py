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
