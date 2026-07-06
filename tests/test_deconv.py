"""Tests for cfstats deconv: FFT-WPS intensity helper, reference parsing,
and the NNLS deconvolution math."""

import os

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# fft_wps_intensity helper (in cfstats.ft)
# ---------------------------------------------------------------------------

class TestFftWpsIntensity:
    def test_flat_signal_returns_nan(self):
        from cfstats.ft import fft_wps_intensity
        assert np.isnan(fft_wps_intensity(np.ones(2000)))

    def test_empty_signal_returns_nan(self):
        from cfstats.ft import fft_wps_intensity
        assert np.isnan(fft_wps_intensity(np.array([])))

    def test_periodic_signal_returns_finite(self):
        from cfstats.ft import fft_wps_intensity
        # a signal with ~196 bp periodicity should yield a finite intensity in
        # the 193-199 bp band
        x = np.arange(4000)
        signal = np.sin(2 * np.pi * x / 196.0)
        val = fft_wps_intensity(signal, ampmin=193, ampmax=199)
        assert np.isfinite(val)
        assert val > 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_strip_gene_version(self):
        from cfstats.deconv import _strip_gene_version
        assert _strip_gene_version("ENSG00000123456.7") == "ENSG00000123456"
        assert _strip_gene_version("ENSG00000123456") == "ENSG00000123456"
        # non-ENSG left untouched
        assert _strip_gene_version("GAPDH") == "GAPDH"

    def test_read_prebuilt_matrix_tsv(self, tmp_path):
        from cfstats.deconv import _read_prebuilt_matrix
        df = pd.DataFrame(
            {"Bcell": [1.0, 2.0], "Tcell": [3.0, 4.0]},
            index=["ENSG00000000001.2", "ENSG00000000002.3"],
        )
        p = tmp_path / "ref.tsv"
        df.to_csv(p, sep="\t")
        ref = _read_prebuilt_matrix(str(p))
        assert list(ref.columns) == ["Bcell", "Tcell"]
        # versions stripped
        assert "ENSG00000000001" in ref.index
        assert ref.index.name == "ENSG"


# ---------------------------------------------------------------------------
# deconvolution math
# ---------------------------------------------------------------------------

class TestDeconvolve:
    def _make_reference(self, n_genes=200, n_types=4, seed=0):
        rng = np.random.RandomState(seed)
        genes = [f"ENSG{ i:011d}" for i in range(n_genes)]
        types = [f"celltype_{c}" for c in range(n_types)]
        data = rng.rand(n_genes, n_types)
        return pd.DataFrame(data, index=genes, columns=types)

    def test_recovers_known_fractions_no_standardize(self):
        from cfstats.deconv import deconvolve
        ref = self._make_reference()
        true = np.array([0.5, 0.3, 0.2, 0.0])
        signal = pd.Series(ref.values @ true, index=ref.index, name="s1")
        frac, residual, boot_std = deconvolve(signal, ref, standardize=False)
        assert pytest.approx(frac.sum(), abs=1e-6) == 1.0
        np.testing.assert_allclose(frac.values, true, atol=1e-6)
        assert isinstance(residual, float) and residual >= 0.0
        assert boot_std is None

    def test_fractions_non_negative_and_sum_to_one(self):
        from cfstats.deconv import deconvolve
        ref = self._make_reference(seed=1)
        rng = np.random.RandomState(2)
        signal = pd.Series(rng.rand(ref.shape[0]), index=ref.index, name="s1")
        frac, residual, boot_std = deconvolve(signal, ref, standardize=True)
        assert (frac.values >= -1e-9).all()
        assert pytest.approx(frac.sum(), abs=1e-6) == 1.0
        assert list(frac.index) == list(ref.columns)
        assert boot_std is None

    def test_negative_relationship_recovers_fractions(self):
        """Regression: when FFT-WPS is *negatively* related to expression the
        old code returned all-zero weights. With orientation handling the
        fractions must be recovered and sum to 1."""
        from cfstats.deconv import deconvolve
        ref = self._make_reference()
        true = np.array([0.6, 0.4, 0.0, 0.0])
        # negatively related signal
        signal = pd.Series(-(ref.values @ true), index=ref.index, name="s1")
        frac, residual, boot_std = deconvolve(signal, ref, standardize=False, relationship="negative")
        assert pytest.approx(frac.sum(), abs=1e-6) == 1.0
        assert (frac.values >= -1e-9).all()
        np.testing.assert_allclose(frac.values, true, atol=1e-5)

    def test_auto_relationship_never_all_zero(self):
        """`auto` orientation must never collapse to all-zero (invalid) output."""
        from cfstats.deconv import deconvolve
        ref = self._make_reference(seed=3)
        true = np.array([0.7, 0.0, 0.3, 0.0])
        signal = pd.Series(-(ref.values @ true), index=ref.index, name="neg")
        frac, residual, boot_std = deconvolve(signal, ref, standardize=True, relationship="auto")
        assert pytest.approx(frac.sum(), abs=1e-6) == 1.0
        assert frac.values.max() > 0

    def test_too_few_shared_genes_raises(self):
        from cfstats.deconv import deconvolve
        ref = self._make_reference()
        signal = pd.Series([1.0, 2.0], index=["ENSGx", "ENSGy"], name="s1")
        with pytest.raises(RuntimeError):
            deconvolve(signal, ref)

    def test_residual_norm_is_finite(self):
        from cfstats.deconv import deconvolve
        ref = self._make_reference(seed=5)
        rng = np.random.RandomState(6)
        signal = pd.Series(rng.rand(ref.shape[0]), index=ref.index, name="s1")
        _, residual, _ = deconvolve(signal, ref, standardize=True)
        assert np.isfinite(residual)
        assert residual >= 0.0

    def test_bootstrap_std_shape_and_non_negative(self):
        from cfstats.deconv import deconvolve
        ref = self._make_reference(seed=7)
        true = np.array([0.5, 0.3, 0.2, 0.0])
        signal = pd.Series(-(ref.values @ true), index=ref.index, name="s1")
        frac, residual, boot_std = deconvolve(
            signal, ref, standardize=True, relationship="negative", n_bootstrap=50)
        assert boot_std is not None
        assert boot_std.shape == frac.shape
        assert (boot_std.values >= 0).all()
        assert list(boot_std.index) == list(frac.index)


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------

class TestCli:
    def test_deconv_missing_args_exits_2(self):
        import sys
        from cfstats.__main__ import main
        old = sys.argv
        try:
            sys.argv = ["cfstats", "deconv"]
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 2
        finally:
            sys.argv = old

    def test_lazy_cmd_resolves(self):
        from cfstats.__main__ import lazy_cmd
        cmd = lazy_cmd("deconv", "deconv")
        # importing/resolving should not raise
        mod = __import__("cfstats.deconv", fromlist=["deconv"])
        assert callable(getattr(mod, "deconv"))
