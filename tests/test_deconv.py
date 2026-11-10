"""Tests for cfstats deconv: FFT-WPS intensity helper, reference parsing,
and the NNLS deconvolution math."""

import io
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
        assert np.isnan(np.nanmean(fft_wps_intensity(np.ones(2000))))

    def test_empty_signal_returns_nan(self):
        from cfstats.ft import fft_wps_intensity
        assert np.isnan(np.nanmean(fft_wps_intensity(np.array([]))))

    def test_periodic_signal_returns_finite(self):
        from cfstats.ft import fft_wps_intensity
        # a signal with ~196 bp periodicity should yield a finite intensity in
        # the 193-199 bp band
        x = np.arange(4000)
        signal = np.sin(2 * np.pi * x / 196.0)
        val = np.nanmean(fft_wps_intensity(signal, ampmin=193, ampmax=199))
        assert np.isfinite(val)
        assert val > 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class TestReferenceLoading:
    def test_default_uses_packaged_broad_matrix_without_cache(self, monkeypatch):
        from types import SimpleNamespace
        from cfstats import deconv

        monkeypatch.setattr(deconv, '_cache_dir', lambda: pytest.fail('Default must not need a cache'))
        monkeypatch.setattr(deconv, '_download', lambda *args: pytest.fail('Default must not download'))
        ref = deconv.load_reference(SimpleNamespace())
        assert ref.shape == (19620, 171)
        assert deconv.DEFAULT_REFERENCE_ATLAS.endswith('cellxgene_tabsap_trophoblasts_broad_celltypes.log2.parquet')
        assert all(dtype == np.dtype('float32') for dtype in ref.dtypes)
        assert all(' ' not in name for name in ref.columns)
        assert np.isfinite(ref.to_numpy()).all()
        assert (ref.to_numpy() < 0).all()
        assert ref.columns[0] == 'VCT'

    def test_packaged_tissue_matrix_remains_available(self, monkeypatch):
        from types import SimpleNamespace
        from pathlib import Path
        from cfstats import deconv

        path = Path(deconv.DEFAULT_REFERENCE_ATLAS).with_name(
            'cellxgene_tabsap_trophoblasts_tissues.log2.parquet')
        monkeypatch.setattr(deconv, '_cache_dir', lambda: pytest.fail('Prebuilt matrix must not need a cache'))
        ref = deconv.load_reference(SimpleNamespace(reference_atlas=str(path)))
        assert ref.shape == (19620, 465)
        assert all(dtype == np.dtype('float32') for dtype in ref.dtypes)

    def test_explicit_matrix_overrides_default_and_url(self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        from cfstats import deconv

        path = tmp_path / 'custom.tsv'
        expected = pd.DataFrame({'custom': [1., 2.]}, index=['ENSG1', 'ENSG2'])
        expected.to_csv(path, sep='\t')
        monkeypatch.setattr(deconv, '_cache_dir', lambda: pytest.fail('Prebuilt matrix must not need a cache'))
        actual = deconv.load_reference(SimpleNamespace(reference_atlas=str(path), atlas_url='custom.h5ad'))
        pd.testing.assert_frame_equal(actual, expected.rename_axis('ENSG'))

    def test_explicit_url_still_downloads_and_aggregates(self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        from cfstats import deconv
        from unittest.mock import Mock

        monkeypatch.setattr(deconv, '_cache_dir', lambda: str(tmp_path))
        download = Mock()
        aggregate = Mock(return_value=pd.DataFrame({'custom': [1.]}))
        monkeypatch.setattr(deconv, '_download', download)
        monkeypatch.setattr(deconv, '_build_or_load_pseudobulk', aggregate)
        url = 'https://example.org/custom.h5ad'
        actual = deconv.load_reference(SimpleNamespace(atlas_url=url))
        download.assert_called_once_with(url, str(tmp_path / 'custom.h5ad'))
        aggregate.assert_called_once_with(str(tmp_path / 'custom.h5ad'), str(tmp_path), None, 10, 20000, False)
        assert actual is aggregate.return_value


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
# ranked correlation (Leuven/Kate method)
# ---------------------------------------------------------------------------

class TestRankCorrelate:
    def _make_reference(self, n_genes=200, seed=0):
        rng = np.random.RandomState(seed)
        genes = [f"ENSG{i:011d}" for i in range(n_genes)]
        data = rng.rand(n_genes, 3)
        return pd.DataFrame(
            data, index=genes, columns=["type_a", "type_b", "type_c"])

    def test_negative_correlator_gets_rank_one(self):
        """The column most negatively correlated with the signal (strongest
        contribution, since FFT-WPS decreases with expression) must be rank 1."""
        from cfstats.deconv import rank_correlate
        ref = self._make_reference()
        # signal = -(type_b) + noise-free: type_b should correlate at -1
        signal = pd.Series(-ref["type_b"].values, index=ref.index, name="s1")
        ranks, corr = rank_correlate(signal, ref)
        assert corr["type_b"] == pytest.approx(-1.0)
        assert ranks["type_b"] == 1

    def test_ranks_are_permutation_ascending_correlation(self):
        from cfstats.deconv import rank_correlate
        ref = self._make_reference()
        rng = np.random.RandomState(1)
        signal = pd.Series(rng.rand(ref.shape[0]), index=ref.index, name="s1")
        ranks, corr = rank_correlate(signal, ref)
        assert sorted(ranks.values) == [1, 2, 3]
        order = corr.sort_values().index
        for i, name in enumerate(order):
            assert ranks[name] == i + 1

    def test_nan_column_ranks_last(self):
        from cfstats.deconv import rank_correlate
        ref = self._make_reference()
        ref["type_c"] = np.nan
        rng = np.random.RandomState(2)
        signal = pd.Series(rng.rand(ref.shape[0]), index=ref.index, name="s1")
        ranks, corr = rank_correlate(signal, ref)
        assert np.isnan(corr["type_c"])
        assert ranks["type_c"] == 3

    def test_too_few_shared_genes_raises(self):
        from cfstats.deconv import rank_correlate
        ref = self._make_reference()
        signal = pd.Series([1.0, 2.0], index=["ENSGx", "ENSGy"], name="s1")
        with pytest.raises(RuntimeError):
            rank_correlate(signal, ref)


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------

class TestNnlsOutput:
    @pytest.mark.parametrize('long_output', [False, True])
    @pytest.mark.parametrize('bootstrap', [False, True])
    def test_contribution_order_per_sample(self, long_output, bootstrap, tmp_path, monkeypatch):
        from types import SimpleNamespace
        from cfstats import deconv

        columns = ['low', 'high', 'tie', 'missing']
        ref = pd.DataFrame(columns=columns)
        monkeypatch.setattr(deconv, 'load_reference', lambda args: ref)
        monkeypatch.setattr(deconv, '_open_gene_db', lambda path: [])
        monkeypatch.setattr(deconv, 'compute_sample_fftwps',
                            lambda samfile, args, db: samfile)
        def fit(sample, ref, **kwargs):
            fractions = [0.1, 0.45, 0.45] if sample == 's1.bam' else [0.7, 0.2, 0.1]
            std = pd.Series([0.11, 0.22, 0.33], index=columns[:3]) if bootstrap else None
            return pd.Series(fractions, index=columns[:3]), 0.5, std
        monkeypatch.setattr(deconv, 'deconvolve', fit)
        output = tmp_path / 'result.tsv'
        args = SimpleNamespace(samfiles=['s1.bam', 's2.bam'], output=str(output),
                               long=long_output, gfffile='unused', no_standardize=False,
                               n_bootstrap=int(bootstrap))
        deconv.deconv(args)
        paths = [output]
        if bootstrap:
            paths.append(tmp_path / 'result.bootstrap_std.tsv')
        for path in paths:
            table = pd.read_csv(path, sep='\t')
            if long_output:
                for sample, order in [('s1.bam', ['high', 'tie', 'low', 'missing']),
                                      ('s2.bam', columns)]:
                    rows = table[table['sample'] == sample]
                    assert rows['feature'].tolist() == ['fit_residual'] + order
                    assert pd.isna(rows.iloc[-1]['value'])
            else:
                assert table.columns.tolist() == ['sample', 'fit_residual'] + columns
            if path == output:
                if long_output:
                    contributions = table[table['feature'].isin(columns)].groupby('sample')['value']
                    assert all(values.dropna().is_monotonic_decreasing for _, values in contributions)
                else:
                    assert table.iloc[0]['low'] == pytest.approx(0.1)
                    assert table.iloc[1]['low'] == pytest.approx(0.7)


class TestCorrelationOutput:
    @pytest.mark.parametrize('mode', ['rank', 'corr', 'rankcorr', 'rank+corr'])
    @pytest.mark.parametrize('long_output', [False, True])
    @pytest.mark.parametrize('stdout', [False, True])
    def test_single_output_order_and_labels(self, mode, long_output, stdout,
                                           tmp_path, monkeypatch, capsys):
        from types import SimpleNamespace
        from cfstats import deconv

        genes = [f'ENSG{i:011d}' for i in range(20)]
        signal = pd.Series(np.arange(20), index=genes)
        ref = pd.DataFrame({'z_type': signal, 'a_type': -signal}, index=genes)
        monkeypatch.setattr(deconv, 'load_reference', lambda args: ref)
        monkeypatch.setattr(deconv, '_open_gene_db', lambda path: [])
        monkeypatch.setattr(deconv, 'compute_sample_fftwps',
                            lambda samfile, args, db: signal)
        output = tmp_path / 'result.tsv'
        args = SimpleNamespace(
            samfiles=['s1.bam', 's2.bam'], rank=mode in ('rank', 'rank+corr'),
            corr=mode in ('corr', 'rank+corr'), rankcorr=mode == 'rankcorr',
            leuven=False, output='-' if stdout else str(output),
            long=long_output, gfffile='unused', n_bootstrap=10)
        deconv.deconv(args)
        captured = capsys.readouterr()
        text = captured.out if stdout else output.read_text()
        assert captured.err == ''
        assert list(tmp_path.iterdir()) == ([] if stdout else [output])
        assert args.leuven is True
        kinds = ['rank', 'corr'] if mode in ('rankcorr', 'rank+corr') else [mode]
        if long_output:
            result = pd.read_csv(io.StringIO(text), sep='\t')
            assert result.columns.tolist() == ['sample', 'feature', 'value']
            for sample in args.samfiles:
                rows = result[result['sample'] == sample]
                assert rows['feature'].tolist() == [
                    f'{cell}.{kind}' for cell in ['a_type', 'z_type'] for kind in kinds]
                for kind in kinds:
                    assert rows.loc[rows['feature'].str.endswith(f'.{kind}'), 'value'].is_monotonic_increasing
        else:
            result = pd.read_csv(io.StringIO(text), sep='\t', index_col=0)
            assert result.index.tolist() == args.samfiles
            assert result.columns.tolist() == [
                f'{cell}.{kind}' for cell in ref.columns for kind in kinds]
        values = dict(zip(result['feature'], result['value'])) if long_output else result.iloc[0].to_dict()
        for kind in kinds:
            assert values[f'a_type.{kind}'] == pytest.approx(1 if kind == 'rank' else -1)
            assert values[f'z_type.{kind}'] == pytest.approx(2 if kind == 'rank' else 1)


class TestCli:
    @pytest.mark.parametrize('flag', ['--rank', '--corr', '--rankcorr'])
    def test_correlation_flags(self, flag, monkeypatch):
        from unittest.mock import patch
        from cfstats.__main__ import main, lazy_cmd
        monkeypatch.setattr('sys.argv', ['cfstats', 'deconv', 'x.bam', flag, '--long'])
        with patch.object(lazy_cmd, '__call__') as call:
            main()
        args = call.call_args.args[0]
        assert getattr(args, flag[2:]) is True
        assert args.long is True

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
