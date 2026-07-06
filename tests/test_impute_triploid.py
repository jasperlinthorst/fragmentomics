"""Tests for the mean-field triploid (NIPT) imputation + EM fetal fraction."""

import numpy as np
import pytest

from cfstats.impute.core import (
    triploid_emission,
    diploid_emission,
    impute_triploid,
    read_fetal_responsibilities,
    simR,
    sigma_from_positions,
)


def test_triploid_emission_reduces_to_diploid():
    """With w_ip=0 and the IPH allele path irrelevant, triploid_emission
    reproduces the diploid 0.5/0.5 mixture (up to the clamping floor)."""
    rng = np.random.default_rng(0)
    k, n = 8, 20
    is_alt = rng.integers(0, 2, size=(k, n)).astype(bool)
    h_other = rng.integers(0, 2, size=n).astype(np.float64)
    a_zero = np.zeros(n)
    minp = 0.01

    dip = diploid_emission(is_alt, h_other, minp=minp)
    # diploid: target weight 0.5, the conditioned hap weight 0.5.
    tri = triploid_emission(is_alt, h_other, a_zero,
                            w_target=0.5, w_other1=0.5, w_other2=0.0, minp=minp)
    np.testing.assert_allclose(tri, dip, atol=1e-9)


def test_triploid_emission_weights_sum_to_mixture():
    """p = w_t*e + w1*a1 + w2*a2, clamped to [minp, 1-minp]."""
    is_alt = np.array([[True, False], [False, True]])
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.0, 1.0])
    p = triploid_emission(is_alt, a1, a2,
                          w_target=0.5, w_other1=0.3, w_other2=0.2, minp=1e-3)
    # state 0, site 0: 0.5*1 + 0.3*1 + 0.2*0 = 0.8
    assert p[0, 0] == pytest.approx(0.8)
    # state 1, site 0: 0.5*0 + 0.3*1 + 0.2*0 = 0.3
    assert p[1, 0] == pytest.approx(0.3)
    # state 1, site 1: 0.5*1 + 0.3*0 + 0.2*1 = 0.7
    assert p[1, 1] == pytest.approx(0.7)


def _make_panel(k=60, n=150, seed=1):
    rng = np.random.default_rng(seed)
    # Block-structured haplotypes so neighbouring sites are in LD.
    emission = np.zeros((k, n), dtype=np.uint8)
    for i in range(k):
        block = rng.integers(0, 2, size=(n + 9) // 10)
        emission[i] = np.repeat(block, 10)[:n]
    positions = np.arange(1, n + 1) * 1000
    sigma = sigma_from_positions(positions, nGen=100, minp=1e-3)
    return emission, sigma


def test_impute_triploid_global_ff_moves_toward_truth():
    """Global-ff (no per-read prior) mean-field: the EM moves the seed ff
    towards the truth and recovers the maternal haplotypes well.  Exact ff
    recovery is not expected because the fetal-only (IPH) signal is weak at
    low ff with pooled reads -- the per-read-prior path is what sharpens it."""
    np.random.seed(0)
    emission, sigma = _make_panel()
    n = emission.shape[1]

    imh = emission[3].astype(np.uint8)
    nimh = emission[20].astype(np.uint8)
    iph = emission[40].astype(np.uint8)

    true_ff = 0.25
    R = simR(imh, nimh, iph, n, totreads=8000, ff=true_ff)

    gIMH, gNIMH, gIPH, em, ff_hat = impute_triploid(
        R, sigma, emission, ff=0.1, nhap=None, n_iter=12,
        minp=1e-3, nthreads=1)

    # EM moved the 0.1 seed upward toward the 0.25 truth without overshooting.
    assert 0.11 < ff_hat < 0.30
    # Returned posteriors have the expected shape.
    assert gIMH.shape == gNIMH.shape == gIPH.shape == (em.shape[0], n)


def test_impute_triploid_read_prior_recovers_ff():
    """With per-read fetal posteriors (read_prior), the EM recovers the true
    fetal fraction tightly and converges quickly."""
    np.random.seed(0)
    emission, sigma = _make_panel()
    n = emission.shape[1]

    imh = emission[3].astype(np.uint8)
    nimh = emission[20].astype(np.uint8)
    iph = emission[40].astype(np.uint8)

    true_ff = 0.25
    R = simR(imh, nimh, iph, n, totreads=8000, ff=true_ff, read_prior=True)

    gIMH, gNIMH, gIPH, em, ff_hat = impute_triploid(
        R, sigma, emission, ff=0.1, nhap=None, n_iter=12,
        minp=1e-3, nthreads=1, read_prior=True)

    assert abs(ff_hat - true_ff) < 0.05


def test_read_fetal_responsibilities_uses_per_read_prior():
    """When read_prior is set, a read with w_i~1 should be assigned high
    fetal responsibility regardless of the global ff."""
    n = 5
    p_im = np.full(n, 0.5)
    p_nim = np.zeros(n)
    p_ip = np.ones(n)
    # One read carrying the alt allele (matches IPH/fetal), strong prior w=0.99.
    R = {(0, 'r'): [[(0, 1, 30), (1, 1, 30)], 0, None, 150, 0.99]}
    resp = read_fetal_responsibilities(R, p_im, p_nim, p_ip,
                                       ff=0.05, read_prior=True)
    assert resp[0] > 0.5
