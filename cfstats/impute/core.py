"""Population-reference HMM imputation of maternal/fetal genotypes.

Architecture
------------
The imputation pipeline models each haplotype as a mosaic of K reference
haplotypes connected by a Li & Stephens hidden Markov model. Transition
probabilities (``sigma``) encode recombination rates; the emission matrix
(uint8, k x n) stores phased reference alleles (0=ref, 1=alt).

**Diploid mode** (``--diploid``)
    Uses *mean-field variational inference*.  All reads are pooled and an
    initial haploid forward-backward pass produces a first posterior.
    Then, iteratively:
    1. A haplotype path is sampled from the current posterior.
    2. Emissions are *adjusted* to account for the sampled haplotype
       (reads come 50/50 from either haplotype).
    3. A forward-backward pass with the adjusted (float64) emissions
       produces the other haplotype's posterior.
    Iterating a few times separates the two haplotype signals.  Without
    the conditioning step the model fails at heterozygous sites (11% vs
    81% concordance) because LD forces the posterior to commit to one
    allele.  See ``impute_diploid``.

**Triploid / NIPT mode** (default)
    Uses *block-Gibbs read-label assignment* with three labels
    (inherited-maternal, non-inherited-maternal, inherited-paternal).
    For each read, a local window of the HMM log-likelihood is evaluated
    for all three label assignments and the best is kept.  After
    convergence, three independent forward-backward passes produce
    per-haplotype posteriors.  See ``gibbs``.

**Custom reference training** (``cfstats impute build-reference / train``)
    When a phased reference panel is unavailable but a large collection
    of BAM/CRAM/VCF files is, the pipeline can learn HMM parameters
    from the data via EM.  See ``build_reference`` and ``train_model``.

C extension
-----------
The compiled ``_hmm`` module provides OpenMP-parallelised forward and
backward passes.  Two variants exist: uint8 emission (``forwardHaploid``,
``backwardHaploid``) for standard haploid passes, and float64 emission
(``forwardHaploidDouble``, ``backwardHaploidDouble``) for the diploid
adjusted-emission passes.  Both release the GIL.
"""

from __future__ import annotations

import gzip
import io
import logging
import multiprocessing
import os
import pickle
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pysam
from scipy.stats import binom

from cfstats.impute import _hmm as imputeff_hmm

np.random.seed(0)
random.seed(0)


# ===================================================================
# Observation helpers
# ===================================================================

def RtoX(R, n, label=None, usetruthlabels=False):
    """Convert a read dictionary *R* into an (n, 2) uint8 ref/alt count matrix.

    Parameters
    ----------
    R : dict
        Keyed by (ref_start, read_name).  Each value is a list:
        ``[observations, assigned_label, truth_label, template_length]``
        where *observations* is a list of ``(variant_index, allele, qual)``.
    n : int
        Number of variant sites (columns in the emission matrix).
    label : int or None
        If an int, only reads assigned to this label contribute.
        If None, all reads contribute.
    usetruthlabels : bool
        Use the truth label (index 2) instead of the assigned label.
    """
    X = np.zeros((n, 2), dtype=np.uint8)
    for r in R:
        obs = R[r][0]
        assigned = R[r][2] if usetruthlabels else R[r][1]
        if assigned == label or label is None:
            for idx, allele, qual in obs:
                X[idx][allele] += 1
    return X


# ===================================================================
# Forward-backward utilities
# ===================================================================

def forward_backward_haploid(x, sigma, emission, nthreads=1):
    """Haploid forward-backward with uint8 emission.

    Returns the posterior *gamma* matrix (k, n) where ``gamma[h, t]`` is
    the probability that haplotype *h* is the underlying state at site *t*.
    """
    k = emission.shape[0]
    ini = np.ones(k) / k
    alpha, ac = imputeff_hmm.forwardHaploid(
        x, ini, sigma, emission, scale=1, nthreads=nthreads)
    beta, bc = imputeff_hmm.backwardHaploid(
        x, sigma, emission, scale=ac, nthreads=nthreads)
    return np.exp(np.log(alpha) + np.log(beta) - np.log(ac))


def forward_backward_haploid_double(x, sigma, emission_probs, nthreads=1):
    """Haploid forward-backward with float64 emission.

    Identical to ``forward_backward_haploid`` except emissions are
    continuous probabilities (not binary {0,1}).  Used by the diploid
    mean-field variational inference where the emission of each
    reference haplotype is adjusted to condition on the other haplotype.
    """
    k = emission_probs.shape[0]
    ini = np.ones(k) / k
    em = np.ascontiguousarray(emission_probs, dtype=np.float64)
    alpha, ac = imputeff_hmm.forwardHaploidDouble(
        x, ini, sigma, em, scale=1, nthreads=nthreads)
    beta, bc = imputeff_hmm.backwardHaploidDouble(
        x, sigma, em, scale=ac, nthreads=nthreads)
    return np.exp(np.log(alpha) + np.log(beta) - np.log(ac))


# ===================================================================
# Reference loading
# ===================================================================

def loadref_vcf(vcf, contig=None, start=None, stop=None,
                nGen=100, avgr=1, minp=0.001):
    """Load a phased reference VCF and return ``(sigma, emission, variants)``.

    Tries ``bcftools`` for fast bulk GT extraction first; falls back to
    ``pysam`` if bcftools is unavailable.

    Returns
    -------
    sigma : ndarray (n-1,)
        Probability of *no* recombination between consecutive sites.
    emission : ndarray (k, n) uint8
        Reference haplotype alleles (0=ref, 1=alt).
    variants : list of (id, pos, ref, alt)
        One entry per site.
    """
    try:
        return _loadref_vcf_bcftools(vcf, contig, start, stop, nGen, avgr, minp)
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        print("bcftools not available, falling back to pysam (slower).")
        return _loadref_vcf_pysam(vcf, contig, start, stop, nGen, avgr, minp)


def _loadref_vcf_bcftools(vcf, contig, start, stop, nGen, avgr, minp):
    """Fast VCF loading using bcftools + numpy bulk digit extraction."""
    region_args = []
    if contig:
        region = contig
        if start is not None and stop is not None:
            region += ':%d-%d' % (start, stop)
        elif start is not None:
            region += ':%d' % start
        region_args = ['-r', region]

    # Query 1: variant metadata
    cmd_info = ['bcftools', 'query', '-f',
                '%ID\t%POS\t%REF\t%ALT\n'] + region_args + [vcf]
    info_raw = subprocess.check_output(cmd_info, text=True).rstrip('\n')
    if not info_raw:
        raise ValueError("No variants found in region")

    variants = []
    sigma = []
    pp = 0
    for line in info_raw.split('\n'):
        vid, pos_s, ref, alt = line.split('\t')
        pos = int(pos_s)
        variants.append((vid, pos, ref, alt))
        rr = 1 - (nGen * (pos - pp) / 1e6 / 100)
        if pp != 0:
            sigma.append(min(rr, 1 - minp))
        pp = pos

    n_variants = len(variants)
    sigma = np.array(sigma, dtype=np.float64)

    # Query 2: GT data only -- extract digits with numpy for speed
    cmd_gt = ['bcftools', 'query', '-f',
              '[\t%GT]\n'] + region_args + [vcf]
    gt_raw = subprocess.check_output(cmd_gt)

    raw_arr = np.frombuffer(gt_raw, dtype=np.uint8)
    digits = raw_arr[(raw_arr >= 48) & (raw_arr <= 57)] - 48
    n_haplotypes = len(digits) // n_variants
    emission = digits.reshape(n_variants, n_haplotypes).T.copy()

    return sigma, emission, variants


def _loadref_vcf_pysam(vcf, contig, start, stop, nGen, avgr, minp):
    """Fallback VCF loading using pysam (slow for large panels)."""
    vcffile = pysam.VariantFile(vcf)
    emission = []
    variants = []
    sigma = []

    pp = 0
    for rec in vcffile.fetch(contig=contig, start=start, stop=stop):
        variants.append((rec.id, rec.pos, rec.ref, rec.alts[0]))
        col = [np.uint8(g) for gt in rec.samples.values() for g in gt['GT']]
        rr = 1 - (nGen * (rec.pos - pp) / 1e6 * (1 / 100))
        if pp != 0:
            sigma.append(min(rr, 1 - minp))
        emission.append(col)
        pp = rec.pos

    sigma = np.array(sigma, dtype=np.float64)
    emission = np.array(emission).transpose().copy()
    vcffile.close()
    return sigma, emission, variants


def loadref_haplegend(hapfile, legendfile,
                      nGen=100, avgr=1, minp=0.001,
                      start=None, stop=None):
    """Load a phased haplotype reference from hap.gz / legend.gz (IMPUTE format)."""
    variants = []
    sigma = []
    keep_idx = []

    with gzip.open(legendfile, 'rt') as legend:
        legend.readline()
        pp = 0
        idx = 0
        for line in legend:
            vid, position, a0, a1 = line.rstrip().split(" ")
            position = int(position)
            if start is not None and position < start:
                idx += 1
                continue
            if stop is not None and position > stop:
                idx += 1
                continue
            keep_idx.append(idx)
            variants.append((vid, position, a0, a1))
            rr = 1 - (nGen * (position - pp) / 1e6 * (1 / 100))
            if pp != 0:
                sigma.append(min(rr, 1 - minp))
            pp = position
            idx += 1

    with gzip.open(hapfile, 'rb') as hap:
        raw = hap.read()
    emission = np.loadtxt(io.BytesIO(raw), dtype=np.uint8)
    if keep_idx and len(keep_idx) < emission.shape[0]:
        emission = emission[keep_idx, :]
    emission = emission.T.copy()

    sigma = np.array(sigma, dtype=np.float64)
    return sigma, emission, variants


# ===================================================================
# Read loading
# ===================================================================

def getR(file, chrom, variants, ref=None,
         addchr=False, rmchr=False, ff=0.1, stepper='all'):
    """Pile up reads from *file* at the reference *variants*.

    Returns a dict keyed by ``(reference_start, read_name)`` with value
    ``[observations, assigned_label, truth_label_or_None, template_length]``.
    Each observation is ``(variant_index, allele, base_quality)``.
    Initial label assignment is random (50% label 0, ~(50-ff/2)% label 1,
    ~ff/2% label 2) to seed the Gibbs sampler.
    """
    R = {}
    bamfile = pysam.AlignmentFile(file, reference_filename=ref)
    i = 0

    if addchr:
        chrom = 'chr' + chrom
    if rmchr:
        chrom = chrom.replace('chr', '')

    for vid, pos, refa, alt in variants:
        for pc in bamfile.pileup(chrom, int(pos) - 1, int(pos),
                                 truncate=True, multiple_iterators=False,
                                 stepper=stepper):
            for pcr in pc.pileups:
                if pcr.query_position is None:
                    continue
                pb = pcr.alignment.query_sequence[pcr.query_position]
                pq = pcr.alignment.query_qualities[pcr.query_position]
                if pb == refa:
                    a = 0
                elif pb == alt:
                    a = 1
                else:
                    continue

                k = (pcr.alignment.reference_start,
                     pcr.alignment.query_name)
                if k in R:
                    R[k][0].append((i, a, pq))
                else:
                    s = random.random()
                    R[k] = [
                        [(i, a, pq)],
                        0 if s < 0.5 else 1 if s < 1 - (ff / 2) else 2,
                        None,
                        pcr.alignment.template_length,
                    ]
        i += 1
        if i % 1000 == 0:
            print("Processing variants (%d/%d)..." % (i, len(variants)))

    bamfile.close()
    return R


# ===================================================================
# Haplotype pre-selection
# ===================================================================

def preselect_haplotypes(R, emission, nhap, n):
    """Select the *nhap* best-matching reference haplotypes by allele overlap.

    For each haplotype, sums the log-likelihood contribution at read
    positions.  This is O(k * total_obs) and avoids allocating the full
    k x n forward/backward matrices.
    """
    k = emission.shape[0]
    minp = 0.001

    obs = []
    for r in R:
        for idx, allele, qual in R[r][0]:
            obs.append((idx, allele))

    if not obs:
        return list(range(min(nhap, k)))

    obs_arr = np.array(obs, dtype=np.int64)
    indices = obs_arr[:, 0]
    alleles = obs_arr[:, 1]

    hap_at_obs = emission[:, indices].astype(np.float64)
    hap_at_obs = np.clip(hap_at_obs, minp, 1 - minp)

    alt_mask = (alleles == 1).astype(np.float64)
    ref_mask = 1.0 - alt_mask

    scores = (np.log(hap_at_obs) * alt_mask[np.newaxis, :] +
              np.log(1.0 - hap_at_obs) * ref_mask[np.newaxis, :]).sum(axis=1)

    best = np.argsort(scores)[-nhap:]
    return list(best)


# ===================================================================
# Diploid imputation  (mean-field variational)
# ===================================================================

def diploid_emission(emission_is_alt, h_alleles, minp=0.001):
    """Build float64 adjusted emissions conditioned on a sampled haplotype.

    At each site, reads come from h1 or h2 with 50% probability.  If we
    know h1's allele, the expected emission for a *reference* haplotype
    that carries the alt allele at that site depends on whether h1 was
    also alt (then we expect ~100% alt reads) or ref (then ~50%).

    Parameters
    ----------
    emission_is_alt : ndarray (k, n) bool
        ``emission > 0.5`` for the (sub-selected) reference panel.
    h_alleles : ndarray (n,) float
        Sampled allele per site for the conditioning haplotype (0 or 1).
    minp : float
        Emission floor (prevents underflow).

    Returns
    -------
    ndarray (k, n) float64
        Adjusted emission probabilities for the conditioned HMM pass.
    """
    h_is_alt = (h_alleles > 0.5)[np.newaxis, :]   # (1, n)
    return np.where(
        h_is_alt,
        np.where(emission_is_alt, 1 - minp, 0.5),
        np.where(emission_is_alt, 0.5, minp)
    )


def sample_haplotype_path(gamma, emission):
    """Sample a haplotype path from the posterior *gamma*.

    At each site, draws a reference haplotype from the categorical
    distribution ``gamma[:, t]`` and returns that haplotype's allele.

    Parameters
    ----------
    gamma : ndarray (k, n)
        Posterior from a forward-backward pass.
    emission : ndarray (k, n) uint8
        Reference panel alleles.

    Returns
    -------
    ndarray (n,)
        Sampled allele (0 or 1) per site.
    """
    k, n = gamma.shape
    g = gamma / gamma.sum(axis=0, keepdims=True)
    cumprob = np.cumsum(g, axis=0)
    u = np.random.rand(n)
    idx = (cumprob < u[np.newaxis, :]).sum(axis=0)
    idx = np.clip(idx, 0, k - 1)
    return emission[idx, np.arange(n)]


def impute_diploid(R, sigma, emission, nhap=None, n_iter=3,
                   minp=0.001, nthreads=1):
    """Diploid imputation via mean-field variational inference.

    1. Pre-select *nhap* best-matching haplotypes (fast overlap score).
    2. Pool all reads and run an initial haploid forward-backward.
    3. Iterate *n_iter* times: sample one haplotype, adjust emissions
       for the other, run a float64 forward-backward, and repeat in the
       other direction.
    4. Average the post-burn-in marginals.

    Parameters
    ----------
    R : dict
        Read dictionary from ``getR``.
    sigma : ndarray (n-1,)
        No-recombination probabilities.
    emission : ndarray (k, n) uint8
        Full reference panel.
    nhap : int or None
        Number of haplotypes to pre-select (None = use all).
    n_iter : int
        Gibbs-style sampling iterations (0 = naive single-pass).
    minp : float
        Emission floor.
    nthreads : int
        OpenMP thread count for the C forward/backward.

    Returns
    -------
    p1 : ndarray (n,)
        Marginal P(alt) for haplotype 1.
    p2 : ndarray (n,)
        Marginal P(alt) for haplotype 2.
    emission : ndarray (k_sel, n) uint8
        The (possibly sub-selected) emission matrix used.
    """
    # --- haplotype pre-selection ---
    if nhap is not None and isinstance(nhap, int):
        hap_indices = preselect_haplotypes(R, emission, nhap, emission.shape[1])
        emission = emission[hap_indices, :]
        print("Using %d pre-selected haplotypes for diploid pass." % len(hap_indices))

    k = emission.shape[0]
    n = emission.shape[1]
    is_alt = (emission > 0.5)

    x_all = RtoX(R, n, label=None)
    print("Pooled observation matrix: %d sites with data" %
          (x_all.sum(axis=1) > 0).sum())

    # --- initial haploid pass (uint8 emission, pooled reads) ---
    gamma1 = forward_backward_haploid(x_all, sigma, emission, nthreads=nthreads)
    print("Diploid: initial haploid pass done")

    if n_iter == 0:
        # Naive mode: single pass, same marginal for both haplotypes.
        p = (gamma1 * emission).sum(axis=0) / gamma1.sum(axis=0)
        print("Naive mode: single haploid pass (n_iter=0), dosage = 2*p")
        return p, p, emission

    # --- iterative mean-field conditioning ---
    burnin = max(0, n_iter // 3)
    p1_sum = np.zeros(n, dtype=np.float64)
    p2_sum = np.zeros(n, dtype=np.float64)
    n_avg = 0

    for gi in range(n_iter):
        # Sample h1 from gamma1, condition on it, infer h2
        h1_alleles = sample_haplotype_path(gamma1, emission)
        eff2 = diploid_emission(is_alt, h1_alleles, minp)
        gamma2 = forward_backward_haploid_double(
            x_all, sigma, eff2, nthreads=nthreads)

        # Sample h2 from gamma2, condition on it, infer h1
        h2_alleles = sample_haplotype_path(gamma2, emission)
        eff1 = diploid_emission(is_alt, h2_alleles, minp)
        gamma1 = forward_backward_haploid_double(
            x_all, sigma, eff1, nthreads=nthreads)

        p1_cur = (gamma1 * emission).sum(axis=0) / gamma1.sum(axis=0)
        p2_cur = (gamma2 * emission).sum(axis=0) / gamma2.sum(axis=0)

        if gi >= burnin:
            p1_sum += p1_cur
            p2_sum += p2_cur
            n_avg += 1

        print("Diploid iter %d: p1_mean=%.4f p2_mean=%.4f%s" %
              (gi + 1, p1_cur.mean(), p2_cur.mean(),
               " (averaging)" if gi >= burnin else " (burn-in)"))

    p1 = p1_sum / max(n_avg, 1)
    p2 = p2_sum / max(n_avg, 1)
    print("Averaged %d post-burn-in samples" % n_avg)
    return p1, p2, emission


# ===================================================================
# Triploid (NIPT) block-Gibbs read labelling
# ===================================================================

def compute_block_window(centralsnpi, n_full, windowsize=32):
    """Compute a symmetric window [start, stop) around *centralsnpi*.

    Clamps to [0, n_full).  Used by the Gibbs read-label reassignment
    to evaluate a local log-likelihood around each read's central SNP.
    """
    if n_full <= windowsize:
        return 0, n_full
    start = centralsnpi - windowsize // 2
    stop = centralsnpi + windowsize // 2
    if start < 0:
        stop -= start
        start = 0
    if stop > n_full:
        start -= (stop - n_full)
        stop = n_full
    return max(start, 0), stop


def loglikelihood_block(R, centralsnpi, sigma, emission,
                        useprior=True, ff=0.1,
                        windowsize=32, nthreads=1):
    """Block log-likelihood for the triploid model at *centralsnpi*.

    Evaluates the HMM forward pass in a local window for each of the
    three labels and combines with a binomial prior on label counts.
    """
    n_full = emission.shape[1]
    start, stop = compute_block_window(centralsnpi, n_full, windowsize)

    em_block = emission[:, start:stop]
    sig_block = np.ones(stop - start - 1, dtype=np.double)
    ini = np.ones(em_block.shape[0]) / em_block.shape[0]

    n_per_label = [sum(1 for r in R if R[r][1] == l) for l in range(3)]
    n_total = len(R)

    ll = 0.0
    priors = [0.5, 0.5 - (ff / 2), ff / 2]
    for label in range(3):
        alpha, c = imputeff_hmm.forwardHaploid(
            RtoX(R, n_full, label)[start:stop],
            ini, sig_block, em_block, nthreads=nthreads)
        ll += -np.log(c).sum()
        if useprior:
            ll += np.log(binom.pmf(n_per_label[label], n_total,
                                   priors[label]) + 1e-308)
    return ll


def loglikelihood(R, sigma, emission, useprior=True, ff=0.1,
                  nthreads=1):
    """Full-length triploid log-likelihood (sum over all three labels)."""
    k = emission.shape[0]
    ini = np.ones(k) / k

    n_per_label = [sum(1 for r in R if R[r][1] == l) for l in range(3)]
    n_total = len(R)
    priors = [0.5, 0.5 - (ff / 2), ff / 2]

    def _fwd(label):
        return imputeff_hmm.forwardHaploid(
            RtoX(R, emission.shape[1], label),
            ini, sigma, emission, nthreads=nthreads)

    if nthreads > 1:
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(_fwd, (0, 1, 2)))
    else:
        results = [_fwd(l) for l in range(3)]

    ll = 0.0
    for label, (alpha, c) in enumerate(results):
        ll += -np.log(c).sum()
        if useprior:
            ll += np.log(binom.pmf(n_per_label[label], n_total,
                                   priors[label]) + 1e-308)
    return ll


def gibbs(R, sigma, emission, nhap=400, ff=0.1, useprior=True,
          verbose=False, maxiterfull=3, maxiterpart=None,
          nthreads=1):
    """Block-Gibbs read-label assignment for the triploid NIPT model.

    Phase 1 -- haplotype pre-selection:
        If *nhap* is an integer and the panel is too large for full HMM
        passes (>1 GB), fall back to ``preselect_haplotypes``.  Otherwise
        run up to *maxiterfull* full forward-backward passes to select the
        best-matching haplotypes.

    Phase 2 -- label reassignment:
        For each read, evaluate the local HMM log-likelihood under each
        of the three possible label assignments and keep the best.
        Iterate until convergence or *maxiterpart* iterations.

    Parameters
    ----------
    R : dict
        Read dictionary (modified in-place: labels are updated).
    sigma, emission : ndarray
        Reference panel.
    nhap : int or None
        Number of haplotypes to pre-select.
    ff : float
        Expected fetal fraction prior.
    maxiterfull : int
        Max full-panel passes for haplotype selection.
    maxiterpart : int or None
        Max label-reassignment iterations (None = until convergence).
    """
    from scipy.special import logsumexp

    subset = set()

    if isinstance(nhap, int):
        bestn = nhap

    outer_iters = maxiterfull if isinstance(nhap, int) else 1

    # --- Phase 1: haplotype pre-selection ---
    if isinstance(nhap, int):
        k = emission.shape[0]
        n = emission.shape[1]

        mem_estimate_gb = k * n * 8 * 2 / 1e9
        if mem_estimate_gb > 1.0:
            print("Panel too large for full HMM pre-selection "
                  "(%.1f GB per pass); using fast allele-overlap selection."
                  % mem_estimate_gb)
            nhap = preselect_haplotypes(R, emission, bestn, n)
        else:
            for itermain in range(outer_iters):
                print("Running full pass: %d" % itermain)
                ini = np.ones(k) / k

                results = []
                for label in range(3):
                    x = RtoX(R, emission.shape[1], label)
                    alpha, ac = imputeff_hmm.forwardHaploid(
                        x, ini, sigma, emission, nthreads=nthreads)
                    beta, bc = imputeff_hmm.backwardHaploid(
                        x, sigma, emission, scale=ac, nthreads=nthreads)
                    gammalog = (np.log(alpha) - np.log(ac)) + \
                               (np.log(beta) - np.log(ac))
                    results.append(gammalog)

                gsum = results[0]
                for g in results[1:]:
                    gsum = gsum + g

                subset_ = set(np.argsort(logsumexp(gsum, axis=1))[-bestn:])

                if subset == subset_:
                    print("No change in selecting best k haplotypes, break.")
                    break
                else:
                    subset = subset_

            nhap = list(subset) if subset else preselect_haplotypes(
                R, emission, bestn, n)

    # --- Phase 2: Gibbs label reassignment ---
    ll = loglikelihood(R, sigma, emission, useprior=useprior, ff=ff,
                       nthreads=nthreads)
    print("init", ll)

    ll_ = None
    it = 0

    while ll != ll_ and (maxiterpart is None or it < maxiterpart):
        it += 1
        print("iter: %d (%.10f)" % (it, ll))
        ll_ = ll

        for r in sorted(R.keys(), key=lambda l: l[0]):
            org = R[r][1]
            rO = R[r][0]
            centralsnpi = sorted(rO, key=lambda l: l[0])[int(len(rO) / 2)][0]

            ll = loglikelihood_block(R, centralsnpi, sigma, emission,
                                     useprior=useprior, ff=ff,
                                     windowsize=32, nthreads=nthreads)

            opta = org + 1 if org < 2 else 0
            optb = org - 1 if org > 0 else 2

            R[r][1] = opta
            tmp1 = loglikelihood_block(R, centralsnpi, sigma, emission,
                                       useprior=useprior, ff=ff,
                                       windowsize=32, nthreads=nthreads)

            R[r][1] = optb
            tmp2 = loglikelihood_block(R, centralsnpi, sigma, emission,
                                       useprior=useprior, ff=ff,
                                       windowsize=32, nthreads=nthreads)

            if tmp1 > tmp2:
                tmp = tmp1
                R[r][1] = opta
            else:
                tmp = tmp2
                R[r][1] = optb

            if tmp > ll:
                if verbose:
                    print("keep %s (%d->%d): %.4f" % (r, org, R[r][1], tmp))
                ll = tmp
            else:
                R[r][1] = org

        ll = loglikelihood(R, sigma, emission, useprior=useprior, ff=ff,
                           nthreads=nthreads)


# ===================================================================
# VCF output
# ===================================================================

class _BGZFWriter:
    """Minimal text wrapper around pysam.BGZFile."""

    def __init__(self, path):
        self._fh = pysam.BGZFile(path, "wb")

    def write(self, text):
        self._fh.write(text.encode("utf-8"))

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _open_vcf_writer(vcffile):
    """Return a writable file-like and whether to close it at the end."""
    if vcffile.endswith('.gz'):
        return _BGZFWriter(vcffile), True
    elif vcffile == '-':
        return sys.stdout, False
    else:
        return open(vcffile, 'wt'), True


def outputvcf_diploid_marginals(p1, p2, R, emission,
                                chrom, variants, vcffile, sample):
    """Write diploid imputation results to VCF.

    Genotype probabilities from two marginal P(alt) vectors:
        GP(0/0) = (1-p1)(1-p2)
        GP(0/1) = p1(1-p2) + (1-p1)p2
        GP(1/1) = p1*p2
    """
    k = emission.shape[0]
    n = emission.shape[1]

    X_all = RtoX(R, n, label=None)
    refaltcnt = emission.sum(axis=0, dtype=np.int64)

    writer, close_at_end = _open_vcf_writer(vcffile)
    try:
        vcf = writer
        vcf.write("##fileformat=VCFv4.3\n")
        vcf.write("##total_reads=%d\n" % len(R))
        vcf.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        vcf.write('##FORMAT=<ID=GP,Number=3,Type=Float,Description="Posterior diploid genotype probability of 0/0, 0/1, and 1/1">\n')
        vcf.write('##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">\n')
        vcf.write('##FORMAT=<ID=RC,Number=1,Type=Integer,Description="Reference allele count in input.">\n')
        vcf.write('##FORMAT=<ID=AC,Number=1,Type=Integer,Description="Alternative allele count in input.">\n')
        vcf.write('##INFO=<ID=RAAF,Number=1,Type=Float,Description="Alternative allele frequency in the reference set">\n')
        vcf.write('##INFO=<ID=INFO_SCORE,Number=1,Type=Float,Description="Info score for the imputed genotype.">\n')
        vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t%s\n" % sample)

        for pi, (vid, pos, ref, alt) in enumerate(variants):
            pa, pb = p1[pi], p2[pi]
            gp0 = (1 - pa) * (1 - pb)
            gp1 = pa * (1 - pb) + (1 - pa) * pb
            gp2 = pa * pb
            dosage = gp1 + 2 * gp2
            gt_str = ["0/0", "0/1", "1/1"][np.argmax([gp0, gp1, gp2])]

            thetaHat = dosage / 2
            denom = 2 * thetaHat * (1 - thetaHat)
            info_score = 1 - ((dosage - dosage ** 2) / denom) if denom > 0 else 1.0

            vcf.write(
                "{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\tPASS\t"
                "RAAF={raaf:.4f};INFO_SCORE={info:.2f}\t"
                "GT:GP:DS:RC:AC\t"
                "{gt}:{gp0:.3f},{gp1:.3f},{gp2:.3f}:{ds:.3f}:"
                "{rc}:{ac}\n".format(
                    chrom=chrom, pos=pos, ref=ref, alt=alt, vid=vid,
                    gt=gt_str, gp0=gp0, gp1=gp1, gp2=gp2, ds=dosage,
                    rc=X_all[pi, 0], ac=X_all[pi, 1],
                    raaf=refaltcnt[pi] / float(k), info=info_score,
                )
            )
    finally:
        if close_at_end:
            writer.close()


def outputvcf(gammaIMH, gammaNIMH, gammaIPH, R, emission,
              chrom, variants, vcffile, sample, minp=0.001):
    """Write triploid (NIPT) imputation results to VCF.

    Produces two sample columns: maternal genotype and fetal genotype,
    each with phased GT, diploid GP, and dosage.
    """
    k = gammaIMH.shape[0]
    n = gammaIMH.shape[1]
    assert gammaIMH.shape == gammaNIMH.shape == gammaIPH.shape

    readcnt = np.zeros(3)
    iszd = [[], [], []]
    for r in R:
        readcnt[R[r][1]] += 1
        iszd[R[r][1]].append(abs(R[r][3]))

    order = np.argsort(readcnt)
    IPHreadcnt, NIMHreadcnt, IMHreadcnt = readcnt[order]

    print("Reads assigned to inherited maternal allele:", IMHreadcnt)
    print("Reads assigned to non-inherited maternal allele:", NIMHreadcnt)
    print("Reads assigned to inherited paternal allele:", IPHreadcnt)

    ff = (IPHreadcnt / float(NIMHreadcnt + IPHreadcnt)) if (NIMHreadcnt + IPHreadcnt) else 0.0
    print("MLE fetal fraction: %.4f" % ff)

    Ximh = RtoX(R, n, order[2])
    Xnimh = RtoX(R, n, order[1])
    Xiph = RtoX(R, n, order[0])

    for lbl, idx in (("IMH", order[2]), ("NIMH", order[1]), ("IPH", order[0])):
        arr = np.array(iszd[idx])
        if len(arr):
            print("Insertsize %s: mean=%.1f std=%.1f n=%d" % (lbl, arr.mean(), arr.std(), len(arr)))

    refaltcnt = emission.sum(axis=0, dtype=np.int64)

    writer, close_at_end = _open_vcf_writer(vcffile)
    try:
        vcf = writer
        vcf.write("##fileformat=VCFv4.3\n")
        vcf.write("##ff=%.4f\n" % ff)
        vcf.write("##total_reads=%d\n" % len(R))
        vcf.write("##inherited_maternal_reads=%d\n" % IMHreadcnt)
        vcf.write("##noninherited_maternal_reads=%d\n" % NIMHreadcnt)
        vcf.write("##inherited_paternal_reads=%d\n" % IPHreadcnt)
        vcf.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype">\n')
        vcf.write('##FORMAT=<ID=GP,Number=3,Type=Float,Description="Posterior diploid genotype probability of 0/0, 0/1, and 1/1">\n')
        vcf.write('##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">\n')
        vcf.write('##FORMAT=<ID=RAC1,Number=1,Type=Integer,Description="Reference allele count in input assigned to first allele.">\n')
        vcf.write('##FORMAT=<ID=AAC1,Number=1,Type=Integer,Description="Alternative allele count in input assigned to first allele.">\n')
        vcf.write('##FORMAT=<ID=RAC2,Number=1,Type=Integer,Description="Reference allele count in input assigned to second allele.">\n')
        vcf.write('##FORMAT=<ID=AAC2,Number=1,Type=Integer,Description="Alternative allele count in input assigned to second allele.">\n')
        vcf.write('##INFO=<ID=RAAF,Number=1,Type=Float,Description="Alternative allele frequency in the reference set">\n')
        vcf.write('##INFO=<ID=INFO_SCORE_FETAL,Number=1,Type=Float,Description="Info score for the imputed fetal genotype.">\n')
        vcf.write('##INFO=<ID=INFO_SCORE_MATERNAL,Number=1,Type=Float,Description="Info score for the imputed maternal genotype.">\n')
        vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t%s\t%s_fetal\n" % (sample, sample))

        for pi, (vid, pos, ref, alt) in enumerate(variants):
            pref = 1 - abs(emission[:, pi] - minp)
            palt = abs(emission[:, pi] - minp)

            gtIMH_0 = (gammaIMH[:, pi] * pref).sum()
            gtIMH_1 = (gammaIMH[:, pi] * palt).sum()
            gtNIMH_0 = (gammaNIMH[:, pi] * pref).sum()
            gtNIMH_1 = (gammaNIMH[:, pi] * palt).sum()
            gtIPH_0 = (gammaIPH[:, pi] * pref).sum()
            gtIPH_1 = (gammaIPH[:, pi] * palt).sum()

            gtIMH = gtIMH_1 / (gtIMH_0 + gtIMH_1)
            gtIPH = gtIPH_1 / (gtIPH_0 + gtIPH_1)
            gtNIMH = gtNIMH_1 / (gtNIMH_0 + gtNIMH_1)

            totpm = ((gtIMH_0 * gtNIMH_0) + (gtIMH_1 * gtNIMH_1) +
                     (gtIMH_1 * gtNIMH_0) + (gtIMH_0 * gtNIMH_1))
            gtm0 = (gtIMH_0 * gtNIMH_0) / totpm
            gtm1 = ((gtIMH_1 * gtNIMH_0) + (gtIMH_0 * gtNIMH_1)) / totpm
            gtm2 = (gtIMH_1 * gtNIMH_1) / totpm

            totpf = ((gtIMH_1 * gtIPH_1) + (gtIMH_0 * gtIPH_0) +
                     (gtIMH_0 * gtIPH_1) + (gtIMH_1 * gtIPH_0))
            gtf0 = (gtIMH_0 * gtIPH_0) / totpf
            gtf1 = ((gtIMH_1 * gtIPH_0) + (gtIMH_0 * gtIPH_1)) / totpf
            gtf2 = (gtIMH_1 * gtIPH_1) / totpf

            maternal_dosage = gtm1 + (2 * gtm2)
            fetal_dosage = gtf1 + (2 * gtf2)

            eij_m = maternal_dosage
            thetaHat_m = eij_m / 2
            eij_f = fetal_dosage
            thetaHat_f = eij_f / 2

            denom_m = 2 * thetaHat_m * (1 - thetaHat_m)
            denom_f = 2 * thetaHat_f * (1 - thetaHat_f)
            info_m = 1 - ((eij_m - eij_m ** 2) / denom_m) if denom_m > 0 else 1.0
            info_f = 1 - ((eij_f - eij_f ** 2) / denom_f) if denom_f > 0 else 1.0

            vcf.write(
                "{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\tPASS\t"
                "RAAF={raaf:.4f};INFO_SCORE_MATERNAL={infom:.2f};INFO_SCORE_FETAL={infof:.2f}\t"
                "GT:GP:DS:RAC1:AAC1:RAC2:AAC2\t"
                "{gtIMH:d}|{gtNIMH:d}:{gtm0:.3f},{gtm1:.3f},{gtm2:.3f}:{mds:.3f}:"
                "{imhrc}:{imhac}:{nimhrc}:{nimhac}\t"
                "{gtIMH:d}|{gtIPH:d}:{gtf0:.3f},{gtf1:.3f},{gtf2:.3f}:{fds:.3f}:"
                "{imhrc}:{imhac}:{iphrc}:{iphac}\n".format(
                    chrom=chrom, pos=pos, ref=ref, alt=alt, vid=vid,
                    gtIMH=int(round(gtIMH)), gtNIMH=int(round(gtNIMH)),
                    gtIPH=int(round(gtIPH)),
                    gtm0=gtm0, gtm1=gtm1, gtm2=gtm2,
                    gtf0=gtf0, gtf1=gtf1, gtf2=gtf2,
                    mds=maternal_dosage, fds=fetal_dosage,
                    imhrc=Ximh[pi, 0], imhac=Ximh[pi, 1],
                    nimhrc=Xnimh[pi, 0], nimhac=Xnimh[pi, 1],
                    iphrc=Xiph[pi, 0], iphac=Xiph[pi, 1],
                    raaf=refaltcnt[pi] / float(k),
                    infom=info_m, infof=info_f,
                )
            )
    finally:
        if close_at_end:
            writer.close()


# ===================================================================
# Reference building & training
# (for when no phased reference panel exists but BAM/VCF files do)
# ===================================================================

def load_genotypes_from_file(ifile_idx, filepath, n, positions,
                             filterflag=3840, cramref=None):
    """Load genotypes from a single BAM/CRAM/VCF file into an allele-count
    tensor.  Called in parallel by ``build_reference``.

    Returns an (N, n, 2) uint8 array where N is the number of samples
    (1 for BAM/CRAM, number of VCF samples for VCF).
    """
    if filepath.endswith(('.bam', '.cram', '.sam')):
        bamfile = pysam.AlignmentFile(filepath, reference_filename=cramref)
        Xi = np.zeros((1, n, 2), dtype=np.uint8)
        ngen = 0
        i = 0
        for chrom in sorted(positions):
            for pos, refa, alt in positions[chrom]:
                rc = ac = 0
                for pc in bamfile.pileup(chrom, pos - 1, pos, truncate=True,
                                         max_depth=255,
                                         multiple_iterators=False,
                                         flag_filter=filterflag):
                    for pcr in pc.pileups:
                        if pcr.query_position is None:
                            continue
                        pb = pcr.alignment.query_sequence[pcr.query_position]
                        if pb == refa:
                            rc += 1
                        elif pb == alt:
                            ac += 1
                rc = min(rc, 255)
                ac = min(ac, 255)
                if rc or ac:
                    ngen += 1
                Xi[0, i, 0] = rc
                Xi[0, i, 1] = ac
                i += 1
        logging.info("Loaded %d genotypes from: %s (%d)" % (ngen, filepath, ifile_idx))
        bamfile.close()
        return Xi
    elif filepath.endswith(('.vcf', '.vcf.gz', '.bcf')):
        vcffile = pysam.VariantFile(filepath)
        N = len(vcffile.header.samples)
        logging.info("Loading %d genotypes from: %s" % (N, filepath))
        Xi = np.zeros((N, n, 2), dtype=np.uint8)
        i = 0
        for chrom in sorted(positions):
            chriterator = iter(vcffile.fetch(chrom))
            rec = next(chriterator)
            for pos, refa, alt in positions[chrom]:
                while rec.pos < pos:
                    rec = next(chriterator)
                if rec.pos == pos:
                    for Ni, sample in enumerate(rec.samples):
                        refcnt = sum(1 for g in rec.samples[sample]['GT'] if g == 0)
                        altcnt = sum(1 for g in rec.samples[sample]['GT'] if g == 1)
                        Xi[Ni, i, 0] = refcnt
                        Xi[Ni, i, 1] = altcnt
                i += 1
        logging.info("Done loading %s." % filepath)
        return Xi
    else:
        logging.warning("Unknown file type: %s, skipping." % filepath)
        return None


def build_reference(targetfile, ifiles, outputprefix="imputeff",
                    region=None, maxvar=int(10e6),
                    addchr=False, rmchr=False, filterflag=3840,
                    nproc=1, cramref=None):
    """Preprocess BAM/VCF files against target sites into a reference pickle.

    Parameters
    ----------
    targetfile : str
        VCF with target sites, or .pos/.txt file (chrom pos ref alt).
    ifiles : list of str
        BAM/CRAM/VCF files or .txt manifests (one path per line).
    outputprefix : str
        Output pickle will be ``{outputprefix}.reference.pickle``.
    """
    logging.info("Loading target sites from: %s" % targetfile)
    positions = {}
    sigma = []
    avgr = 0.5
    n = 0

    if targetfile.endswith((".vcf", ".vcf.gz", ".bcf")):
        targets = pysam.VariantFile(targetfile)
        pp = None
        for i, rec in enumerate(targets.fetch(region=region)):
            chrom = rec.contig
            if addchr and not chrom.startswith('chr'):
                chrom = 'chr' + chrom
            elif rmchr and chrom.startswith('chr'):
                chrom = chrom[3:]
            pos = rec.pos
            ref = rec.ref
            if len(rec.alts) != 1:
                continue
            alt = rec.alts[0]
            if pp is not None:
                sigma.append(1 - ((pos - pp) * (avgr / 1e6)))
            positions.setdefault(chrom, []).append((pos, ref, alt))
            n += 1
            pp = pos
            if i + 1 == maxvar:
                break
    elif targetfile.endswith((".pos", ".txt")):
        with open(targetfile, 'r') as posfile:
            pp = None
            for line in posfile:
                chrom, pos, ref, alt = line.strip().split()
                if addchr and not chrom.startswith('chr'):
                    chrom = 'chr' + chrom
                elif rmchr and chrom.startswith('chr'):
                    chrom = chrom[3:]
                pos = int(pos)
                positions.setdefault(chrom, []).append((pos, ref, alt))
                if pp is not None:
                    sigma.append(1 - ((pos - pp) * (avgr / 1e6)))
                pp = pos
                n += 1

    logging.info("Loaded %d sites across %d chromosomes." % (n, len(positions)))
    sigma = np.array(sigma)

    # Expand manifests
    expanded = []
    for f in ifiles:
        if f.endswith(".txt"):
            with open(f, 'r') as manifest:
                for line in manifest:
                    path = line.rstrip()
                    if os.path.exists(path):
                        expanded.append(path)
                    else:
                        logging.error("File does not exist: %s" % path)
        else:
            if os.path.exists(f):
                expanded.append(f)
            else:
                logging.error("File does not exist: %s" % f)

    def _load(args):
        idx, fp = args
        return load_genotypes_from_file(idx, fp, n, positions,
                                        filterflag=filterflag,
                                        cramref=cramref)

    with multiprocessing.Pool(processes=nproc) as pool:
        X = pool.map(_load, list(enumerate(expanded)))

    X = [x for x in X if x is not None]
    X = np.concatenate(X, axis=0)

    outpath = '%s.reference.pickle' % outputprefix
    pickle.dump((X, sigma, positions), open(outpath, 'wb'))
    logging.info("Wrote reference to: %s" % outpath)
    return outpath


def train_model(reference_pickle, k=4, maxiter=40, outputprefix=None,
                initmodel=None, ngen=100, nproc=1):
    """EM-train an HMM on a reference pickle produced by ``build_reference``.

    Parameters
    ----------
    reference_pickle : str
        Path to the ``.reference.pickle`` file.
    k : int
        Number of HMM states.
    maxiter : int
        Maximum EM iterations.
    outputprefix : str or None
        Output model pickle name prefix.
    initmodel : str or None
        Path to a previously trained model for warm-starting.
    ngen : int
        Generations since founding.
    nproc : int
        Thread count for the C extension.

    Returns
    -------
    str
        Path to the trained model pickle.
    """
    if outputprefix is None:
        outputprefix = "%s_k%d_iter%d" % (
            os.path.basename(reference_pickle).replace(".pickle", "").replace(".gz", ""),
            k, maxiter)

    X, sigma, positions = pickle.load(open(reference_pickle, 'rb'))

    N = X.shape[0]
    n = X.shape[1]

    logging.info("Reference contains %d samples with %d genotypes" % (N, n))

    if initmodel is not None:
        start, a, sigma, emission, positions, k, allelecounts = \
            pickle.load(open(initmodel, 'rb'))
    else:
        a = np.ones((k, n - 1))
        a = a / a.sum(axis=0)
        start = np.ones(k) / k
        try:
            import rpy2.robjects as robjects
        except ImportError:
            logging.error(
                "Training init requires 'rpy2' (used to reproduce the original "
                "seeded R RNG). Install with `pip install rpy2`, or pass --initmodel."
            )
            sys.exit(1)
        emission = np.array(np.array(robjects.r("""
        set.seed(42)
        x <- array(runif(%d * %d))
        """ % (n, k))).reshape(k, n, order='F').tolist())

    start, a, sigma, emission, C = imputeff_hmm.fit(
        X, start, a, sigma, emission,
        maxiter=maxiter, nthreads=nproc, ngen=ngen)

    outpath = '%s.model.pickle' % outputprefix
    pickle.dump((start, a, sigma, emission, positions, k,
                 X.sum(axis=0)), open(outpath, 'wb'))
    logging.info("Wrote trained model to: %s" % outpath)
    return outpath


# ===================================================================
# Legacy / unused  (kept for backward compatibility and testing)
# ===================================================================

def OtoX(O, n, label, usetruthlabels=False):
    """LEGACY: convert list-of-tuples observations to allele-count matrix.
    Superseded by ``RtoX`` which uses the read-dictionary format.
    """
    X = np.zeros((n, 2), dtype=np.uint8)
    for idx, allele, qual, assigned, truth, isize, qname in O:
        if usetruthlabels:
            assigned = truth
        if assigned == label or label is None:
            X[idx][allele] += 1
    return X


def simR(imH, nimH, ipH, n, totreads=1000, ff=0.1):
    """LEGACY: simulate reads from three haplotypes for testing."""
    rid = 0
    R = {}

    for ri in np.random.randint(0, n, int(0.5 * totreads)):
        s = random.random()
        tl = 165
        R[(ri, rid)] = [
            [(ri, int(imH[ri]), 30)],
            0 if s < 0.5 else 1 if s < 1 - (ff / 2) else 2,
            0,
            tl,
        ]
        rid += 1

    for ri in np.random.randint(0, n, int((0.5 - (ff / 2)) * totreads)):
        s = random.random()
        tl = 165
        R[(ri, rid)] = [
            [(ri, int(nimH[ri]), 30)],
            0 if s < 0.5 else 1 if s < 1 - (ff / 2) else 2,
            1,
            tl,
        ]
        rid += 1

    for ri in np.random.randint(0, n, int((ff / 2) * totreads)):
        s = random.random()
        tl = 145
        R[(ri, rid)] = [
            [(ri, int(ipH[ri]), 30)],
            0 if s < 0.5 else 1 if s < 1 - (ff / 2) else 2,
            2,
            tl,
        ]
        rid += 1

    return R


def precision(R):
    """LEGACY: accuracy of label assignment vs truth labels (simulation only)."""
    c, e = 0, 0
    for r in R:
        if R[r][1] == R[r][2]:
            c += 1
        else:
            e += 1
    return c / (c + e) if (c + e) else 0.0
