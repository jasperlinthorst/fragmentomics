"""Population-reference HMM imputation of maternal/fetal genotypes.

Architecture
------------
The imputation pipeline models each haplotype as a mosaic of K reference
haplotypes connected by a Li & Stephens hidden Markov model.

Key Variables and Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~
a : ndarray, shape (k, n-1)
    Transition matrix: a[i, t] = P(state=i at position t | state at t-1).
    Under Li & Stephens, this encodes coalescence probabilities combined
    with recombination rates. Each column sums to 1.

sigma : ndarray, shape (n-1,)
    Probability of *no recombination* between consecutive sites.
    sigma[t] = exp(-d * rho) where d is distance in Morgans and rho
    is the population-scaled recombination rate. Used to compute the
    effective transition probabilities combined with the coalescent prior.

emission : ndarray, shape (k, n) or (k, n) float64
    Emission probabilities P(observed allele | hidden state).
    For standard haploid: uint8 0/1 for reference haplotype alleles.
    For trained models: float64 encoding P(alt | state=k at position=n).
    In diploid mode with adjusted emissions, this becomes float64
    encoding expected allele contributions accounting for the other haplotype.

y : ndarray, shape (2,) uint8
    Diploid genotype observation encoded as [ref_count, alt_count].
    Used in emission calculations: P(y | state) = emission[state, pos]^alt * 
    (1-emission[state, pos])^ref, or with binomial sampling for read data.

start : ndarray, shape (k,)
    Initial state probabilities P(state at position 0). Usually uniform
    (1/k) unless warm-starting from a trained model.

X : ndarray, shape (N, n, 2) uint8
    Reference panel allele counts. X[i, j, 0] = ref count, X[i, j, 1] = alt count
    for sample i at variant j. Summing over samples gives the pooled
    observation matrix used during EM training.

alpha : ndarray, shape (k*k, n) or (k, n)
    Forward probabilities in HMM forward-backward. alpha[s, t] = P(observations
    up to t, state=s at t | model). Computed recursively:
    alpha[t] = emission(y[t]) * sum_s(alpha[t-1] * trans[s->*]).

beta : ndarray, shape (k*k, n) or (k, n)
    Backward probabilities. beta[s, t] = P(observations from t+1 to end |
    state=s at t, model). Computed backwards: beta[t] = sum_s'(emission(y[t+1])
    * trans[*->s'] * beta[s', t+1]).

c : ndarray, shape (n,)
    Scaling coefficients for log-likelihood computation. c[t] normalizes
    alpha[t] to prevent underflow: c[t] = sum_s(alpha[s, t]).
    log-likelihood = sum_t(log(c[t])).

C : ndarray, shape (k*k, n) or similar
    Marginal probabilities of state pairs. Used in Baum-Welch (EM)
    training: C[(i*k+j), t] = P(state=i at t-1, state=j at t | data).
    These are the expected transition counts used to update 'a'.

gamma : ndarray, shape (k,) or (k*k,)
    Marginal state probabilities (expected state occupation counts).
    gamma[s] = P(state=s | data) = alpha[s] * beta[s] / P(data).
    Used in EM to update emission probabilities.

Modes of Operation
~~~~~~~~~~~~~~~~~~
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

**Custom reference training** (``cfstats imputeref``)
    When a phased reference panel is unavailable but a large collection
    of BAM/CRAM/VCF files is, the pipeline can learn HMM parameters
    from the data via EM. See ``build_reference`` (allele counting from
    BAM/VCF files) and ``train_model`` (Baum-Welch parameter estimation).

C Extension Interface
~~~~~~~~~~~~~~~~~~~~~
The compiled ``_hmm`` module provides OpenMP-parallelised forward and
backward passes. Two variants exist:

* ``forwardHaploid``, ``backwardHaploid``: Standard passes with uint8
  emission matrix (haplotype alleles as 0/1). Memory-efficient for
  standard imputation with phased reference panels.

* ``forwardHaploidDouble``, ``backwardHaploidDouble``: Float64 emission
  passes for diploid adjusted-emission calculations. Required when
  emission probabilities are not binary (e.g., 0.5 expected contribution
  from a heterozygous haplotype).

Both variants:
    - Release the Python GIL for true parallelism
    - Use OpenMP ``#pragma omp parallel for`` over haplotype states
    - Operate on column-major contiguous arrays for cache efficiency
    - Return log-likelihood and scaling coefficients for numerical stability

Mean-Field Extension to Three Haplotypes (NIPT with Fetal Fraction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The diploid mean-field approach (conditioning on one haplotype to separate
signals) can be extended to three haplotypes representing NIPT cfDNA:

*   **Three haplotypes:** inherited-maternal (IM), non-inherited-maternal (NIM),
    inherited-paternal (IP)

*   **Variable contributions (fetal fraction ff):**
    Each read comes from:
    - Maternal-only DNA: IM + NIM (probability 1-ff)
    - Fetal DNA: IM + IP (probability ff)
    
    Expected contribution fractions:
    - IM: 0.5*(1-ff) + 0.5*ff = 0.5 (always 50%)
    - NIM: 0.5*(1-ff) = 0.5*(1-ff)
    - IP: 0.5*ff

*   **Mean-field algorithm:**
    1. Initialize all three haplotype posteriors (e.g., from marginal genotypes)
    2. Iteratively for each haplotype:
       a. Sample a path for the haplotype being updated (e.g., IM)
       b. Adjust emissions accounting for contributions from the other two:
          E[allele|read] = 0.5*sampled_IP + (1-ff)*0.5*sampled_NIM + 0.5*target_IM
          The adjustment removes expected contributions from known haplotypes,
          leaving only the target haplotype signal plus noise.
       c. Forward-backward with adjusted emissions produces updated posterior
    3. Iterate until convergence (or fixed iterations)

*   **Emission adjustment formula:**
    For read r with allele a at position t, when updating haplotype h:
    
    adj_emission[i,t] = (1-w_h)*emission[other_hap_1,t] + w_h*emission[i,t]
    
    where w_h is the contribution weight (0.5 for IM, 0.5*(1-ff) for NIM,
    0.5*ff for IP). The adjustment blends the reference emission with the
    expected contribution from the other conditioned haplotypes.

*   **Alternative: Direct three-haplotype HMM:**
    Instead of mean-field, a full k^3 state HMM tracks all three haplotypes
    simultaneously. Transition becomes a 3-way coalescence problem, and
    emission is a mixture of three contributions. This is theoretically
    cleaner but O(k^3) in memory and O(k^3*n) in time, which may be
    prohibitive for k>10. The mean-field approach remains O(k^2) per iteration.

See ``impute_diploid`` for the diploid mean-field implementation; triploid
would follow similar structure with three haplotypes and ff-weighted emissions.
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


def _get_command_line_header():
    """Generate VCF header line with cfstats impute command and timestamp."""
    from datetime import datetime
    cmd = ' '.join(sys.argv) if sys.argv else 'cfstats impute'
    date_str = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    return f'##cfstats_imputeCommand={cmd}; Date={date_str}'


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


def read_w(R, r):
    """Per-read fetal posterior ``w_i`` for read key *r* (None if unavailable).

    Stored as the optional 5th element of ``R[r]`` (see :func:`getR`); reads
    loaded without an ``XF`` tag (or by legacy callers) have no 5th element and
    return ``None``, signalling "use the global fetal-fraction prior".
    """
    entry = R[r]
    return entry[4] if len(entry) > 4 else None


def read_label_logprior(w, ff):
    """Log per-read label prior ``[log P(IMH), log P(NIMH), log P(IPH)]``.

    The three triploid labels are 0=IMH (inherited maternal, shared by mother
    and fetus), 1=NIMH (non-inherited maternal = maternal-only) and 2=IPH
    (inherited paternal = fetal-only).  A per-read fetal posterior
    ``w = P(read is fetal-derived)`` induces the prior::

        P(IMH)  = 0.5            # always shared
        P(NIMH) = 0.5 * (1 - w)  # maternal-only
        P(IPH)  = 0.5 * w        # fetal-only

    When ``w is None`` this falls back to the global fetal-fraction prior
    ``[0.5, 0.5 - ff/2, ff/2]``, so ``w = ff`` recovers the original model
    exactly (the per-read prior is a strict generalization).
    """
    if w is None:
        p = (0.5, 0.5 - ff / 2.0, ff / 2.0)
    else:
        p = (0.5, 0.5 * (1.0 - w), 0.5 * w)
    return [np.log(pi + 1e-308) for pi in p]


def read_prior_loglik(R, ff=0.1):
    """Sum of per-read label log-priors over all reads at their current labels."""
    total = 0.0
    for r in R:
        total += read_label_logprior(read_w(R, r), ff)[R[r][1]]
    return total


# ===================================================================
# Forward-backward utilities
# ===================================================================

def forward_backward_haploid(x, sigma, emission, nthreads=1, alpha_mat=None,
                             minp=0.01):
    """Haploid forward-backward with uint8 emission.

    Returns the posterior *gamma* matrix (k, n) where ``gamma[h, t]`` is
    the probability that haplotype *h* is the underlying state at site *t*.

    Parameters
    ----------
    alpha_mat : ndarray (k, n-1) float64 or None
        Learned switching targets.  ``alpha_mat[i, t]`` is the probability
        of switching *into* state *i* when a recombination occurs between
        sites *t* and *t+1*.  Each column should sum to 1.  When ``None``
        (the default), a uniform 1/k is used for every state and position
        (classic Li & Stephens).
    minp : float
        Minimum emission probability floor to prevent underflow.
        Default is 0.01.
    """
    k = emission.shape[0]
    ini = np.ones(k) / k
    alpha, ac = imputeff_hmm.forwardHaploid(
        x, ini, sigma, emission, scale=1, nthreads=nthreads,
        alphaMat=alpha_mat, minp=minp)

    beta, bc = imputeff_hmm.backwardHaploid(
        x, sigma, emission, scale=ac, nthreads=nthreads,
        alphaMat=alpha_mat, minp=minp)

    gamma = alpha * beta / ac

    if np.any(np.isnan(gamma)):
        logging.warning("NaN values detected in gamma matrix, replacing with zeros")

    gamma = np.nan_to_num(gamma, nan=0.0)
    return gamma


def forward_backward_haploid_double(x, sigma, emission_probs, nthreads=1,
                                    alpha_mat=None):
    """Haploid forward-backward with float64 emission.

    Identical to ``forward_backward_haploid`` except emissions are
    continuous probabilities (not binary {0,1}).  Used by the diploid
    mean-field variational inference where the emission of each
    reference haplotype is adjusted to condition on the other haplotype.

    Parameters
    ----------
    alpha_mat : ndarray (k, n-1) float64 or None
        Learned switching targets (see ``forward_backward_haploid``).
    """
    k = emission_probs.shape[0]
    ini = np.ones(k) / k
    em = np.ascontiguousarray(emission_probs, dtype=np.float64)
    alpha, ac = imputeff_hmm.forwardHaploidDouble(
        x, ini, sigma, em, scale=1, nthreads=nthreads,
        alphaMat=alpha_mat)
    beta, bc = imputeff_hmm.backwardHaploidDouble(
        x, sigma, em, scale=ac, nthreads=nthreads,
        alphaMat=alpha_mat)
    gamma = alpha * beta / ac

    if np.any(np.isnan(gamma)):
        logging.warning("NaN values detected in gamma matrix, replacing with zeros")

    gamma = np.nan_to_num(gamma, nan=0.0)
    return gamma


# ===================================================================
# Reference loading
# ===================================================================


def load_genetic_map(genetic_map_file):
    """Load a PLINK-format genetic map file into position/cM arrays.

    Expected columns (space- or tab-separated, with header)::

        position  COMBINED_rate.cM.Mb.  Genetic_Map.cM.

    Parameters
    ----------
    genetic_map_file : str
        Path to a (gzipped) genetic map file.

    Returns
    -------
    positions : ndarray (m,) int
        Physical positions (bp, 1-based).
    genetic_pos : ndarray (m,) float
        Cumulative genetic position in cM.
    """
    positions = []
    genetic_pos = []
    opener = gzip.open if genetic_map_file.endswith('.gz') else open
    with opener(genetic_map_file, 'rt') as fh:
        fh.readline()  # skip header
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                positions.append(int(parts[0]))
                genetic_pos.append(float(parts[2]))
    return np.array(positions, dtype=np.int64), np.array(genetic_pos, dtype=np.float64)


def sigma_from_positions(positions, nGen, minp=0.01, genetic_map=None):
    """Compute sigma (no-recombination probability) between consecutive sites.

    Uses the Li & Stephens formula ``sigma = exp(-nGen * d_Morgan)``
    where ``d_Morgan`` is the genetic distance in Morgans between sites.

    If *genetic_map* is provided it is a ``(map_positions, map_cM)`` tuple
    returned by :func:`load_genetic_map`; genetic distance is interpolated
    from the map.  Otherwise a uniform rate of 1 cM/Mb is assumed.

    Parameters
    ----------
    positions : sequence of int
        Physical positions (bp) of the variant sites.
    nGen : int
        Effective number of generations (population size parameter).
    minp : float
        Minimum sigma value (clips ``1 - sigma`` away from 1).
    genetic_map : tuple (map_pos, map_cM) or None
        Pre-loaded genetic map from :func:`load_genetic_map`.

    Returns
    -------
    sigma : ndarray (n-1,)
        Probability of no recombination between consecutive sites.
    """
    positions = np.asarray(positions, dtype=np.int64)
    n = len(positions)
    if n < 2:
        return np.array([], dtype=np.float64)

    if genetic_map is not None:
        map_pos, map_cM = genetic_map
        cM_at_sites = np.interp(positions, map_pos, map_cM)
        d_Morgan = np.diff(cM_at_sites) / 100.0
    else:
        d_Mb = np.diff(positions) / 1e6
        d_Morgan = d_Mb / 100.0  # 1 cM/Mb

    sigma = np.exp(-nGen * d_Morgan)
    # sigma = np.clip(sigma, minp, 1.0 - minp)
    return sigma


def loadref_vcf(vcf, contig=None, start=None, stop=None,
                nGen=100, avgr=1, minp=0.01, genetic_map=None):
    """Load a phased reference VCF and return ``(sigma, emission, variants)``.

    Tries ``bcftools`` for fast bulk GT extraction first; falls back to
    ``pysam`` if bcftools is unavailable.

    Automatically detects and loads cfstats-trained model VCFs (with IMP_K
    header) vs regular phased reference VCFs.

    Parameters
    ----------
    genetic_map : tuple (map_pos, map_cM) or None
        Pre-loaded genetic map from :func:`load_genetic_map`.  When provided,
        sigma is computed from interpolated genetic distances rather than
        assuming a uniform recombination rate.

    Returns
    -------
    sigma : ndarray (n-1,)
        Probability of *no* recombination between consecutive sites.
    emission : ndarray (k, n) uint8 or float64
        Reference haplotype alleles (0=ref, 1=alt) for regular VCFs,
        or emission probabilities for trained model VCFs.
    variants : list of (id, pos, ref, alt)
        One entry per site.
    """
    # Check if this is a trained model VCF by looking for IMP_K header
    if _is_trained_model_vcf(vcf):
        logging.info("Detected trained model VCF format")
        return _loadref_vcf_trained(vcf, contig, start, stop)

    try:
        return _loadref_vcf_bcftools(vcf, contig, start, stop, nGen, avgr, minp,
                                     genetic_map=genetic_map)
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        print("bcftools not available, falling back to pysam (slower).")
        return _loadref_vcf_pysam(vcf, contig, start, stop, nGen, avgr, minp,
                                   genetic_map=genetic_map)


def _is_trained_model_vcf(vcf_path):
    """Check if VCF is a cfstats trained model by looking for IMP_K header."""
    import gzip
    try:
        if vcf_path.endswith('.gz'):
            with gzip.open(vcf_path, 'rt') as f:
                for line in f:
                    if line.startswith('##'):
                        if 'IMP_K=' in line:
                            return True
                    elif line.startswith('#CHROM'):
                        return False
        else:
            with open(vcf_path, 'r') as f:
                for line in f:
                    if line.startswith('##'):
                        if 'IMP_K=' in line:
                            return True
                    elif line.startswith('#CHROM'):
                        return False
        return False
    except Exception:
        return False


def _loadref_vcf_trained(vcf, contig=None, start=None, stop=None):
    """Load a cfstats trained model VCF.

    Extracts emission probabilities from PL fields, transition
    probabilities from INFO/IMP_TP, and recombination from INFO/IMP_RP.

    Returns
    -------
    sigma : ndarray (n-1,)
        Recombination probabilities (IMP_RP from INFO).
    emission : ndarray (k, n) float64
        Emission probabilities decoded from PL fields.
    variants : list of (id, pos, ref, alt)
        One entry per site.
    """
    import gzip
    import tempfile
    import shutil

    # Handle gzip files: decompress to temp file for pysam compatibility
    vcf_path = vcf
    if vcf.endswith('.gz'):
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.vcf', delete=False) as tmp:
            with gzip.open(vcf, 'rb') as gz:
                shutil.copyfileobj(gz, tmp)
            vcf_path = tmp.name

    try:
        vcffile = pysam.VariantFile(vcf_path)

        # Read header metadata
        k = None
        start_probs = None
        header = vcffile.header
        for record in header.records:
            if record.key == 'IMP_K':
                k = int(record.value)
            elif record.key == 'IMP_START':
                start_probs = np.array([float(x) for x in record.value.split(',')])

        if k is None:
            raise ValueError("Trained model VCF missing IMP_K header")

        logging.info("Loading trained model with k=%d states" % k)

        # Collect variants
        variants = []
        sigma_list = []
        emission_list = []

        # Build fetch region
        region = None
        if contig:
            region = contig
            if start is not None and stop is not None:
                region = f"{contig}:{start}-{stop}"
            elif start is not None:
                region = f"{contig}:{start}"

        iterator = vcffile.fetch(region) if region else vcffile.fetch()

        for rec in iterator:
            vid = rec.id or '.'
            variants.append((vid, rec.pos, rec.ref, rec.alts[0] if rec.alts else '.'))

            # Get recombination probability
            rp = rec.info.get('IMP_RP', 0.5)  # Default if not found (first variant)
            sigma_list.append(rp)

            # Get PL fields and decode emission probabilities
            # PL = [phred(P_ref), 0, phred(P_alt)]
            # phred(P) = -10 * log10(P)
            # So P_alt = 10^(-PL[2]/10)
            var_emission = []
            for sample in rec.samples.values():
                pl = sample.get('PL', None)
                if pl is not None and len(pl) >= 3:
                    # Decode P(alt) from PL[2]
                    p_alt = 10 ** (-pl[2] / 10.0)
                    var_emission.append(p_alt)
                else:
                    var_emission.append(0.5)  # Default neutral
            emission_list.append(var_emission)

        vcffile.close()

        # Convert to arrays
        n = len(variants)
        sigma = np.array(sigma_list[:n-1] if len(sigma_list) > n-1 else sigma_list, dtype=np.float64)

        # emission shape: (k, n)
        emission = np.array(emission_list, dtype=np.float64).T  # (n, k) -> (k, n)

        # Validate shapes
        if emission.shape != (k, n):
            logging.warning("Emission shape mismatch: expected (%d, %d), got %s" % (k, n, emission.shape))
            # Pad or truncate as needed
            new_emission = np.full((k, n), 0.5, dtype=np.float64)
            min_k = min(emission.shape[0], k)
            min_n = min(emission.shape[1], n)
            new_emission[:min_k, :min_n] = emission[:min_k, :min_n]
            emission = new_emission

        logging.info("Loaded trained model: sigma=%s, emission=%s, variants=%d" %
                     (sigma.shape, emission.shape, len(variants)))

        return sigma, emission, variants
    finally:
        # Clean up temp file if created
        if vcf.endswith('.gz') and vcf_path != vcf and os.path.exists(vcf_path):
            os.unlink(vcf_path)


def _loadref_vcf_bcftools(vcf, contig, start, stop, nGen, avgr, minp,
                          genetic_map=None):
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
    for line in info_raw.split('\n'):
        vid, pos_s, ref, alt = line.split('\t')
        variants.append((vid, int(pos_s), ref, alt))

    n_variants = len(variants)
    positions = [v[1] for v in variants]
    sigma = sigma_from_positions(positions, nGen, minp=minp, genetic_map=genetic_map)

    # Query 2: GT data only -- extract digits with numpy for speed
    cmd_gt = ['bcftools', 'query', '-f',
              '[\t%GT]\n'] + region_args + [vcf]
    gt_raw = subprocess.check_output(cmd_gt)

    raw_arr = np.frombuffer(gt_raw, dtype=np.uint8)
    digits = raw_arr[(raw_arr >= 48) & (raw_arr <= 57)] - 48
    n_haplotypes = len(digits) // n_variants

    emission = digits.reshape(n_variants, n_haplotypes).T.copy()

    print("emission from vcf:",emission.shape, n_haplotypes)

    return sigma, emission, variants


def _loadref_vcf_pysam(vcf, contig, start, stop, nGen, avgr, minp,
                       genetic_map=None):
    """Fallback VCF loading using pysam (slow for large panels)."""
    vcffile = pysam.VariantFile(vcf)
    emission = []
    variants = []

    for rec in vcffile.fetch(contig=contig, start=start, stop=stop):
        variants.append((rec.id, rec.pos, rec.ref, rec.alts[0]))
        col = [np.uint8(g) for gt in rec.samples.values() for g in gt['GT']]
        emission.append(col)

    positions = [v[1] for v in variants]
    sigma = sigma_from_positions(positions, nGen, minp=minp, genetic_map=genetic_map)
    emission = np.array(emission).transpose().copy()
    vcffile.close()
    return sigma, emission, variants


def loadref_haplegend(hapfile, legendfile,
                      nGen=100, avgr=1, minp=0.01,
                      start=None, stop=None, genetic_map=None):
    """Load a phased haplotype reference from hap.gz / legend.gz (IMPUTE format)."""
    variants = []
    keep_idx = []

    with gzip.open(legendfile, 'rt') as legend:
        legend.readline()
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
            idx += 1

    with gzip.open(hapfile, 'rb') as hap:
        raw = hap.read()
    emission = np.loadtxt(io.BytesIO(raw), dtype=np.uint8)
    if keep_idx and len(keep_idx) < emission.shape[0]:
        emission = emission[keep_idx, :]
    emission = emission.T.copy()

    positions = [v[1] for v in variants]
    sigma = sigma_from_positions(positions, nGen, minp=minp, genetic_map=genetic_map)
    return sigma, emission, variants


# ===================================================================
# Read loading
# ===================================================================

def getR(file, chrom, variants, ref=None,
         addchr=False, rmchr=False, ff=0.1, stepper='all', min_base_quality=17,
         read_prior=False, read_prior_tag='XF', phred_max=60):
    """Pile up reads from *file* at the reference *variants*.

    Returns a dict keyed by ``(reference_start, read_name)`` with value
    ``[observations, assigned_label, truth_label_or_None, template_length,
    w_i]``.  Each observation is ``(variant_index, allele, base_quality)``.
    Initial label assignment is random (50% label 0, ~(50-ff/2)% label 1,
    ~ff/2% label 2) to seed the Gibbs sampler.

    When *read_prior* is set, the per-read fetal posterior is decoded from the
    Phred-encoded ``read_prior_tag`` (default ``XF``, written by the read
    classifier) as ``w_i = 10**(-XF/10)`` and stored as the 5th element; reads
    without the tag get ``w_i = None`` (global-prior fallback).
    """
    R = {}
    bamfile = pysam.AlignmentFile(file, reference_filename=ref)
    i = 0

    if addchr:
        chrom = 'chr' + chrom
    if rmchr:
        chrom = chrom.replace('chr', '')

    reads_considered=set()
    for vid, pos, refa, alt in variants:
        for pc in bamfile.pileup(chrom, int(pos) - 1, int(pos),
                                 truncate=True, multiple_iterators=False, 
                                 stepper=stepper, min_base_quality=min_base_quality, min_mapping_quality=min_base_quality):
            for pcr in pc.pileups:
                if pcr.query_position is None or pcr.alignment.is_secondary or pcr.alignment.is_unmapped or pcr.alignment.is_duplicate:
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
                    w_i = None
                    if read_prior and pcr.alignment.has_tag(read_prior_tag):
                        q = float(pcr.alignment.get_tag(read_prior_tag))
                        q = min(max(q, 0.0), float(phred_max))
                        w_i = 10.0 ** (-q / 10.0)
                    R[k] = [
                        [(i, a, pq)],
                        0 if s < 0.5 else 1 if s < 1 - (ff / 2) else 2,
                        None,
                        pcr.alignment.template_length,
                        w_i,
                    ]
                    reads_considered.add(pcr.alignment.query_name)
        i += 1
        if i % 10000 == 0:
            print("Processing variants (%d/%d)..." % (i, len(variants)))

    logging.info("%d reads considered."%len(reads_considered))
    
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
    minp = 0.01

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


def preselect_haplotypes_random(k, nhap, seed=None):
    """Randomly select *nhap* haplotypes from the full panel (random-init).

    Parameters
    ----------
    k : int
        Total number of haplotypes in reference panel.
    nhap : int
        Number of haplotypes to select.
    seed : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    list of int
        Sorted list of selected haplotype indices (0-based).
    """
    if seed is not None:
        np.random.seed(seed)
    nhap = min(nhap, k)
    selected = np.random.choice(k, size=nhap, replace=False)
    return sorted(selected.tolist())


def rescore_haplotypes_posterior(R, emission_full, gamma, nhap, minp=0.01):
    """Score haplotypes by posterior-weighted read likelihood.

    For each haplotype in the full panel, computes the sum of posterior
    probabilities at read positions, weighted by allele match/mismatch.
    This is used to iteratively refine the haplotype subset.

    Parameters
    ----------
    R : dict
        Read dictionary from ``getR``.
    emission_full : ndarray (K, n) uint8
        Full reference panel (all K haplotypes).
    gamma : ndarray (k_subset, n)
        Posterior probabilities from forward-backward (for current subset).
    nhap : int
        Number of top haplotypes to return.
    minp : float
        Floor for emission probabilities.

    Returns
    -------
    list of int
        Indices of top-scoring haplotypes from the FULL panel (0-based).
    """
    K_full = emission_full.shape[0]

    # Collect read positions and alleles
    obs = []
    for r in R:
        for idx, allele, qual in R[r][0]:
            obs.append((idx, allele))

    if not obs:
        return list(range(min(nhap, K_full)))

    obs_arr = np.array(obs, dtype=np.int64)
    indices = obs_arr[:, 0]
    alleles = obs_arr[:, 1]

    # Get full panel alleles at observation positions
    hap_at_obs = emission_full[:, indices].astype(np.float64)
    hap_at_obs = np.clip(hap_at_obs, minp, 1 - minp)

    # For each site, get the max posterior across the current subset
    # This approximates how well each full-panel hap matches the "best" path
    alt_mask = (alleles == 1).astype(np.float64)
    ref_mask = 1.0 - alt_mask

    # Score = sum over reads of (log P(match) * indicator)
    # Similar to preselect_haplotypes but using posterior-weighting
    scores = (np.log(hap_at_obs) * alt_mask[np.newaxis, :] +
              np.log(1.0 - hap_at_obs) * ref_mask[np.newaxis, :]).sum(axis=1)

    best = np.argsort(scores)[-nhap:]
    return list(best)


# ===================================================================
# Diploid imputation  (mean-field variational)
# ===================================================================

def diploid_emission(emission_is_alt, h_alleles, minp=0.01):
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


def recast_haps(hap1, hap2, gp0, gp1, gp2):
    """Force phased haplotype calls to agree with the GP argmax genotype.

    For each site the argmax genotype
    ``gt3 = argmax(gp0, gp1, gp2)`` is compared to ``round(hap1)+round(hap2)``.
    Sites that disagree are corrected:

    - GT=0 → hap1=0, hap2=0
    - GT=2 → hap1=1, hap2=1
    - GT=1 → the haplotype with higher probability gets the alt allele

    Parameters
    ----------
    hap1, hap2 : ndarray (n,) float
        Per-site marginal P(alt) for each phased haplotype.
    gp0, gp1, gp2 : ndarray (n,) float
        Posterior genotype probabilities for 0/0, 0/1, 1/1.

    Returns
    -------
    hap1, hap2 : ndarray (n,) float
        Corrected haplotype probabilities (values remain soft but
        discordant sites are snapped to 0 or 1).
    """
    hap1 = hap1.copy()
    hap2 = hap2.copy()

    gp = np.stack([gp0, gp1, gp2], axis=1)  # (n, 3)
    gt3 = np.argmax(gp, axis=1)              # 0, 1, or 2
    gt1 = np.round(hap1).astype(int) + np.round(hap2).astype(int)

    to_change = np.where(gt3 != gt1)[0]

    # GT=0: both haplotypes → ref
    w0 = to_change[gt3[to_change] == 0]
    hap1[w0] = 0.0
    hap2[w0] = 0.0

    # GT=2: both haplotypes → alt
    w2 = to_change[gt3[to_change] == 2]
    hap1[w2] = 1.0
    hap2[w2] = 1.0

    # GT=1: assign alt to the haplotype with higher current probability
    w1 = to_change[gt3[to_change] == 1]
    a1 = hap1[w1]
    a2 = hap2[w1]
    hap1[w1] = np.where(a1 >= a2, 1.0, 0.0)
    hap2[w1] = np.where(a1 >= a2, 0.0, 1.0)

    return hap1, hap2


def impute_diploid(R, sigma, emission, nhap=None, n_iter=3,
                   minp=0.01, nthreads=1, use_random_init=False, knew=None,
                   random_seed=None, phasing_iter=True, dump_prefix=None):
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
    use_random_init : bool
        If True, use random initial selection and iterative
        posterior-guided haplotype re-selection (no msPBWT dependency).
    knew : int or None
        Number of new haplotypes to add per iteration when using random-init.
        If None, defaults to nhap (replace entire panel each iteration).
    random_seed : int or None
        Random seed for random-init initial selection.
    phasing_iter : bool
        If True (default), run one extra forward-backward pass warm-started
        from the converged read assignments and call ``recast_haps`` to
        produce per-site phased haplotype calls consistent with the GP
        argmax genotype.  The phased calls replace the averaged marginals
        for GT/DS/GP in the output VCF.

    Returns
    -------
    p1 : ndarray (n,)
        Marginal P(alt) for haplotype 1.
    p2 : ndarray (n,)
        Marginal P(alt) for haplotype 2.
    emission : ndarray (k_sel, n) uint8
        The (possibly sub-selected) emission matrix used.
    phased_haps : tuple (hap1, hap2) or None
        Per-site phased 0/1 calls after ``recast_haps`` correction, or
        ``None`` when *phasing_iter* is False.
    """
    K_full = emission.shape[0]  # Remember full panel size
    emission_full = emission.copy()  # Keep full panel for random-init re-selection
    hap_indices_full = None  # Track mapping from subset to full panel

    # --- haplotype pre-selection ---
    if nhap is not None and isinstance(nhap, int):
        if use_random_init:
            # Random-init: random initial selection
            hap_indices = preselect_haplotypes_random(K_full, nhap, seed=random_seed)
            print("Random-init: Using %d randomly selected haplotypes (seed=%s)." %
                  (len(hap_indices), random_seed))
        else:
            # Original: overlap-based selection
            print(nhap, emission.shape[1])
            hap_indices = preselect_haplotypes(R, emission, nhap, emission.shape[1])
            print("Using %d pre-selected haplotypes for diploid pass." % len(hap_indices))
        hap_indices_full = hap_indices.copy()
        emission = emission[hap_indices, :]

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

    _dump_hap_indices = hap_indices_full if hap_indices_full is not None else np.arange(emission.shape[0], dtype=np.int32)
    if dump_prefix is not None:
        np.savez_compressed(
            '%s_iter0_init.npz' % dump_prefix,
            gamma=gamma1.astype(np.float32),
            emission=emission.astype(np.uint8),
            sigma=sigma.astype(np.float64),
            hap_path=np.zeros(n, dtype=np.uint8),
            hap_indices=_dump_hap_indices,#.astype(np.int32),
        )
        print("Dumped initial pass: %s_iter0_init.npz" % dump_prefix)

    # Set default Knew for random-init mode
    if use_random_init and knew is None:
        knew = nhap  # Default: replace entire panel each iteration

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

        if dump_prefix is not None:
            np.savez_compressed(
                '%s_iter%d_hap1.npz' % (dump_prefix, gi + 1),
                gamma=gamma1.astype(np.float32),
                emission=emission.astype(np.uint8),
                sigma=sigma.astype(np.float64),
                hap_path=h2_alleles.astype(np.uint8),
                hap_indices=_dump_hap_indices #.astype(np.int32),
            )
            np.savez_compressed(
                '%s_iter%d_hap2.npz' % (dump_prefix, gi + 1),
                gamma=gamma2.astype(np.float32),
                emission=emission.astype(np.uint8),
                sigma=sigma.astype(np.float64),
                hap_path=h1_alleles.astype(np.uint8),
                hap_indices=_dump_hap_indices#.astype(np.int32),
            )
            print("Dumped iter %d: %s_iter%d_hap{1,2}.npz" % (gi + 1, dump_prefix, gi + 1))

        if gi >= burnin:
            p1_sum += p1_cur
            p2_sum += p2_cur
            n_avg += 1

        print("Diploid iter %d: p1_mean=%.4f p2_mean=%.4f%s" %
              (gi + 1, p1_cur.mean(), p2_cur.mean(),
               " (averaging)" if gi >= burnin else " (burn-in)"))

        # --- Random-init iterative haplotype re-selection ---
        if use_random_init and gi < n_iter - 1 and nhap is not None:
            # Combine gamma1 and gamma2 to get per-hap posterior scores
            gamma_combined = (gamma1 + gamma2) / 2.0

            # Re-score all haplotypes in FULL panel based on current posteriors
            # Use the original overlap-based scoring (posterior-weighted)
            new_full_indices = rescore_haplotypes_posterior(
                R, emission_full, gamma_combined, nhap, minp=minp)

            # Random-init: keep (nhap - knew) from current, add knew new
            current_full_set = set(hap_indices_full)
            new_full_set = set(new_full_indices)

            # Keep some from current set
            n_keep = max(0, nhap - knew)
            if n_keep > 0:
                # Randomly sample from current set to keep
                keep_candidates = list(current_full_set)
                if len(keep_candidates) >= n_keep:
                    keep = np.random.choice(keep_candidates, size=n_keep, replace=False)
                else:
                    keep = keep_candidates
            else:
                keep = []

            # Add new ones from the scored set (excluding those we kept)
            add_candidates = [h for h in new_full_indices if h not in keep]
            n_add = nhap - len(keep)
            if len(add_candidates) >= n_add:
                add = add_candidates[-n_add:]  # Take top n_add
            else:
                # Not enough new candidates, fill randomly from remaining full panel
                remaining = [h for h in range(K_full) if h not in keep and h not in add_candidates]
                need = n_add - len(add_candidates)
                extra = np.random.choice(remaining, size=need, replace=False) if need > 0 else []
                add = add_candidates + list(extra)

            # Update the haplotype subset
            hap_indices_full = sorted(list(keep) + list(add))
            emission = emission_full[hap_indices_full, :]
            k = emission.shape[0]
            is_alt = (emission > 0.5)
            print("  Random-init re-selection: kept %d, added %d new (total %d)" %
                  (len(keep), len(add), len(hap_indices_full)))

    p1 = p1_sum / max(n_avg, 1)
    p2 = p2_sum / max(n_avg, 1)
    print("Averaged %d post-burn-in samples" % n_avg)

    # --- optional phasing pass (warm-started from converged labels) ---
    phased_haps = None
    if phasing_iter:
        print("Phasing iteration (warm-start from converged labels)")
        ph1_alleles = sample_haplotype_path(gamma1, emission)
        eff2_ph = diploid_emission(is_alt, ph1_alleles, minp)
        gamma2_ph = forward_backward_haploid_double(
            x_all, sigma, eff2_ph, nthreads=nthreads)

        ph2_alleles = sample_haplotype_path(gamma2_ph, emission)
        eff1_ph = diploid_emission(is_alt, ph2_alleles, minp)
        gamma1_ph = forward_backward_haploid_double(
            x_all, sigma, eff1_ph, nthreads=nthreads)

        ph1 = (gamma1_ph * emission).sum(axis=0) / np.maximum(gamma1_ph.sum(axis=0), 1e-300)
        ph2 = (gamma2_ph * emission).sum(axis=0) / np.maximum(gamma2_ph.sum(axis=0), 1e-300)

        gp0 = (1 - ph1) * (1 - ph2)
        gp1 = ph1 * (1 - ph2) + (1 - ph1) * ph2
        gp2 = ph1 * ph2
        hap1_phased, hap2_phased = recast_haps(ph1, ph2, gp0, gp1, gp2)
        phased_haps = (hap1_phased, hap2_phased)
        n_recast = int(np.sum(
            (np.round(ph1) + np.round(ph2)) != (np.round(hap1_phased) + np.round(hap2_phased))
        ))
        print("Phasing: recast_haps corrected %d/%d sites" % (n_recast, n))

    return p1, p2, emission, phased_haps


def _build_gl_per_hap(R, n, labels, read_keys, default_bq=30):
    """Build per-haplotype genotype-likelihood arrays.

    ``gl_j`` is a ``(2, n)`` array where ``gl_j[0, i]`` is the joint
    likelihood of all observations at site *i* assigned to hap *j*
    under the hypothesis that the haplotype carries the *reference*
    allele at *i*, and ``gl_j[1, i]`` is the likelihood under the
    *alternate* hypothesis.  Each observation contributes a factor of
    ``1 - 10^(-Q/10)`` (correct call) or ``10^(-Q/10)`` (error) using
    the per-base Phred quality.

    Parameters
    ----------
    labels : ndarray (nReads,) int8 in {1, 2}
    default_bq : int
        Phred score used when an observation's quality is missing/zero.

    Returns
    -------
    gl1, gl2 : ndarray (2, n) float64
        Column-normalised so that ``gl_j[:, i].sum() == 1`` (preserves
        ratios, prevents underflow when many reads stack at one site).
    """
    gl1 = np.ones((2, n), dtype=np.float64)
    gl2 = np.ones((2, n), dtype=np.float64)
    for i, key in enumerate(read_keys):
        target = gl1 if labels[i] == 1 else gl2
        for idx, allele, qual in R[key][0]:
            bq = qual if qual and qual > 0 else default_bq
            p_err = 10.0 ** (-bq / 10.0)
            p_corr = 1.0 - p_err
            if allele == 1:
                target[1, idx] *= p_corr
                target[0, idx] *= p_err
            else:
                target[0, idx] *= p_corr
                target[1, idx] *= p_err
            # renormalise per-site to avoid underflow (ratio preserved)
            s = target[0, idx] + target[1, idx]
            if s > 0:
                target[0, idx] /= s
                target[1, idx] /= s
    return gl1, gl2


def _gl_to_emission_probs(gl, emission, minp=1e-6):
    """Combine a per-hap ``gl`` array with the reference panel.

    Returns the ``(k, n)`` float64 matrix
    ``em_eff[k, i] = gl[0, i] * (1 - emission[k, i]) + gl[1, i] * emission[k, i]``
    which is the marginal emission likelihood of state *k* at site *i*
    integrated over the unknown true allele of hap *j* (REF or ALT).
    This is the ``g`` matrix fed to ``forwardHaploidDouble`` when ``y`` is
    set to ``(0, 1)`` at every site (so the C kernel computes ``e = g``).
    """
    em_f = emission.astype(np.float64)
    em_eff = (gl[0, :][np.newaxis, :] * (1.0 - em_f)
              + gl[1, :][np.newaxis, :] * em_f)
    em_eff = np.clip(em_eff, minp, 1.0 - minp)
    return em_eff


def _gl_dummy_y(n):
    """Return the ``(n, 2)`` uint8 array ``[[0, 1], [0, 1], ...]``.

    Used together with :func:`_gl_to_emission_probs`: passing this *y*
    plus the gl-combined emission *g* to ``forwardHaploidDouble`` makes
    the inner kernel compute ``e = (1 - g)^0 * g^1 = g``, i.e. it uses
    the gl-combined emission directly without applying counts.
    """
    y = np.zeros((n, 2), dtype=np.uint8)
    y[:, 1] = 1
    return y


def forward_backward_gl(gl, sigma, emission, nthreads=1, minp=1e-6):
    """Run the haploid F/B with a genotype-likelihood (gl) emission.

    Builds the combined ``(k, n)`` emission and dispatches to
    :func:`forward_backward_haploid_double`.  Returns the posterior
    ``gamma`` matrix.
    """
    n = emission.shape[1]
    em_eff = _gl_to_emission_probs(gl, emission, minp=minp)
    y = _gl_dummy_y(n)
    return forward_backward_haploid_double(y, sigma, em_eff, nthreads=nthreads)


def _resample_read_labels_gl(R, read_keys, e1, e2, rng, default_bq=30,
                             minp=1e-6):
    """Hard-resample each read's haplotype label using BQ-aware likelihood.

    Replaces the binary-match version with a proper sequencing-error model:
    for an observed allele *a* at site *i* and per-base error ``p_err``,

        P(obs = a | hap j) = (1 - p_err) * P(hap_j has a | gamma)
                           + p_err       * P(hap_j has 1-a | gamma)

    where ``P(hap_j has ALT)`` at site *i* is ``e_j[i]`` (the
    panel-posterior expected alt freq returned by the gl F/B pass).
    """
    e1 = np.clip(e1, minp, 1.0 - minp)
    e2 = np.clip(e2, minp, 1.0 - minp)

    labels = np.empty(len(read_keys), dtype=np.int8)
    for i, key in enumerate(read_keys):
        obs = R[key][0]
        if not obs:
            labels[i] = 1 if rng.random() < 0.5 else 2
            continue
        ll1 = 0.0
        ll2 = 0.0
        for idx, allele, qual in obs:
            bq = qual if qual and qual > 0 else default_bq
            p_err = 10.0 ** (-bq / 10.0)
            p_corr = 1.0 - p_err
            if allele == 1:
                ll1 += np.log(p_corr * e1[idx] + p_err * (1.0 - e1[idx]))
                ll2 += np.log(p_corr * e2[idx] + p_err * (1.0 - e2[idx]))
            else:
                ll1 += np.log(p_corr * (1.0 - e1[idx]) + p_err * e1[idx])
                ll2 += np.log(p_corr * (1.0 - e2[idx]) + p_err * e2[idx])
        diff = ll2 - ll1
        if diff > 50:
            p1 = 0.0
        elif diff < -50:
            p1 = 1.0
        else:
            p1 = 1.0 / (1.0 + np.exp(diff))
        labels[i] = 1 if rng.random() < p1 else 2
    return labels


def _iterative_init_labels(R, n, read_keys, rng, default_bq=30):
    """Sequential read-label initialization.

    The 50/50 random partition that ``np.random.integers`` produces is
    *symmetric*: ``x1 ≈ x2`` so ``gamma1 ≈ gamma2`` after the first F/B
    and the resampling step has no signal to break the tie.  We address
    this by visiting reads in order during the first sweep and sampling
    each one's label *conditional on the labels of all previously-visited
    reads*.  This breaks symmetry deterministically using the LD
    structure in the data (two reads observing the alt allele at the
    same site are pushed onto the same hap; two reads with discordant
    alleles at one site are pushed apart).

    Here we approximate that online behaviour without a streaming F/B
    by using the *empirical* per-site alt-fraction of each hap
    (``gl_j[1, i] / (gl_j[0, i] + gl_j[1, i])``) as the predictive
    probability.  Reads are processed in order of their first observed
    SNP position so neighbouring reads inform each other.

    Returns
    -------
    labels : ndarray (nReads,) int8
        Initial labels in {1, 2} after one sequential sweep.
    """
    gl1 = np.ones((2, n), dtype=np.float64)
    gl2 = np.ones((2, n), dtype=np.float64)

    def first_pos(i):
        obs = R[read_keys[i]][0]
        return obs[0][0] if obs else -1

    order = sorted(range(len(read_keys)), key=first_pos)
    labels = np.empty(len(read_keys), dtype=np.int8)

    for i_read in order:
        key = read_keys[i_read]
        obs = R[key][0]
        if not obs:
            labels[i_read] = 1 if rng.random() < 0.5 else 2
            continue
        ll1 = 0.0
        ll2 = 0.0
        for idx, allele, qual in obs:
            bq = qual if qual and qual > 0 else default_bq
            p_err = 10.0 ** (-bq / 10.0)
            p_corr = 1.0 - p_err
            tot1 = gl1[0, idx] + gl1[1, idx]
            tot2 = gl2[0, idx] + gl2[1, idx]
            pAlt1 = gl1[1, idx] / tot1 if tot1 > 0 else 0.5
            pAlt2 = gl2[1, idx] / tot2 if tot2 > 0 else 0.5
            if allele == 1:
                ll1 += np.log(p_corr * pAlt1 + p_err * (1.0 - pAlt1) + 1e-300)
                ll2 += np.log(p_corr * pAlt2 + p_err * (1.0 - pAlt2) + 1e-300)
            else:
                ll1 += np.log(p_corr * (1.0 - pAlt1) + p_err * pAlt1 + 1e-300)
                ll2 += np.log(p_corr * (1.0 - pAlt2) + p_err * pAlt2 + 1e-300)
        diff = ll2 - ll1
        if diff > 50:
            p1 = 0.0
        elif diff < -50:
            p1 = 1.0
        else:
            p1 = 1.0 / (1.0 + np.exp(diff))
        new_label = 1 if rng.random() < p1 else 2
        labels[i_read] = new_label
        target = gl1 if new_label == 1 else gl2
        for idx, allele, qual in obs:
            bq = qual if qual and qual > 0 else default_bq
            p_err = 10.0 ** (-bq / 10.0)
            p_corr = 1.0 - p_err
            if allele == 1:
                target[1, idx] *= p_corr
                target[0, idx] *= p_err
            else:
                target[0, idx] *= p_corr
                target[1, idx] *= p_err
            s = target[0, idx] + target[1, idx]
            if s > 0:
                target[0, idx] /= s
                target[1, idx] /= s
    return labels


def impute_diploid_gibbs(R, sigma, emission, nhap=None, n_iter=10,
                         minp=0.01, nthreads=1, use_random_init=False,
                         random_seed=None, phasing_iter=True,
                         dump_prefix=None, iterative_init=True,
                         n_inner_sweeps=2, default_bq=30):
    """Diploid imputation via Gibbs sampling on read labels.

    Per iteration:

    1. Build a ``(2, n)`` per-hap genotype-likelihood array ``gl_j`` from
       the reads currently assigned to that hap, weighting each observation
       by ``1 - 10^(-Q/10)`` (correct call) or ``10^(-Q/10)`` (error).  See
       :func:`_build_gl_per_hap` -- this computes the per-haplotype genotype likelihood.
    2. Run a haploid F/B per hap with the gl-combined emission
       ``gl[0, i] * (1 - panel[k, i]) + gl[1, i] * panel[k, i]`` (see
       :func:`forward_backward_gl`).  This replaces the simple ref/alt
       count formula and properly accounts for per-base sequencing error.
    3. Hard-resample each read's label from the BQ-aware posterior
       (:func:`_resample_read_labels_gl`).  ``n_inner_sweeps - 1`` extra
       resampling passes are run using the same ``p_j`` before the next F/B,
       which amortises the F/B cost and accelerates convergence.

    The initial labelling is **not** a 50/50 random split (that creates a
    symmetric ``x1 ≈ x2`` state from which Gibbs takes many iterations to
    escape).  Instead, when ``iterative_init=True`` (default) reads are
    visited in position order and each one is assigned conditional on the
    labels of all previously-visited reads.  See :func:`_iterative_init_labels`.

    Parameters
    ----------
    n_iter : int
        Total outer F/B iterations.  The first ``n_iter // 3`` are treated
        as burn-in and not averaged.
    iterative_init : bool
        If True (default), use sequential read-label initialization to
        break symmetry; otherwise use a 50/50 random split.
    n_inner_sweeps : int
        Number of read-label Gibbs resampling sweeps performed per F/B
        pass (default 2).  Set to 1 to disable inner sweeps.
    default_bq : int
        Phred quality used for observations with missing/zero BQ.
    See ``impute_diploid`` for the remaining parameters.

    Returns
    -------
    p1, p2 : ndarray (n,)
        Posterior P(alt) marginals averaged over post-burn-in samples.
    emission : ndarray
        The (possibly sub-selected) emission matrix used.
    phased_haps : tuple or None
        Phased calls from ``recast_haps`` when ``phasing_iter`` is True.
    """
    rng = np.random.default_rng(random_seed)
    K_full = emission.shape[0]

    # --- haplotype pre-selection ---
    hap_indices_full = None
    if nhap is not None and isinstance(nhap, int):
        if use_random_init:
            hap_indices = preselect_haplotypes_random(K_full, nhap, seed=random_seed)
            print("Gibbs+random-init: %d randomly selected haplotypes (seed=%s)." %
                  (len(hap_indices), random_seed))
        else:
            hap_indices = preselect_haplotypes(R, emission, nhap, emission.shape[1])
            print("Gibbs: %d pre-selected (overlap-based) haplotypes." % len(hap_indices))
        hap_indices_full = list(hap_indices)
        emission = emission[hap_indices, :]

    n = emission.shape[1]
    is_alt = (emission > 0.5)

    # --- ordered list of reads for label vector ---
    read_keys = list(R.keys())
    nReads = len(read_keys)
    if nReads == 0:
        raise ValueError("No reads in R; cannot run Gibbs diploid imputation.")

    # --- initial read labelling ---
    if iterative_init:
        labels = _iterative_init_labels(R, n, read_keys, rng,
                                        default_bq=default_bq)
        print("Gibbs: iterative sequential init: %d -> hap1, %d -> hap2" %
              ((labels == 1).sum(), (labels == 2).sum()))
    else:
        labels = (rng.integers(0, 2, size=nReads, dtype=np.int8) + 1)
        print("Gibbs: 50/50 random init: %d -> hap1, %d -> hap2" %
              ((labels == 1).sum(), (labels == 2).sum()))

    _dump_hap_indices = (np.array(hap_indices_full, dtype=np.int32)
                         if hap_indices_full is not None
                         else np.arange(emission.shape[0], dtype=np.int32))

    burnin = max(0, n_iter // 3)
    p1_sum = np.zeros(n, dtype=np.float64)
    p2_sum = np.zeros(n, dtype=np.float64)
    n_avg = 0

    gamma1 = None
    gamma2 = None

    for gi in range(n_iter):
        # --- build per-hap gl arrays from current labels (BQ-weighted) ---
        gl1, gl2 = _build_gl_per_hap(R, n, labels, read_keys,
                                     default_bq=default_bq)

        # --- gl-based haploid F/B per haplotype ---
        gamma1 = forward_backward_gl(gl1, sigma, emission,
                                     nthreads=nthreads, minp=1e-6)
        gamma2 = forward_backward_gl(gl2, sigma, emission,
                                     nthreads=nthreads, minp=1e-6)

        # --- per-site posterior expected P(alt) for each hap ---
        denom1 = np.maximum(gamma1.sum(axis=0), 1e-300)
        denom2 = np.maximum(gamma2.sum(axis=0), 1e-300)
        p1_cur = (gamma1 * emission).sum(axis=0) / denom1
        p2_cur = (gamma2 * emission).sum(axis=0) / denom2

        if dump_prefix is not None:
            np.savez_compressed(
                '%s_gibbs_iter%d_hap1.npz' % (dump_prefix, gi + 1),
                gamma=gamma1.astype(np.float32),
                emission=emission.astype(np.uint8),
                sigma=sigma.astype(np.float64),
                labels=labels.astype(np.int8),
                gl=gl1.astype(np.float32),
                hap_indices=_dump_hap_indices,
            )
            np.savez_compressed(
                '%s_gibbs_iter%d_hap2.npz' % (dump_prefix, gi + 1),
                gamma=gamma2.astype(np.float32),
                emission=emission.astype(np.uint8),
                sigma=sigma.astype(np.float64),
                labels=labels.astype(np.int8),
                gl=gl2.astype(np.float32),
                hap_indices=_dump_hap_indices,
            )

        if gi >= burnin:
            p1_sum += p1_cur
            p2_sum += p2_cur
            n_avg += 1

        n_flipped_str = ""
        if gi < n_iter - 1:
            # --- hard resample (Gibbs step), n_inner_sweeps times ---
            prev_labels = labels.copy()
            for _is in range(n_inner_sweeps):
                labels = _resample_read_labels_gl(
                    R, read_keys, p1_cur, p2_cur, rng,
                    default_bq=default_bq)
            n_flipped = int(np.sum(labels != prev_labels))
            n_flipped_str = " flipped=%d" % n_flipped

        print("Gibbs iter %d: hap1_reads=%d hap2_reads=%d "
              "p1_mean=%.4f p2_mean=%.4f%s%s" %
              (gi + 1, (labels == 1).sum(), (labels == 2).sum(),
               p1_cur.mean(), p2_cur.mean(),
               n_flipped_str,
               " (averaging)" if gi >= burnin else " (burn-in)"))

    p1 = p1_sum / max(n_avg, 1)
    p2 = p2_sum / max(n_avg, 1)
    print("Gibbs: averaged %d post-burn-in samples" % n_avg)

    # --- optional phasing pass: one mean-field-style refinement ---
    phased_haps = None
    if phasing_iter and gamma1 is not None and gamma2 is not None:
        print("Gibbs: phasing iteration (warm-start from final Gibbs sample)")
        ph1_alleles = sample_haplotype_path(gamma1, emission)
        eff2_ph = diploid_emission(is_alt, ph1_alleles, minp)
        x_all = RtoX(R, n, label=None)
        gamma2_ph = forward_backward_haploid_double(
            x_all, sigma, eff2_ph, nthreads=nthreads)

        ph2_alleles = sample_haplotype_path(gamma2_ph, emission)
        eff1_ph = diploid_emission(is_alt, ph2_alleles, minp)
        gamma1_ph = forward_backward_haploid_double(
            x_all, sigma, eff1_ph, nthreads=nthreads)

        ph1 = (gamma1_ph * emission).sum(axis=0) / np.maximum(gamma1_ph.sum(axis=0), 1e-300)
        ph2 = (gamma2_ph * emission).sum(axis=0) / np.maximum(gamma2_ph.sum(axis=0), 1e-300)

        gp0 = (1 - ph1) * (1 - ph2)
        gp1 = ph1 * (1 - ph2) + (1 - ph1) * ph2
        gp2 = ph1 * ph2
        hap1_phased, hap2_phased = recast_haps(ph1, ph2, gp0, gp1, gp2)
        phased_haps = (hap1_phased, hap2_phased)
        n_recast = int(np.sum(
            (np.round(ph1) + np.round(ph2)) != (np.round(hap1_phased) + np.round(hap2_phased))
        ))
        print("Phasing: recast_haps corrected %d/%d sites" % (n_recast, n))

    return p1, p2, emission, phased_haps


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
                        windowsize=32, nthreads=1, read_prior=False):
    """Block log-likelihood for the triploid model at *centralsnpi*.

    Evaluates the HMM forward pass in a local window for each of the
    three labels and combines with a label prior.  By default the prior is a
    binomial on the global label counts; when *read_prior* is set it is the sum
    of per-read label log-priors (see :func:`read_label_logprior`), which lets a
    per-read fetal posterior steer the assignment.
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
        if useprior and not read_prior:
            ll += np.log(binom.pmf(n_per_label[label], n_total,
                                   priors[label]) + 1e-308)
    if useprior and read_prior:
        ll += read_prior_loglik(R, ff=ff)
    return ll


def loglikelihood(R, sigma, emission, useprior=True, ff=0.1,
                  nthreads=1, read_prior=False):
    """Full-length triploid log-likelihood (sum over all three labels).

    With *read_prior* the global binomial label-count prior is replaced by the
    sum of per-read label log-priors (see :func:`read_label_logprior`).
    """
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
        if useprior and not read_prior:
            ll += np.log(binom.pmf(n_per_label[label], n_total,
                                   priors[label]) + 1e-308)
    if useprior and read_prior:
        ll += read_prior_loglik(R, ff=ff)
    return ll


def gibbs(R, sigma, emission, nhap=400, ff=0.1, useprior=True,
          verbose=False, maxiterfull=3, maxiterpart=None,
          nthreads=1, read_prior=False):
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
                       nthreads=nthreads, read_prior=read_prior)
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
                                     windowsize=32, nthreads=nthreads,
                                     read_prior=read_prior)

            opta = org + 1 if org < 2 else 0
            optb = org - 1 if org > 0 else 2

            R[r][1] = opta
            tmp1 = loglikelihood_block(R, centralsnpi, sigma, emission,
                                       useprior=useprior, ff=ff,
                                       windowsize=32, nthreads=nthreads,
                                       read_prior=read_prior)

            R[r][1] = optb
            tmp2 = loglikelihood_block(R, centralsnpi, sigma, emission,
                                       useprior=useprior, ff=ff,
                                       windowsize=32, nthreads=nthreads,
                                       read_prior=read_prior)

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
                           read_prior=read_prior,
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
                                chrom, variants, vcffile, sample,
                                phased_haps=None):
    """Write diploid imputation results to VCF.

    Genotype probabilities from two marginal P(alt) vectors:
        GP(0/0) = (1-p1)(1-p2)
        GP(0/1) = p1(1-p2) + (1-p1)p2
        GP(1/1) = p1*p2

    When *phased_haps* is provided (a ``(hap1, hap2)`` tuple from
    ``recast_haps``), the GT field is written as phased (``0|0`` etc.)
    and DS/GP are recomputed from the phased haplotype calls.
    """
    k = emission.shape[0]
    n = emission.shape[1]

    X_all = RtoX(R, n, label=None)
    refaltcnt = emission.sum(axis=0, dtype=np.int64)

    use_phased = phased_haps is not None
    if use_phased:
        hap1_ph, hap2_ph = phased_haps

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
        vcf.write(_get_command_line_header() + "\n")
        vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t%s\n" % sample)

        for pi, (vid, pos, ref, alt) in enumerate(variants):
            if use_phased:
                a1 = hap1_ph[pi]
                a2 = hap2_ph[pi]
                gp0 = (1 - a1) * (1 - a2)
                gp1 = a1 * (1 - a2) + (1 - a1) * a2
                gp2 = a1 * a2
                dosage = gp1 + 2 * gp2
                h1 = int(round(a1))
                h2 = int(round(a2))
                gt_str = "%d|%d" % (h1, h2)
            else:
                pa, pb = p1[pi], p2[pi]
                gp0 = (1 - pa) * (1 - pb)
                gp1 = pa * (1 - pb) + (1 - pa) * pb
                gp2 = pa * pb
                dosage = gp1 + 2 * gp2
                gt_str = ["0/0", "0/1", "1/1"][np.argmax([gp0, gp1, gp2])]

            fij = gp1 + 4 * gp2
            thetaHat = dosage / 2
            denom = 2 * thetaHat * (1 - thetaHat)
            info_score = max(0.0, 1 - ((fij - dosage ** 2) / denom)) if denom > 0 else 1.0

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
              chrom, variants, vcffile, sample, minp=0.01):
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
        vcf.write(_get_command_line_header() + "\n")
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

            fij_m = gtm1 + 4 * gtm2
            fij_f = gtf1 + 4 * gtf2
            denom_m = 2 * thetaHat_m * (1 - thetaHat_m)
            denom_f = 2 * thetaHat_f * (1 - thetaHat_f)
            info_m = max(0.0, 1 - ((fij_m - eij_m ** 2) / denom_m)) if denom_m > 0 else 1.0
            info_f = max(0.0, 1 - ((fij_f - eij_f ** 2) / denom_f)) if denom_f > 0 else 1.0

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

def _load_genotypes_worker(args):
    """Worker for build_reference pool - must be top-level for pickling."""
    idx, fp, n, positions, filterflag, cramref = args
    return load_genotypes_from_file(idx, fp, n, positions,
                                    filterflag=filterflag, cramref=cramref)


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
            # Try alternate chromosome naming if primary fails
            chroms_to_try = [chrom]
            if chrom.startswith('chr'):
                chroms_to_try.append(chrom[3:])  # Try without 'chr'
            else:
                chroms_to_try.append('chr' + chrom)  # Try with 'chr'

            pileup_success = False
            for chrom_try in chroms_to_try:
                try:
                    for pos, refa, alt in positions[chrom]:
                        rc = ac = 0
                        for pc in bamfile.pileup(chrom_try, pos - 1, pos, truncate=True,
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
                    pileup_success = True
                    if chrom_try != chrom:
                        logging.info("Chromosome naming mismatch: target uses '%s', CRAM uses '%s' (%s)" % (chrom, chrom_try, ifile_idx))
                    break  # Success, don't try other names
                except ValueError as e:
                    # Chromosome not found, try next name variant
                    continue

            if not pileup_success:
                logging.warning("Chromosome %s not found in %s (tried: %s)" % (chrom, filepath, chroms_to_try))
                # Skip positions for this chromosome
                i += len(positions[chrom])

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
            try:
                chriterator = iter(vcffile.fetch(chrom))
                rec = next(chriterator)
            except StopIteration:
                # Chromosome not in VCF, skip all positions for this chrom
                i += len(positions[chrom])
                continue
            for pos, refa, alt in positions[chrom]:
                try:
                    while rec.pos < pos:
                        rec = next(chriterator)
                except StopIteration:
                    # No more records for this chrom, skip remaining positions
                    i += 1
                    continue
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

    # Prepare arguments for worker - must be picklable
    worker_args = [(idx, fp, n, positions, filterflag, cramref)
                   for idx, fp in enumerate(expanded)]

    with multiprocessing.Pool(processes=nproc) as pool:
        X = pool.map(_load_genotypes_worker, worker_args)

    X = [x for x in X if x is not None]
    X = np.concatenate(X, axis=0)

    outpath = '%s.reference.pickle' % outputprefix
    pickle.dump((X, sigma, positions), open(outpath, 'wb'))
    logging.info("Wrote reference to: %s" % outpath)
    return outpath


def train_model(reference_pickle, k=4, maxiter=40, outputprefix=None,
                initmodel=None, ngen=100, nproc=1, output_format='pickle'):
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
        Output model name prefix (extension auto-added based on format).
    initmodel : str or None
        Path to a previously trained model for warm-starting.
    ngen : int
        Generations since founding.
    nproc : int
        Thread count for the C extension.
    output_format : str
        Output format: 'pickle' (default) or 'vcf'. VCF format encodes
        emission/transition/start parameters as ancient haplotype samples
        that can be used directly with 'impute run'.

    Returns
    -------
    str
        Path to the trained model file.
    """
    if outputprefix is None:
        outputprefix = "%s_k%d_iter%d" % (
            os.path.basename(reference_pickle).replace(".pickle", "").replace(".gz", ""),
            k, maxiter)

    X, sigma, positions = pickle.load(open(reference_pickle, 'rb'))

    # Ensure arrays own their data (not views) and are C-contiguous for C extension
    X = np.ascontiguousarray(X)
    sigma = np.ascontiguousarray(sigma)

    N = X.shape[0]
    n = X.shape[1]

    logging.info("Reference contains %d samples with %d genotypes" % (N, n))

    if initmodel is not None:
        # Detect if initmodel is VCF (from previous imputeref run) or pickle
        if initmodel.endswith('.vcf') or initmodel.endswith('.vcf.gz'):
            logging.info("Loading warm-start model from VCF: %s" % initmodel)
            sigma_init, emission, variants = _loadref_vcf_trained(initmodel)
            # Infer k from emission shape
            k = emission.shape[0]
            n_loaded = emission.shape[1]
            if n_loaded != n:
                logging.warning("Warm-start VCF has %d variants, reference has %d. Using reference dimensions." % (n_loaded, n))
            # Initialize uniform start and transition probs (will be updated by EM)
            start = np.ascontiguousarray(np.ones(k) / k)
            a = np.ascontiguousarray(np.ones((k, n - 1)) / k)
            # Use sigma from reference, not from VCF (positions may differ)
            logging.info("Warm-start: k=%d, emission shape=%s" % (k, emission.shape))
        else:
            logging.info("Loading warm-start model from pickle: %s" % initmodel)
            start, a, sigma, emission, positions, k, allelecounts = \
                pickle.load(open(initmodel, 'rb'))
            # Ensure contiguous arrays
            start = np.ascontiguousarray(start)
            a = np.ascontiguousarray(a)
            emission = np.ascontiguousarray(emission)
    else:
        a = np.ones((k, n - 1))
        a = np.ascontiguousarray(a / a.sum(axis=0))
        start = np.ascontiguousarray(np.ones(k) / k)

        # try:
        #     import rpy2.robjects as robjects
        #     emission = np.array(np.array(robjects.r("""
        #     set.seed(42)
        #     x <- array(runif(%d * %d))
        #     """ % (n, k))).reshape(k, n, order='F').tolist())
        #     emission = np.ascontiguousarray(emission)
        # except ImportError:
        #     logging.error(
        #         "Training init requires 'rpy2' (used to reproduce the original "
        #         "seeded R RNG). Install with `pip install rpy2`, or pass --initmodel."
        #     )
        emission = np.ascontiguousarray(np.random.rand(k, n))

    logging.info("Starting HMM fit with maxiter=%d, nthreads=%d, ngen=%d" % (maxiter, nproc, ngen))
    result = imputeff_hmm.fit(
        X, start, a, sigma, emission,
        maxiter=maxiter, nthreads=nproc, ngen=ngen)
    start, a, sigma, emission, C = result
    logging.info("HMM fit completed, writing model to disk")

    if output_format.lower() == 'vcf':
        outpath = '%s.model.vcf.gz' % outputprefix
        _write_trained_model_vcf(outpath, start, a, sigma, emission, positions, k)
    else:
        outpath = '%s.model.pickle' % outputprefix
        pickle.dump((start, a, sigma, emission, positions, k,
                     X.sum(axis=0)), open(outpath, 'wb'))
        logging.info("Wrote trained model to: %s" % outpath)
    return outpath


def _write_trained_model_vcf(outpath, start, a, sigma, emission, positions, k):
    """Write trained HMM parameters as a VCF with ancient haplotypes.

    The VCF encodes:
    - Emission probabilities in PL (Phred-scaled likelihood) fields
      where each "sample" represents one HMM state/haplotype
    - Transition probabilities in INFO/IMP_TP (comma-separated per state)
    - Recombination probabilities in INFO/IMP_RP
    - Start probabilities in header ##IMP_START

    Parameters
    ----------
    outpath : str
        Output VCF path (should end in .vcf or .vcf.gz)
    start : ndarray, shape (k,)
        Initial state probabilities
    a : ndarray, shape (k, n-1)
        Transition probabilities between states
    sigma : ndarray, shape (n-1,)
        Recombination probabilities
    emission : ndarray, shape (k, n)
        Emission probabilities (P(alt|state))
    positions : dict
        {chrom: [(pos, ref, alt), ...]} from reference pickle
    k : int
        Number of HMM states
    """
    import gzip

    # Determine if we need gzip output
    use_gzip = outpath.endswith('.gz')

    # Build header
    header_lines = [
        "##fileformat=VCFv4.2",
        "##source=cfstats impute train",
        "##IMP_K=%d" % k,
        "##IMP_START=%s" % ','.join(f"{s:.6f}" for s in start),
        '##INFO=<ID=IMP_TP,Number=%d,Type=Float,Description="Transition probabilities to this variant for each state">' % k,
        '##INFO=<ID=IMP_RP,Number=1,Type=Float,Description="Recombination probability at this position">',
        '##FORMAT=<ID=PL,Number=3,Type=Integer,Description="Phred-scaled likelihoods for RR,RA,AA genotypes encoding emission P(alt|state)">',
    ]

    # Add sample columns for each HMM state (ancient haplotypes)
    samples = [f"HAP{i}" for i in range(k)]

    # Build CHROM list from positions dict
    chroms = sorted(positions.keys())

    # Flatten all positions into sorted order for VCF
    all_sites = []
    for chrom in chroms:
        for i, (pos, ref, alt) in enumerate(positions[chrom]):
            all_sites.append((chrom, pos, ref, alt, i))
    all_sites.sort(key=lambda x: (x[0], x[1]))

    n = len(all_sites)

    # Open output file
    if use_gzip:
        f = gzip.open(outpath, 'wt')
    else:
        f = open(outpath, 'w')

    # Write header
    for line in header_lines:
        f.write(line + '\n')

    # CHROM POS ID REF ALT QUAL FILTER INFO FORMAT samples...
    f.write('#' + '\t'.join(['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL',
                              'FILTER', 'INFO', 'FORMAT'] + samples) + '\n')

    # Write variant records
    for i, (chrom, pos, ref, alt, _) in enumerate(all_sites):
        # Build INFO field
        info_parts = []

        # Transition probabilities (for position i, from previous)
        # For position 0, no transition; for others, use a[:, i-1]
        if i > 0:
            tp_values = a[:, i-1]
            info_parts.append("IMP_TP=%s" % ','.join(f"{p:.6f}" for p in tp_values))
            info_parts.append("IMP_RP=%.6f" % sigma[i-1])

        info_str = ';'.join(info_parts) if info_parts else '.'

        # Build genotype/PL fields for each state
        # PL encoding: emission prob for alt allele
        # PL = [-10*log10(P), 0, -10*log10(1-P)] where P is emission prob
        # This creates: RR unlikely if P high, RA neutral, AA likely if P high
        pl_fields = []
        for j in range(k):
            p_alt = emission[j, i]
            # Clip to avoid log(0)
            p_alt = max(min(p_alt, 0.999999), 0.000001)
            p_ref = 1.0 - p_alt

            # Phred scale: -10 * log10(P)
            pl_rr = int(round(-10 * np.log10(p_ref)))  # P(REF)
            pl_ra = 0  # Middle ground
            pl_aa = int(round(-10 * np.log10(p_alt)))  # P(ALT)

            pl_fields.append(f"{pl_rr},{pl_ra},{pl_aa}")

        # Write record
        qual = '.'
        filter_str = 'PASS'
        format_str = 'PL'

        row = [chrom, str(pos), '.', ref, alt, qual, filter_str,
               info_str, format_str] + pl_fields
        f.write('\t'.join(row) + '\n')

    f.close()
    logging.info("Wrote trained model VCF to: %s" % outpath)


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


def _isize_fetal_posterior(tl, ff, mat_mean=165.0, fet_mean=145.0, sd=25.0):
    """Per-read fetal posterior from insert size under a 2-Gaussian model.

    Models maternal and fetal fragment lengths as Gaussians with means
    *mat_mean* / *fet_mean* and shared *sd*, mixed at fetal fraction *ff*::

        w = ff*N(tl; fet) / (ff*N(tl; fet) + (1-ff)*N(tl; mat))

    This mirrors the isize-only mixture used by the read classifier, so the
    synthetic ``XF`` posteriors behave like the real ones.
    """
    a = abs(int(tl))
    lf = np.exp(-0.5 * ((a - fet_mean) / sd) ** 2)
    lm = np.exp(-0.5 * ((a - mat_mean) / sd) ** 2)
    num = ff * lf
    return float(num / (num + (1.0 - ff) * lm + 1e-300))


def simR(imH, nimH, ipH, n, totreads=1000, ff=0.1, read_prior=False,
         mat_mean=165.0, fet_mean=145.0, isize_sd=25.0):
    """Simulate reads from three haplotypes for testing.

    Each read is drawn from one of the three haplotypes (IMH / NIMH / IPH) with
    a fragment length sampled from the maternal or fetal insert-size Gaussian.
    When *read_prior* is set, an insert-size fetal posterior ``w_i`` is attached
    as the 5th element of each read entry (mirroring the ``XF`` tag), so the
    per-read-prior Gibbs path can be exercised on synthetic data.
    """
    rid = 0
    R = {}

    def _make(ri, hap_allele, truth, fragment_mean):
        nonlocal rid
        s = random.random()
        tl = int(round(np.random.normal(fragment_mean, isize_sd)))
        entry = [
            [(ri, int(hap_allele), 30)],
            0 if s < 0.5 else 1 if s < 1 - (ff / 2) else 2,
            truth,
            tl,
        ]
        if read_prior:
            entry.append(_isize_fetal_posterior(tl, ff, mat_mean, fet_mean,
                                                 isize_sd))
        R[(ri, rid)] = entry
        rid += 1

    # IMH reads are shared maternal/fetal: draw fragment length from whichever
    # genome the molecule came from (fetal with probability ff).
    for ri in np.random.randint(0, n, int(0.5 * totreads)):
        is_fetal = random.random() < ff
        _make(ri, imH[ri], 0, fet_mean if is_fetal else mat_mean)

    for ri in np.random.randint(0, n, int((0.5 - (ff / 2)) * totreads)):
        _make(ri, nimH[ri], 1, mat_mean)

    for ri in np.random.randint(0, n, int((ff / 2) * totreads)):
        _make(ri, ipH[ri], 2, fet_mean)

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
