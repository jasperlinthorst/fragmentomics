"""Population-reference HMM imputation of maternal/fetal genotypes and fetal
fraction estimation from low-coverage NIPT sequencing.

Ported from the standalone `imputeff` package. The optimized HMM forward/
backward and Baum-Welch fit routines live in the compiled extension
``cfstats.impute._hmm``.
"""

from cfstats.impute import core  # noqa: F401
