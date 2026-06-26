"""Command-line entry points for ``cfstats impute``.

Supports three modes:
    - **run** (default): impute genotypes from a BAM/CRAM using a reference panel
    - **build-reference**: preprocess BAM/VCF files into a reference pickle
    - **train**: EM-fit an HMM on a reference pickle

Modes ``build-reference`` and ``train`` are intended for the case where
no phased population reference panel is available, but a collection of
BAM/CRAM/VCF files is.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cfstats.impute import core as imputefflib


# ===================================================================
# impute run  (main entry point)
# ===================================================================

def impute_ref(args):
    """Run genotype imputation (diploid or triploid/NIPT mode)."""
    nthreads = getattr(args, 'nthreads', None) or getattr(args, 'nproc', 1) or 1

    # --- load genetic map (optional) ------------------------------------
    genetic_map = None
    genetic_map_file = getattr(args, 'genetic_map', None)
    if genetic_map_file is not None:
        print("Loading genetic map: %s" % genetic_map_file)
        genetic_map = imputefflib.load_genetic_map(genetic_map_file)
        print("Genetic map loaded: %d entries" % len(genetic_map[0]))

    # --- load reference -------------------------------------------------
    if args.reference.endswith(('.vcf', '.vcf.gz', '.bcf')):
        sigma, emission, variants = imputefflib.loadref_vcf(
            args.reference,
            contig=args.chrom,
            start=args.start,
            stop=args.stop,
            nGen=args.ngen, avgr=args.avgr, minp=args.minp,
            genetic_map=genetic_map)
    else:
        hap = args.reference + ".hap.gz"
        legend = args.reference + ".legend.gz"
        if os.path.exists(hap) and os.path.exists(legend):
            sigma, emission, variants = imputefflib.loadref_haplegend(
                hap, legend, nGen=args.ngen, avgr=args.avgr, minp=args.minp,
                start=args.start, stop=args.stop, genetic_map=genetic_map)
        else:
            logging.error("Unknown reference file, expecting vcf/bcf/vcf.gz or "
                          "hap.gz/legend.gz prefix: %s" % args.reference)
            sys.exit(1)

    # --- read input -----------------------------------------------------
    simulation = False
    if args.input.endswith(('.sam', '.bam', '.cram')):
        R = imputefflib.getR(args.input, args.chrom, variants,
                             ref=args.cramref,
                             addchr=args.addchr, rmchr=args.rmchr,
                             min_base_quality=args.min_base_quality,
                             read_prior=getattr(args, 'read_prior', False),
                             read_prior_tag=getattr(args, 'read_prior_tag', 'XF'))
        print("Number of reads: %d" % len(R))
        print("Number of observed alleles: %d" % sum(len(R[r][0]) for r in R))
        if args.maxnreads is not None and len(R) > args.maxnreads:
            keep = set(random.sample(list(R.keys()), args.maxnreads))
            R = {k: R[k] for k in keep}
    elif args.input.count('/') == 2:
        imhi, nimhi, iphi = [int(i) for i in args.input.split('/')]
        emission = np.unique(emission, axis=0)
        np.random.shuffle(emission)
        emission = emission[:100]
        mh = max([imhi, nimhi, iphi])
        if mh >= emission.shape[0]:
            logging.error(
                "Specified haplotype index (%d) for simulation is larger "
                "than the number of unique reference haplotypes." % mh)
            sys.exit(1)
        imh = emission[imhi, :].round().astype(np.uint8)
        nimh = emission[nimhi, :].round().astype(np.uint8)
        iph = emission[iphi, :].round().astype(np.uint8)
        if args.maxnreads is None:
            logging.error("For simulating specify --maxnreads.")
            sys.exit(1)
        R = imputefflib.simR(imh, nimh, iph, emission.shape[1],
                             totreads=args.maxnreads, ff=args.ff)
        simulation = True
    else:
        logging.error("Unknown input file, expecting sam/bam/cram: %s" % args.input)
        sys.exit(1)

    # --- imputation mode -----------------------------------------------
    nhap = getattr(args, 'nhap', None)
    diploid = getattr(args, 'diploid', False)

    if diploid:
        n_iter = getattr(args, 'gibbs_fulliter', 3)
        if n_iter is None:
            n_iter = 3

        random_init = getattr(args, 'random_init', False)
        knew = getattr(args, 'knew', None)
        phasing_iter = not getattr(args, 'nophase', False)
        use_gibbs = getattr(args, 'gibbs', False)

        dump_prefix = getattr(args, 'dump', None)

        if use_gibbs:
            p1, p2, emission, phased_haps = imputefflib.impute_diploid_gibbs(
                R, sigma, emission,
                nhap=nhap, n_iter=n_iter,
                minp=args.minp, nthreads=nthreads,
                use_random_init=random_init,
                random_seed=args.seed, phasing_iter=phasing_iter,
                dump_prefix=dump_prefix)
        else:
            p1, p2, emission, phased_haps = imputefflib.impute_diploid(
                R, sigma, emission,
                nhap=nhap, n_iter=n_iter,
                minp=args.minp, nthreads=nthreads,
                use_random_init=random_init, knew=knew,
                random_seed=args.seed, phasing_iter=phasing_iter,
                dump_prefix=dump_prefix)

        if not simulation:
            sample = args.sample or os.path.basename(args.input).split('.')[0]
            imputefflib.outputvcf_diploid_marginals(
                p1, p2, R, emission,
                args.chrom, variants, args.output, sample,
                phased_haps=phased_haps)
        else:
            dosage = p1 + p2
            gt = np.round(dosage).astype(np.int8)
            print("Error rate imputing (diploid):",
                  np.abs(gt - imh).sum() / len(gt))
    else:
        # --- triploid (NIPT) mode: Gibbs read labelling ---
        read_prior = getattr(args, 'read_prior', False)
        imputefflib.gibbs(R, sigma, emission,
                          ff=args.ff, nhap=nhap, useprior=True, verbose=False,
                          maxiterfull=args.gibbs_fulliter,
                          maxiterpart=args.gibbs_partiter,
                          nthreads=nthreads, read_prior=read_prior)

        imputefflib.gibbs(R, sigma, emission,
                          ff=args.ff, nhap=nhap, useprior=False, verbose=False,
                          maxiterfull=args.gibbs_fulliter,
                          maxiterpart=args.gibbs_partiter,
                          nthreads=nthreads, read_prior=read_prior)

        # Subset emission for final pass if nhap was used
        if nhap is not None and isinstance(nhap, int):
            hap_indices = imputefflib.preselect_haplotypes(
                R, emission, nhap, emission.shape[1])
            emission = emission[hap_indices, :]
            print("Using %d pre-selected haplotypes for final pass." % len(hap_indices))

        k = emission.shape[0]
        n = emission.shape[1]

        # Determine read-label counts and ordering
        readcnt = np.zeros(3)
        for r in R:
            readcnt[R[r][1]] += 1
        order = np.argsort(readcnt)

        IPHreadcnt, NIMHreadcnt, IMHreadcnt = readcnt[order]
        print("Reads assigned to IMH:", IMHreadcnt,
              "NIMH:", NIMHreadcnt, "IPH:", IPHreadcnt)
        if (NIMHreadcnt + IPHreadcnt) > 0:
            print("MLE fetal fraction: %.4f" %
                  (IPHreadcnt / float(NIMHreadcnt + IPHreadcnt)))

        x1 = imputefflib.RtoX(R, n, order[2])  # IMH
        x2 = imputefflib.RtoX(R, n, order[1])  # NIMH
        x3 = imputefflib.RtoX(R, n, order[0])  # IPH

        # Final forward-backward per label (GIL-free, can run in parallel)
        if nthreads > 1:
            with ThreadPoolExecutor(max_workers=3) as pool:
                gammaIMH, gammaNIMH, gammaIPH = list(pool.map(
                    lambda x: imputefflib.forward_backward_haploid(
                        x, sigma, emission, nthreads=nthreads),
                    (x1, x2, x3)))
        else:
            gammaIMH = imputefflib.forward_backward_haploid(
                x1, sigma, emission, nthreads=nthreads)
            gammaNIMH = imputefflib.forward_backward_haploid(
                x2, sigma, emission, nthreads=nthreads)
            gammaIPH = imputefflib.forward_backward_haploid(
                x3, sigma, emission, nthreads=nthreads)

        if not simulation:
            sample = args.sample or os.path.basename(args.input).split('.')[0]
            imputefflib.outputvcf(
                gammaIMH, gammaNIMH, gammaIPH, R, emission,
                args.chrom, variants, args.output, sample)
        else:
            ed = np.abs(emission - args.minp)
            palt_fn = lambda g: (g * ed).sum(axis=0) / ((g * ed).sum(axis=0) + (g * (1 - ed)).sum(axis=0))
            err_fn = lambda h, h_: np.abs(h - h_).sum() / len(h)

            imh_ = palt_fn(gammaIMH).round().astype(np.int8)
            nimh_ = palt_fn(gammaNIMH).round().astype(np.int8)
            iph_ = palt_fn(gammaIPH).round().astype(np.int8)
            print("Error rate imputing IMH:", err_fn(imh_, imh))
            print("Error rate imputing NIMH:", err_fn(nimh_, nimh))
            print("Error rate imputing IPH:", err_fn(iph_, iph))


# ===================================================================
# build-reference
# ===================================================================

def build_reference(args):
    """CLI wrapper for core.build_reference."""
    imputefflib.build_reference(
        targetfile=args.targetfile,
        ifiles=args.ifiles,
        outputprefix=getattr(args, 'outputprefix', None) or "imputeff",
        region=getattr(args, 'region', None),
        maxvar=getattr(args, 'maxvar', int(10e6)),
        addchr=getattr(args, 'addchr', False),
        rmchr=getattr(args, 'rmchr', False),
        filterflag=getattr(args, 'filterflag', 3840),
        nproc=getattr(args, 'nproc', 1),
        cramref=getattr(args, 'cramref', None),
    )


# ===================================================================
# train
# ===================================================================

def train(args):
    """CLI wrapper for core.train_model."""
    outpath = imputefflib.train_model(
        reference_pickle=args.reference_pickle,
        k=getattr(args, 'k', 4),
        maxiter=getattr(args, 'maxiter', 40),
        outputprefix=getattr(args, 'outputprefix', None),
        initmodel=getattr(args, 'initmodel', None),
        ngen=getattr(args, 'ngen', 100),
        nproc=getattr(args, 'nproc', 1),
        output_format=getattr(args, 'output_format', 'pickle'),
    )
    print("Trained model written to: %s" % outpath)

    if getattr(args, 'interactive', False):
        from matplotlib import pyplot as plt
        import pickle
        # Only show interactive plot for pickle format
        if outpath.endswith('.pickle'):
            data = pickle.load(open(outpath, 'rb'))
            plt.imshow(data[3], aspect='auto')
            plt.colorbar()
            plt.xlabel("variants")
            plt.ylabel("states")
            plt.show()
        else:
            print("Interactive mode only supported for pickle format")


# ===================================================================
# imputeref: build reference and train in one step
# ===================================================================

def imputeref(args):
    """CLI wrapper combining build_reference and train_model.

    Builds a reference panel from BAM/VCF files, then trains an HMM model
    on the resulting allele observation matrix. Supports warm-starting from
    a previously trained model VCF.
    """
    import tempfile
    import os

    # Generate output prefix if not provided
    outputprefix = getattr(args, 'outputprefix', None)
    if outputprefix is None:
        outputprefix = "imputeref_%s_k%d" % (
            os.path.basename(args.targetfile).replace('.vcf', '').replace('.gz', ''),
            getattr(args, 'k', 4)
        )

    # Step 1: Build reference pickle
    logging.info("=== Step 1: Building reference panel ===")
    ref_pickle = imputefflib.build_reference(
        targetfile=args.targetfile,
        ifiles=args.ifiles,
        maxvar=getattr(args, 'maxvar', int(10e6)),
        region=getattr(args, 'region', None),
        outputprefix=outputprefix + ".tmp",
        addchr=getattr(args, 'addchr', False),
        rmchr=getattr(args, 'rmchr', False),
        filterflag=getattr(args, 'filterflag', 3840),
        cramref=getattr(args, 'cramref', None),
        nproc=getattr(args, 'nproc', 1),
    )
    logging.info("Reference built: %s" % ref_pickle)

    # Step 2: Train HMM model
    logging.info("=== Step 2: Training HMM model ===")

    # Handle warm-start from VCF
    warm_start = getattr(args, 'warm_start', None)
    if warm_start is not None:
        logging.info("Warm-starting from: %s" % warm_start)

    model_vcf = imputefflib.train_model(
        reference_pickle=ref_pickle,
        k=getattr(args, 'k', 4),
        maxiter=getattr(args, 'maxiter', 40),
        outputprefix=outputprefix,
        initmodel=warm_start,  # Can be VCF from previous run
        ngen=getattr(args, 'ngen', 100),
        nproc=getattr(args, 'nproc', 1),
        output_format='vcf',  # Always output VCF for imputeref
    )

    # Clean up temporary reference pickle
    try:
        os.unlink(ref_pickle)
        logging.info("Cleaned up temporary: %s" % ref_pickle)
    except OSError:
        pass

    print("Trained model written to: %s" % model_vcf)
    logging.info("Done. Use this model with: cfstats impute %s <input> <chrom>" % model_vcf)
