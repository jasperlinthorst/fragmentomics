/*
 * Hidden Markov Model (HMM) Forward-Backward Implementation
 *
 * This module implements the core Li & Stephens haplotype model for genotype
 * imputation. The model treats each target haplotype as a mosaic of K reference
 * haplotypes, where transitions between states follow a coalescent process with
 * recombination.
 *
 * Core Algorithm: Forward-Backward (FB) for HMMs
 * ------------------------------------------------
 * The FB algorithm computes posterior probabilities P(state | data) by:
 * 1. Forward pass:  alpha[t,s] = P(obs_1..t, state_s @ t)
 * 2. Backward pass: beta[t,s]  = P(obs_t+1..T | state_s @ t)
 * 3. Combine:      gamma[t,s] = alpha[t,s] * beta[t,s] / P(data)
 *
 * Array Layouts and Variable Naming
 * ----------------------------------
 * k (int): Number of hidden states (reference haplotypes or latent clusters)
 * n (int): Number of genetic variants (sites, SNPs)
 * N (int): Number of reference samples (for training)
 *
 * g (uint8_t*, shape k*n): Emission matrix. g[i*n+t] is allele of reference
 *     haplotype i at position t (0=ref, 1=alt). Row-major: [hap0_site0,
 *     hap0_site1, ..., hap1_site0, hap1_site1, ...].
 *
 * y (uint8_t*, shape 2*n): Observed diploid genotypes. y[t*2+0] = ref count,
 *     y[t*2+1] = alt count at position t. For reads: [0,1] or [1,0] for single
 *     observations, [0,2] or [2,0] for homozygous, [1,1] for het.
 *
 * s (double*, shape n-1): Recombination probabilities. s[t] = P(no recombination
 *     between sites t and t+1). Under Li-Stephens: s[t] = exp(-4*Ne*r*d[t])
 *     where d[t] is genetic distance in Morgans.
 *
 * a (double*, shape k*n-1): Transition matrix. a[i*n+t] = P(state=i @ t+1 |
 *     state @ t). Actually encodes: P(state=i) * (1-s[t]) + s[t] * P(colescence).
 *     Columns sum to 1. For Li-Stephens: a[i,t] ~ 1/k * (1-s[t]) + s[t] * delta.
 *
 * start (double*, shape k): Initial state probabilities P(state @ position 0).
 *     Usually uniform 1/k.
 *
 * alpha (double*, shape k*n): Forward probabilities. Stored row-major:
 *     alpha[i*n+t] = P(obs_1..t, state=i @ t). Computed left-to-right.
 *
 * beta (double*, shape k*n): Backward probabilities. beta[i*n+t] =
 *     P(obs_t+1..n | state=i @ t). Computed right-to-left.
 *
 * c (double*, shape n): Scaling coefficients for log-likelihood. c[t] =
 *     sum_i(alpha[i,t]). The log-likelihood is sum(log(c[t])). This prevents
 *     underflow in long sequences.
 *
 * C_ (double*, shape k*k*n): Pairwise marginal probabilities for EM training.
 *     C_[(i*k+j)*n+t] = P(state=i @ t, state=j @ t+1 | data). Used to
 *     compute expected transition counts in Baum-Welch.
 *
 * Parallelization Strategy
 * -------------------------
 * OpenMP is used to parallelize the inner loop over states (k) in both forward
 * and backward passes. The algorithm releases the Python GIL for true parallelism.
 * Each thread handles a subset of states, with reductions for summations.
 *
 * Two Precision Modes
 * --------------------
 * Standard (Haploid) mode: Uses uint8 emission g[i,t] in {0,1}. Memory efficient
 *   for large reference panels (1000s of haplotypes).
 *
 * Double precision mode: Uses float64 emission g[i,t] in [0,1]. Required for
 *   diploid mean-field inference where emissions encode expected contributions:
 *   g[i,t] = 0.5 * P(alt|hap_i,t) + 0.5 * P(alt|sampled_other_hap,t).
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>

/* numpy 2.0 compatibility: these macros now require PyArrayObject* instead of PyObject* */
#define NPY_ARRAY_DATA(arr) PyArray_DATA((PyArrayObject*)(arr))
#define NPY_ARRAY_DIMS(arr) PyArray_DIMS((PyArrayObject*)(arr))
#define NPY_ARRAY_FILLWBYTE(arr, val) PyArray_FILLWBYTE((PyArrayObject*)(arr), (val))
#define NPY_ARRAY_NDIM(arr) PyArray_NDIM((PyArrayObject*)(arr))

#ifdef _OPENMP
# include <omp.h>
#endif

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <float.h>

#undef NDEBUG
#include <assert.h>

// #define logadd(X, Y) ((X) > (Y) ? X+log(1+exp((Y)-(X))) : Y+log(1+exp((X)-(Y))))
// #define logsub(X, Y) ((X) > (Y) ? X+log(1-exp((Y)-(X))) : Y+log(1-exp((X)-(Y))))

/*
 * Compute emission probabilities for diploid genotype y at position t.
 *
 * This computes P(y[t] | state=(i,j)) where state is the pair of maternal
 * and paternal haplotypes. Under the model, each read comes from either
 * haplotype with equal probability 0.5, so:
 *
 *   P(y | i,j) = 0.5 * P(y | i) + 0.5 * P(y | j)
 *
 * For a single observation y=[y_ref, y_alt], P(y | hap_i) follows binomial:
 *   P(y | hap_i) = (1-g[i,t])^y_ref * g[i,t]^y_alt
 *
 * Parameters:
 *   g: emission matrix (k x n) float64, g[i*n+t] = P(alt | state=i @ t)
 *   y: observations (2 x n) uint8, [ref_count, alt_count] at each site
 *   k: number of states
 *   n: number of sites
 *   t: current position (0-indexed)
 *   e: output buffer (k*k) for pairwise emission probs, e[i*k+j] = P(y|i,j)
 */
void emission(double *g, uint8_t *y, int k, int n, int t, double* e) {
    double pobs;
    int i,j;
    for(i=0; i<k; i++){
        for(j=0; j<k; j++){
            /* P(y|i) = (1-g[i,t])^ref * g[i,t]^alt for haplotype i */
            /* P(y|j) = (1-g[j,t])^ref * g[j,t]^alt for haplotype j */
            /* P(y|i,j) = 0.5*P(y|i) + 0.5*P(y|j) for diploid observation */
            pobs= 0.5*( pow(1-g[i*n+t],y[t*2]) * pow(g[i*n+t],y[t*2+1]) ) + \
                  0.5*( pow(1-g[j*n+t],y[t*2]) * pow(g[j*n+t],y[t*2+1]) );
            e[(i*k)+j]=pobs;
        }
    }
}

/*
 * Backward pass for haploid HMM with uint8 emissions.
 *
 * Computes beta[i,t] = P(obs_{t+1}..obs_{n-1} | state=i @ t) recursively:
 *   beta[t] = sum_{i'} P(obs_{t+1} | i') * trans[i->i'] * beta[t+1, i']
 *
 * Under Li-Stephens, transition is:
 *   P(i->i') = sigma[t] * delta(i=i') + (1-sigma[t]) * (1/k)
 * where sigma[t] = P(no recombination between t and t+1).
 *
 * The algorithm incorporates emission probabilities directly using:
 *   e[i] = P(obs_{t+1} | hap_i) = (1-g[i,t+1])^ref * g[i,t+1]^alt
 *
 * Parameters:
 *   k: number of states (reference haplotypes)
 *   n: number of variants
 *   y: observations (2 x n), y[t*2] = ref count, y[t*2+1] = alt count
 *   s: recombination probabilities (n-1), s[t] = P(no recombination t->t+1)
 *   g: emission matrix (k x n) uint8, g[i*n+t] = allele (0/1)
 *   beta: output backward probabilities (k x n), beta[i*n+t] = P(data_{t+1}..|i@t)
 *   c: scaling coefficients (n), for log-likelihood computation
 *   scaleto: scaling mode (0=none, >0=scale to this value)
 *   nthreads: OpenMP thread count
 */
void _backwardHaploid(int k, int n, uint8_t *y, double *s, uint8_t *g, double *beta, double *c, int scaleto, int nthreads, double *alphaMat, double minp) {
    int i;
    double *eb=(double *) malloc(k*sizeof(double));
    int t=n-1;

    if (scaleto>0){
        c[t]=(double)scaleto;
    }

    for(i=0; i<k; i++){
        beta[(i*n)+(t)] = 1.0;

        if (scaleto!=0){
            beta[(i*n)+(t)] *= c[t];
        }
    }

    double colsum;

    for(t=n-2; t>=0; t--){
        colsum=0.0;
        double cmax=0.0;

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) reduction(+:colsum)
        #endif
        for(i=0; i<k; i++){
            double e=pow(1- fabs(((double)g[i*n+(t+1)]-minp)),y[(t+1)*2]) * pow((double)fabs(((double)g[i*n+(t+1)])-minp),y[(t+1)*2+1]) ;
            eb[i] = beta[(i*n)+(t+1)] * e;
            colsum+= eb[i];
        }

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) reduction(max:cmax)
        #endif
        for(i=0; i<k; i++){
            double am_i = (alphaMat != NULL) ? alphaMat[i*(n-1)+t] : (1.0/k);
            double b = (eb[i] * s[t]) + (colsum * (1.0-s[t]) * am_i);

            beta[(i*n)+(t)] = b;

            if (scaleto>0){
                if (b>cmax) cmax=b;
            }
        }

        //determine scaling factor
        if (scaleto!=0){
            if (scaleto>0){
                c[t]=((double)scaleto)/cmax;
            }

            #ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads)
            #endif
            for(i=0; i<k; i++){
                beta[n*i+t]*=c[t];
            }
        }
    }
    free(eb);
}

/*
 * Forward pass for haploid HMM with uint8 emissions.
 *
 * Computes alpha[i,t] = P(obs_0..t, state=i @ t) recursively:
 *   alpha[t] = emission(y[t]) * sum_{states @ t-1} alpha[t-1] * trans[state->i]
 *
 * Parameters:
 *   k: number of states (reference haplotypes)
 *   n: number of variants
 *   y: observations (2 x n), [ref_count, alt_count] at each site
 *   nu: initial state probabilities (k)
 *   s: recombination probabilities (n-1), s[t] = P(no recombination t->t+1)
 *   g: emission matrix (k x n) uint8, haplotype alleles
 *   alpha: output forward probabilities (k x n), alpha[i*n+t] = P(data_0..t, state=i@t)
 *   c: scaling coefficients (n)
 *   scaleto: scaling mode
 *   nthreads: OpenMP thread count
 */
void _forwardHaploid(int k, int n, uint8_t *y, double *nu, double *s, uint8_t *g, double *alpha, double *c, unsigned int scaleto, int nthreads, double *alphaMat, double minp) {
    int i,t=0;
    double *z=(double *) malloc(k*sizeof(double));
    double colsum;

    c[t]=0.0;
    double ct=0.0;
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads) reduction(+:ct)
    #endif
    for(i=0; i<k; i++){
        double e=pow(1-fabs(((double)g[i*n+t])-minp),y[t*2]) * pow( (double) fabs(((double)g[i*n+t])-minp),y[t*2+1]);
        z[i] = nu[i] * e;
        ct += z[i];
    }
    c[t]=((double)scaleto)/ct;

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads)
    #endif
    for(i=0; i<k; i++){
        if (scaleto>0){
            alpha[n*i+t] = c[t]*z[i];
        } else {
            alpha[n*i+t] = z[i];
        }
    }

    colsum=(double)scaleto; //by definition if we use scaling

    for(t=1; t<n; t++){
        ct=0.0;

        if (scaleto==0){ //no scaling!
            colsum=0.0;
            for(i=0; i<k; i++) {
                colsum+=alpha[(i*n)+(t-1)];
            }
        }

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) reduction(+:ct)
        #endif
        for(i=0; i<k; i++){
            //no recomb
            double zi = alpha[(i*n)+(t-1)] * s[t-1]; //p of no recomb between pos t-1 and t

            //recomb
            double am_i = (alphaMat != NULL) ? alphaMat[i*(n-1)+(t-1)] : (1.0/k);
            zi += colsum * (1.0-s[t-1]) * am_i; //p of recomb to state i between pos t-1 and t

            double e=pow(1-fabs(((double)g[i*n+t])-minp),y[t*2]) * pow(fabs(((double)g[i*n+t])-minp),y[t*2+1]);

            zi *= e; //emission of y[t*2] and y[t*2+1] in state i

            z[i] = zi;
            ct += zi;
        }

        c[t]=((double)scaleto)/ct;

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads)
        #endif
        for(i=0; i<k; i++){
            if (scaleto>0){
                alpha[n*i+t] = z[i] * c[t];
            } else {
                alpha[n*i+t] = z[i];
            }
        }
    }
    free(z);
}

/* ---------- Double-emission variants for diploid adjusted emissions ----------
 * Identical to the Haploid versions except:
 *   - g is double* (float64 emission probabilities, not uint8 {0,1})
 *   - emission uses g[i*n+t] directly as P(alt), no minp transform
 * Used by the diploid mean-field variational inference: each HMM pass
 * conditions on the other haplotype's posterior via adjusted emissions.
 */

void _forwardHaploidDouble(int k, int n, uint8_t *y, double *nu, double *s, double *g, double *alpha, double *c, unsigned int scaleto, int nthreads, double *alphaMat) {
    int i,t=0;
    double *z=(double *) malloc(k*sizeof(double));
    double colsum;

    c[t]=0.0;
    double ct=0.0;
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads) reduction(+:ct)
    #endif
    for(i=0; i<k; i++){
        double e=pow(1.0-g[i*n+t],y[t*2]) * pow(g[i*n+t],y[t*2+1]);
        z[i] = nu[i] * e;
        ct += z[i];
    }
    c[t]=((double)scaleto)/ct;

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads)
    #endif
    for(i=0; i<k; i++){
        if (scaleto>0){
            alpha[n*i+t] = c[t]*z[i];
        } else {
            alpha[n*i+t] = z[i];
        }
    }

    colsum=(double)scaleto;

    for(t=1; t<n; t++){
        ct=0.0;

        if (scaleto==0){
            colsum=0.0;
            for(i=0; i<k; i++) {
                colsum+=alpha[(i*n)+(t-1)];
            }
        }

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) reduction(+:ct)
        #endif
        for(i=0; i<k; i++){
            double zi = alpha[(i*n)+(t-1)] * s[t-1];

            double am_i = (alphaMat != NULL) ? alphaMat[i*(n-1)+(t-1)] : (1.0/k);
            zi += colsum * (1.0-s[t-1]) * am_i;

            double e=pow(1.0-g[i*n+t],y[t*2]) * pow(g[i*n+t],y[t*2+1]);

            zi *= e;

            z[i] = zi;
            ct += zi;
        }

        c[t]=((double)scaleto)/ct;

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads)
        #endif
        for(i=0; i<k; i++){
            if (scaleto>0){
                alpha[n*i+t] = z[i] * c[t];
            } else {
                alpha[n*i+t] = z[i];
            }
        }
    }
    free(z);
}

void _backwardHaploidDouble(int k, int n, uint8_t *y, double *s, double *g, double *beta, double *c, int scaleto, int nthreads, double *alphaMat) {
    int i;
    double *eb=(double *) malloc(k*sizeof(double));
    int t=n-1;

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads)
    #endif
    for(i=0; i<k; i++){
        beta[n*i+t] = c[t];
    }

    for(t=n-2; t>=0; t--){
        double colsum=0.0;

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) reduction(+:colsum)
        #endif
        for(i=0; i<k; i++){
            double e=pow(1.0-g[i*n+t+1],y[(t+1)*2]) * pow(g[i*n+t+1],y[(t+1)*2+1]);
            eb[i] = beta[n*i+t+1] * e;
            colsum += eb[i];
        }

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads)
        #endif
        for(i=0; i<k; i++){
            double am_i = (alphaMat != NULL) ? alphaMat[i*(n-1)+t] : (1.0/k);
            beta[n*i+t] = eb[i]*s[t] + colsum*(1.0-s[t])*am_i;

            if (scaleto!=0){
                beta[n*i+t]*=c[t];
            }
        }
    }
    free(eb);
}

void _backward(int k, int n, uint8_t *y, double *s, double *a, double *g, double *beta, double *c) { //, double *xi_nor) {
    int i,j,t;
    // double z[k*k];
    double e[k*k];

    for(i=0; i<k; i++){
        for(j=0; j<k; j++){
            beta[(((i*k)+j)*n)+(n-1)] = c[n-1];
        }
    }

    double k1_sum[k], k2_sum[k], colsum;

    for(t=n-2; t>=0; t--){
        for (i=0;i<k;i++) k1_sum[i]=0.0; //reset
        for (j=0;j<k;j++) k2_sum[j]=0.0; //reset
        colsum=0.0;

        emission(g,y,k,n,t+1,e); //determine emission probabilties for t+1

        for(i=0; i<k; i++){ //init
            for(j=0; j<k; j++){
                k1_sum[i]+= a[(j*(n-1))+(t)] * beta[(((i*k)+j)*n)+(t+1)] * e[(i*k)+j];
                k2_sum[j]+= a[(i*(n-1))+(t)] * beta[(((i*k)+j)*n)+(t+1)] * e[(i*k)+j];
                colsum+=    a[(i*(n-1))+(t)] * a[(j*(n-1))+(t)] * beta[(((i*k)+j)*n)+(t+1)] * e[(i*k)+j];
            }
        }

        for(i=0; i<k; i++){
            for(j=0; j<k; j++){
                beta[(((i*k)+j)*n)+(t)]=beta[(((i*k)+j)*n)+(t+1)] * e[(i*k)+j] * (s[t]*s[t]) + //no recomb
                                        (k1_sum[i] * (s[t]*(1-s[t]))) + //recomb on first allele
                                        (k2_sum[j] * (s[t]*(1-s[t]))) + //recomb on second allele
                                        (colsum    * ((1-s[t]) * (1-s[t]))) ; //recomb on both alleles
                beta[(((i*k)+j)*n)+(t)]*=c[t]; //scale
            }
        }
    }
}

void _forward(int k, int n, uint8_t *y, double *nu, double *s, double *a, double *g, double *alpha, double *c) {
    int i,j,t=0;
    double z[k*k];
    double e[k*k];

    c[t]=0.0;
    emission(g,y,k,n,0,e); //set emission probabilties for t=0
    for(i=0; i<k; i++){
        for(j=0; j<k; j++){
            z[(i*k)+j] = nu[i] * nu[j] * e[(i*k)+j];
            // fprintf(stderr, "j=%d - z[(i*k)+j]=%f   nu[i]=%f  * nu[j]=%f * e[(i*k)+j]=%f \n", j, z[(i*k)+j], nu[i], nu[j], e[(i*k)+j]);
            c[t] += z[(i*k)+j];
        }
    }

    c[t]=1.0/c[t];

    for(j=0; j<k*k; j++){
        alpha[n*j+t] = c[t]*z[j]; //normalise at t=0
    }

    double k1_sum[k], k2_sum[k], colsum;
    for(t=1; t<n; t++){
        c[t]=0.0;
        for (i=0;i<k;i++) k1_sum[i]=0.0; //reset
        for (j=0;j<k;j++) k2_sum[j]=0.0; //reset

        colsum=0.0;
        for(i=0; i<k; i++){ //init
            for(j=0; j<k; j++){
                k1_sum[i]+=alpha[(((i*k)+j)*n)+(t-1)];
                k2_sum[j]+=alpha[(((i*k)+j)*n)+(t-1)];
                colsum+=alpha[(((i*k)+j)*n)+(t-1)];
            }
        }

        emission(g,y,k,n,t,e); //set emission for t

        for(i=0; i<k; i++){
            for(j=0; j<k; j++){
                //0
                z[(i*k)+j]=alpha[(((i*k)+j)*n)+(t-1)] * (s[t-1]*s[t-1]); //p of no recomb between pos t-1 and t
                //1
                z[(i*k)+j]+=k1_sum[i] * a[(j*(n-1))+(t-1)] * (s[t-1]*(1-s[t-1])); //p of recomb of first allele between pos t-1 and t
                z[(i*k)+j]+=k2_sum[j] * a[(i*(n-1))+(t-1)] * (s[t-1]*(1-s[t-1])); //p of recomb of first allele between pos t-1 and t
                //2
                z[(i*k)+j]+=colsum * a[(i*(n-1))+(t-1)] * a[(j*(n-1))+(t-1)] * ((1-s[t-1])*(1-s[t-1])); //p of recomb of both alleles between pos t-1 and t
                z[(i*k)+j]*=e[(i*k)+j];
                c[t] += z[(i*k)+j];
            }
        }

        c[t]=1.0/c[t];
        for(j=0; j<k*k; j++){
            alpha[n*j+t] = z[j]*c[t];
        }
    }
}

static PyObject* backward_impl(PyObject* self, PyObject* args, PyObject *keywds)
{
  PyObject *PY_y, *PY_s, *PY_g, *PY_beta, *PY_c, *PY_a;
  npy_intp betaSize[2];

  if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &PyArray_Type, &PY_y,
                                            &PyArray_Type, &PY_s,
                                            &PyArray_Type, &PY_a,
                                            &PyArray_Type, &PY_g,
                                            &PyArray_Type, &PY_c
                                            )){
    return NULL;
  }

  int k = NPY_ARRAY_DIMS(PY_g)[0];
  int n = NPY_ARRAY_DIMS(PY_y)[0];

  assert(NPY_ARRAY_DIMS(PY_g)[1]==n);
  assert(NPY_ARRAY_DIMS(PY_s)[0]==n-1);
  assert(NPY_ARRAY_DIMS(PY_a)[0]==k);
  assert(NPY_ARRAY_DIMS(PY_a)[1]==n-1);

  uint8_t *y = (uint8_t *)NPY_ARRAY_DATA(PY_y);
  double *s = (double *)NPY_ARRAY_DATA(PY_s);
  double *g = (double *)NPY_ARRAY_DATA(PY_g);
  double *a = (double *)NPY_ARRAY_DATA(PY_a);

  betaSize[0] = k*k; betaSize[1] = n;

  PY_beta = PyArray_SimpleNew(2, betaSize, NPY_DOUBLE);
  NPY_ARRAY_FILLWBYTE(PY_beta,0);
  double *beta = (double *)NPY_ARRAY_DATA(PY_beta);
  double *c = (double*)NPY_ARRAY_DATA(PY_c);

  _backward(k,n,y,s,a,g,beta,c);

  PyObject *ret=Py_BuildValue("O", PY_beta);
  Py_DECREF(PY_beta);

  return ret;
}

static PyObject* backwardHaploid_impl(PyObject* self, PyObject* args, PyObject *keywds)
{
  PyObject *PY_y, *PY_s, *PY_g, *PY_beta, *PY_c;
  PyObject *PY_alphaMat = NULL;
  unsigned int scaleto=0; //by default apply no scaling; -1=use Py_c array; >0 scale to fixed number
  double minp=0.01;

  npy_intp betaSize[2];

  int nthreads=1;
  static char *kwlist[] = {"y", "s", "g", "scale", "nthreads", "alphaMat", "minp", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!O!|OiOd", kwlist, &PyArray_Type, &PY_y,
                                            &PyArray_Type, &PY_s,
                                            &PyArray_Type, &PY_g, &PY_c, &nthreads,
                                            &PY_alphaMat, &minp
                                            )){
      return NULL;
  }

  int k = NPY_ARRAY_DIMS(PY_g)[0];
  int n = NPY_ARRAY_DIMS(PY_y)[0];

  assert(NPY_ARRAY_DIMS(PY_g)[1]==n);
  assert(NPY_ARRAY_DIMS(PY_s)[0]==n-1);

  uint8_t *y = (uint8_t *)NPY_ARRAY_DATA(PY_y);
  double *s = (double *)NPY_ARRAY_DATA(PY_s);
  uint8_t *g = (uint8_t *)NPY_ARRAY_DATA(PY_g);

  double *alphaMat = NULL;
  if (PY_alphaMat != NULL && PY_alphaMat != Py_None && PyArray_Check(PY_alphaMat)) {
      assert(NPY_ARRAY_DIMS(PY_alphaMat)[0]==k);
      assert(NPY_ARRAY_DIMS(PY_alphaMat)[1]==n-1);
      alphaMat = (double *)NPY_ARRAY_DATA(PY_alphaMat);
  }

  betaSize[0] = k; betaSize[1] = n;

  PY_beta = PyArray_SimpleNew(2, betaSize, NPY_DOUBLE);
  NPY_ARRAY_FILLWBYTE(PY_beta,0);
  double *beta = (double *)NPY_ARRAY_DATA(PY_beta);

  if (!PyArray_Check(PY_c)){ //if not an array, it has to be an unsigned int to scale columns to
      PyArg_Parse(PY_c,"I",&scaleto);
      //Py_DECREF(PY_c);
      PY_c = PyArray_SimpleNew(1, &betaSize[1], NPY_DOUBLE);
      NPY_ARRAY_FILLWBYTE(PY_c,0);
  } else { //use c array to scale
      scaleto=-1;
      Py_INCREF(PY_c);
  }

  double *c = (double*)NPY_ARRAY_DATA(PY_c);

  Py_BEGIN_ALLOW_THREADS
  _backwardHaploid(k,n,y,s,g,beta,c,scaleto,nthreads,alphaMat,minp);
  Py_END_ALLOW_THREADS

  PyObject *ret=Py_BuildValue("(O,O)", PY_beta, PY_c);

  Py_DECREF(PY_beta);
  Py_DECREF(PY_c);

  return ret;
}

static PyObject* forward_impl(PyObject* self, PyObject* args, PyObject *keywds)
{
  PyObject *PY_y, *PY_nu, *PY_s, *PY_a, *PY_g, *PY_alpha, *PY_c;
  // PyObject *PYs_y, *PYs_nu, *PYs_Q, *PYs_g;
  npy_intp alphaSize[2];

  if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &PyArray_Type, &PY_y, //Allele allecounts
                                            &PyArray_Type, &PY_nu, //Start probabilties
                                            &PyArray_Type, &PY_s, // transition probability of changing at point p
                                            &PyArray_Type, &PY_a, // transition probabilities of changing to state k at point p
                                            &PyArray_Type, &PY_g)){ //emission probabilties
    return NULL;
  }

  int k = NPY_ARRAY_DIMS(PY_nu)[0];
  int n = NPY_ARRAY_DIMS(PY_y)[0];

  //check that g has k*k rows and n columns
  // assert(NPY_ARRAY_DIMS(PY_g)[0]==k*k);

  //check that input g has k rows and n columns
  assert(NPY_ARRAY_DIMS(PY_g)[0]==k);
  assert(NPY_ARRAY_DIMS(PY_g)[1]==n);
  assert(NPY_ARRAY_DIMS(PY_s)[0]==n-1); //p of no recombination between t and t+1
  assert(NPY_ARRAY_DIMS(PY_a)[0]==k);
  assert(NPY_ARRAY_DIMS(PY_a)[1]==n-1);

  uint8_t *y = (uint8_t *)NPY_ARRAY_DATA(PY_y);
  double *nu = (double *)NPY_ARRAY_DATA(PY_nu);
  double *s = (double *)NPY_ARRAY_DATA(PY_s);
  double *g = (double *)NPY_ARRAY_DATA(PY_g);
  double *a = (double *)NPY_ARRAY_DATA(PY_a);

  alphaSize[0] = k*k; alphaSize[1] = n;

  PY_alpha = PyArray_SimpleNew(2, alphaSize, NPY_DOUBLE);
  NPY_ARRAY_FILLWBYTE(PY_alpha,0);
  double *alpha = (double *)NPY_ARRAY_DATA(PY_alpha);

  PY_c = PyArray_SimpleNew(1, &alphaSize[1], NPY_DOUBLE);
  NPY_ARRAY_FILLWBYTE(PY_c,0);
  double *c = (double*)NPY_ARRAY_DATA(PY_c);

  _forward(k,n,y,nu,s,a,g,alpha,c);

  PyObject *ret=Py_BuildValue("(O,O)", PY_alpha, PY_c);

  Py_DECREF(PY_alpha);
  Py_DECREF(PY_c);

  return ret;
}

static PyObject* forwardHaploid_impl(PyObject* self, PyObject* args, PyObject *keywds)
{
  PyObject *PY_y, *PY_nu, *PY_s, *PY_g, *PY_alpha, *PY_c;
  PyObject *PY_alphaMat = NULL;
  npy_intp alphaSize[2];
  unsigned int scaleto=1;
  int nthreads=1;
  double minp=0.01;

  static char *kwlist[] = {"y", "nu", "s", "g", "scale", "nthreads", "alphaMat", "minp", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!O!O!|IiOd", kwlist,
                                            &PyArray_Type, &PY_y, //Allele allecounts
                                            &PyArray_Type, &PY_nu, //Start probabilties
                                            &PyArray_Type, &PY_s, // transition probability of changing at point p
                                            &PyArray_Type, &PY_g, &scaleto, &nthreads,
                                            &PY_alphaMat, &minp)){ //emission probabilties
      return NULL;
  }

  int k = NPY_ARRAY_DIMS(PY_nu)[0];
  int n = NPY_ARRAY_DIMS(PY_y)[0];

  //check that input g has k rows and n columns
  assert(NPY_ARRAY_DIMS(PY_g)[0]==k);
  assert(NPY_ARRAY_DIMS(PY_g)[1]==n);
  assert(NPY_ARRAY_DIMS(PY_s)[0]==n-1); //p of no recombination between t and t+1

  uint8_t *y = (uint8_t *)NPY_ARRAY_DATA(PY_y);
  double *nu = (double *)NPY_ARRAY_DATA(PY_nu);
  double *s = (double *)NPY_ARRAY_DATA(PY_s);
  uint8_t *g = (uint8_t *)NPY_ARRAY_DATA(PY_g);

  double *alphaMat = NULL;
  if (PY_alphaMat != NULL && PY_alphaMat != Py_None && PyArray_Check(PY_alphaMat)) {
      assert(NPY_ARRAY_DIMS(PY_alphaMat)[0]==k);
      assert(NPY_ARRAY_DIMS(PY_alphaMat)[1]==n-1);
      alphaMat = (double *)NPY_ARRAY_DATA(PY_alphaMat);
  }

  alphaSize[0] = k; alphaSize[1] = n;

  PY_alpha = PyArray_SimpleNew(2, alphaSize, NPY_DOUBLE);
  NPY_ARRAY_FILLWBYTE(PY_alpha,0);
  double *alpha = (double *)NPY_ARRAY_DATA(PY_alpha);

  PY_c = PyArray_SimpleNew(1, &alphaSize[1], NPY_DOUBLE);
  NPY_ARRAY_FILLWBYTE(PY_c,0);
  double *c = (double*)NPY_ARRAY_DATA(PY_c);

  Py_BEGIN_ALLOW_THREADS
  _forwardHaploid(k,n,y,nu,s,g,alpha,c,scaleto,nthreads,alphaMat,minp);
  Py_END_ALLOW_THREADS

  PyObject * ret = Py_BuildValue("(O,O)", PY_alpha, PY_c);

  Py_DECREF(PY_alpha);
  Py_DECREF(PY_c);
  return ret;
}

static PyObject* forwardHaploidDouble_impl(PyObject* self, PyObject* args, PyObject *keywds)
{
  PyObject *PY_y, *PY_nu, *PY_s, *PY_g, *PY_alpha, *PY_c;
  PyObject *PY_alphaMat = NULL;
  npy_intp alphaSize[2];
  unsigned int scaleto=1;
  int nthreads=1;

  static char *kwlist[] = {"y", "nu", "s", "g", "scale", "nthreads", "alphaMat", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!O!O!|IiO", kwlist,
                                            &PyArray_Type, &PY_y,
                                            &PyArray_Type, &PY_nu,
                                            &PyArray_Type, &PY_s,
                                            &PyArray_Type, &PY_g, &scaleto, &nthreads,
                                            &PY_alphaMat)){
      return NULL;
  }

  int k = NPY_ARRAY_DIMS(PY_nu)[0];
  int n = NPY_ARRAY_DIMS(PY_y)[0];

  assert(NPY_ARRAY_DIMS(PY_g)[0]==k);
  assert(NPY_ARRAY_DIMS(PY_g)[1]==n);
  assert(NPY_ARRAY_DIMS(PY_s)[0]==n-1);

  uint8_t *y = (uint8_t *)NPY_ARRAY_DATA(PY_y);
  double *nu = (double *)NPY_ARRAY_DATA(PY_nu);
  double *s = (double *)NPY_ARRAY_DATA(PY_s);
  double *g = (double *)NPY_ARRAY_DATA(PY_g);  /* float64 emission */

  double *alphaMat = NULL;
  if (PY_alphaMat != NULL && PY_alphaMat != Py_None && PyArray_Check(PY_alphaMat)) {
      assert(NPY_ARRAY_DIMS(PY_alphaMat)[0]==k);
      assert(NPY_ARRAY_DIMS(PY_alphaMat)[1]==n-1);
      alphaMat = (double *)NPY_ARRAY_DATA(PY_alphaMat);
  }

  alphaSize[0] = k; alphaSize[1] = n;

  PY_alpha = PyArray_SimpleNew(2, alphaSize, NPY_DOUBLE);
  NPY_ARRAY_FILLWBYTE(PY_alpha,0);
  double *alpha = (double *)NPY_ARRAY_DATA(PY_alpha);

  PY_c = PyArray_SimpleNew(1, &alphaSize[1], NPY_DOUBLE);
  NPY_ARRAY_FILLWBYTE(PY_c,0);
  double *c = (double*)NPY_ARRAY_DATA(PY_c);

  Py_BEGIN_ALLOW_THREADS
  _forwardHaploidDouble(k,n,y,nu,s,g,alpha,c,scaleto,nthreads,alphaMat);
  Py_END_ALLOW_THREADS

  PyObject * ret = Py_BuildValue("(O,O)", PY_alpha, PY_c);

  Py_DECREF(PY_alpha);
  Py_DECREF(PY_c);
  return ret;
}

static PyObject* backwardHaploidDouble_impl(PyObject* self, PyObject* args, PyObject *keywds)
{
  PyObject *PY_y, *PY_s, *PY_g, *PY_beta, *PY_c;
  PyObject *PY_alphaMat = NULL;
  npy_intp betaSize[2];
  int scaleto=1;
  int nthreads=1;

  static char *kwlist[] = {"y", "s", "g", "scale", "nthreads", "alphaMat", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!O!|OiO", kwlist,
                                            &PyArray_Type, &PY_y,
                                            &PyArray_Type, &PY_s,
                                            &PyArray_Type, &PY_g,
                                            &PY_c, &nthreads,
                                            &PY_alphaMat)){
      return NULL;
  }

  int k = NPY_ARRAY_DIMS(PY_g)[0];
  int n = NPY_ARRAY_DIMS(PY_y)[0];

  assert(NPY_ARRAY_DIMS(PY_g)[1]==n);
  assert(NPY_ARRAY_DIMS(PY_s)[0]==n-1);

  uint8_t *y = (uint8_t *)NPY_ARRAY_DATA(PY_y);
  double *s = (double *)NPY_ARRAY_DATA(PY_s);
  double *g = (double *)NPY_ARRAY_DATA(PY_g);  /* float64 emission */

  double *alphaMat = NULL;
  if (PY_alphaMat != NULL && PY_alphaMat != Py_None && PyArray_Check(PY_alphaMat)) {
      assert(NPY_ARRAY_DIMS(PY_alphaMat)[0]==k);
      assert(NPY_ARRAY_DIMS(PY_alphaMat)[1]==n-1);
      alphaMat = (double *)NPY_ARRAY_DATA(PY_alphaMat);
  }

  betaSize[0] = k; betaSize[1] = n;

  PY_beta = PyArray_SimpleNew(2, betaSize, NPY_DOUBLE);
  NPY_ARRAY_FILLWBYTE(PY_beta,0);
  double *beta = (double *)NPY_ARRAY_DATA(PY_beta);

  if (!PyArray_Check(PY_c)){
      PyArg_Parse(PY_c,"I",&scaleto);
      PY_c = PyArray_SimpleNew(1, &betaSize[1], NPY_DOUBLE);
      NPY_ARRAY_FILLWBYTE(PY_c,0);
  } else {
      scaleto=-1;
      Py_INCREF(PY_c);
  }

  double *c = (double*)NPY_ARRAY_DATA(PY_c);

  Py_BEGIN_ALLOW_THREADS
  _backwardHaploidDouble(k,n,y,s,g,beta,c,scaleto,nthreads,alphaMat);
  Py_END_ALLOW_THREADS

  PyObject *ret=Py_BuildValue("(O,O)", PY_beta, PY_c);

  Py_DECREF(PY_beta);
  Py_DECREF(PY_c);

  return ret;
}

static PyObject* fit_impl(PyObject* self, PyObject* args, PyObject *keywds)
{
    PyArrayObject *npV, *npStart, *npSigma, *npEmission, *npAlpha, *npPos;
    npPos=NULL;
    int vi,i,j,it,maxit=10,nthreads=1,ngen=100;

    static char *kwlist[] = {"matrix of sequences","start probabilities",
                             "alpha","recombination probabilities","emission probabilities",
                             "vpos","maxiter","nthreads","ngen", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!O!O!O!|Oiii", kwlist,
                                            &PyArray_Type, &npV,
                                            &PyArray_Type, &npStart,
                                            &PyArray_Type, &npAlpha,
                                            &PyArray_Type, &npSigma,
                                            &PyArray_Type, &npEmission, &npPos, &maxit, &nthreads, &ngen)){
        return NULL;
    }

    long *Vdims = NPY_ARRAY_DIMS(npV);

    npy_intp fwbwSize[2];
    npy_intp cSize[2];

    int k = (int)NPY_ARRAY_DIMS(npStart)[0];
    int n = (int)Vdims[1];

    long *alphaDims = NPY_ARRAY_DIMS(npAlpha);
    npy_intp aSize[2];
    aSize[0]=alphaDims[0];
    aSize[1]=alphaDims[1];

    npy_intp eSize[2];
    eSize[0]=k;
    eSize[1]=n;

    assert(NPY_ARRAY_DIMS(npEmission)[1]==n);
    assert(NPY_ARRAY_DIMS(npSigma)[0]==n-1);
    assert(NPY_ARRAY_DIMS(npAlpha)[0]==k);
    assert(NPY_ARRAY_DIMS(npAlpha)[1]==n-1);

    cSize[0]=maxit;
    cSize[1]=n;

    PyObject *npC = PyArray_SimpleNew(2, cSize, NPY_DOUBLE);
    NPY_ARRAY_FILLWBYTE(npC,0);
    double *C = (double *)NPY_ARRAY_DATA(npC);

    fwbwSize[0]=k*k;
    fwbwSize[1]=n;

    uint8_t *V = (uint8_t *)NPY_ARRAY_DATA(npV); //access the data referenced by the np arrays
    double *nu = (double *)NPY_ARRAY_DATA(npStart);
    double *a = (double *)NPY_ARRAY_DATA(npAlpha);
    double *s = (double *)NPY_ARRAY_DATA(npSigma);
    double *g = (double *)NPY_ARRAY_DATA(npEmission);

    uint32_t *pos;
    if (npPos!=NULL) pos = (uint32_t *)NPY_ARRAY_DATA(npPos);

    double ll=0, pl=0;
    time_t t0;

    //Use Numpy datastructures just in case we want to debug things in Python at some point
    PyObject *npasum = PyArray_SimpleNew(2, aSize, NPY_DOUBLE);
    PyObject *npgammaSum0 = PyArray_SimpleNew(2, eSize, NPY_DOUBLE);
    PyObject *npgammaSum1 = PyArray_SimpleNew(2, eSize, NPY_DOUBLE);

    for(it=0; it<maxit; it++){

        t0 = time(NULL);
        fprintf(stderr, "Iteration: %d\n", it);

        NPY_ARRAY_FILLWBYTE(npasum,0);
        NPY_ARRAY_FILLWBYTE(npgammaSum0,0);
        NPY_ARRAY_FILLWBYTE(npgammaSum1,0);

        double *asum = (double *)NPY_ARRAY_DATA(npasum);
        double *gammaSum0 = (double *)NPY_ARRAY_DATA(npgammaSum0); //emission update
        double *gammaSum1 = (double *)NPY_ARRAY_DATA(npgammaSum1); //emission update

        double priorsum[k*k];
        for(i=0; i<(k*k); i++) priorsum[i]=0.0;

        ll=0.0;

        //release GIL
        Py_BEGIN_ALLOW_THREADS

        double *C_=&C[it*n];

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) reduction(+:ll, \
                                                                 priorsum[:k*k], \
                                                                 asum[:(n-1)*k], \
                                                                 gammaSum0[:n*k], \
                                                                 gammaSum1[:n*k], \
                                                                 C_[:n])
        #endif
        for (vi=0; vi<Vdims[0]; vi++) {
            // fprintf(stderr, "Sequence: %d\n", vi);

            uint8_t* y=&V[vi*n*2];

            double *alpha=(double *) malloc(k*k*n*sizeof(double));
            double *beta=(double *) malloc(k*k*n*sizeof(double));
            double *c=(double *) malloc(n*sizeof(double));

            _forward(k, n, y, nu, s, a, g, alpha, c);
            _backward(k, n, y, s, a, g, beta, c);

            int t=0,i=0,j=0;
            double tmp[k];
            double e[k*k];

            double gamma;
            double k1_pobs;
            double k2_pobs;

            for(t=0; t<n-1; t++) {
                for (i=0;i<k;i++) tmp[i]=0.0; //reset

                for(i=0; i<k; i++){ //init
                    for(j=0; j<k; j++){ //sum over alpha, probability of changing into recombination state
                        tmp[i]+=alpha[(((i*k)+j)*n)+t] * (s[t] * (1-s[t])); //recombination on one allele
                    }
                    tmp[i]+=((1-s[t]) * (1-s[t])) * a[i*(n-1)+t] ; //TODO: I think this should be without the a term!
                }

                emission(g,y,k,n,t+1,e);

                for(i=0; i<k; i++){
                    for(j=0; j<k; j++){
                        asum[i*(n-1)+t] += tmp[j] * beta[(((i*k)+j)*n)+(t+1)] * e[(i*k)+j] * a[i*(n-1)+t] * 2;
                    }
                }
            }

            double colsum;
            for(t=0; t<n; t++){ //for each position
                colsum=0.0;

                for(i=0; i<k; i++){ //for each maternal state (from)
                    for(j=0; j<k; j++){ //for each paternal state (from)
                        gamma=alpha[((i*k)+j)*n+t] * beta[((i*k)+j)*n+t] * (1.0/c[t]);
                        colsum+=gamma;
                    }
                }

                if (t < n-1) {
                    emission(g,y,k,n,t+1,e);
                }

                for(i=0; i<k; i++){ //for each maternal state (from)
                    for(j=0; j<k; j++){ //for each paternal state (from)
                        gamma=alpha[((i*k)+j)*n+t] * beta[((i*k)+j)*n+t] * (1.0/c[t]);
                        k1_pobs=pow(1-g[i*n+t],y[t*2]) * pow(g[i*n+t],y[t*2+1]);
                        k2_pobs=pow(1-g[j*n+t],y[t*2]) * pow(g[j*n+t],y[t*2+1]);
                        gammaSum0[i*n+t] += gamma * (k1_pobs/ (k1_pobs+k2_pobs)) * y[t*2+1];
                        gammaSum1[i*n+t] += gamma * (k1_pobs/ (k1_pobs+k2_pobs)) * y[t*2];
                    }
                }

                if (t==0){
                    for(i=0; i<k; i++){ //for each maternal state (from)
                        for(j=0; j<k; j++){ //for each paternal state (from)
                            gamma=alpha[((i*k)+j)*n+t]*beta[((i*k)+j)*n+t]*(1.0/c[t]);
                            priorsum[((i*k)+j)]+=gamma;///colsum;
                        }
                    }
                }
                // ll += c[t] < 1.0 ? 0 : log(1.0/c[t]);
                ll += log(1.0/c[t]);
                C_[t]+= log(1.0/c[t]);
            }

            free(alpha);
            free(beta);
            free(c);
        }

        //acquire GIL again no more multi-threading
        Py_END_ALLOW_THREADS

        assert(ll<=0.0);

        double minp=1e-3; //min probability, to prevent zero emission probabilities

        //Now update start probabilities
        double z[k],gammatot=0.0;

        int x=0;
        for (x=0;x<k;x++) z[x]=0.0; //reset

        for(i=0; i<k; i++) {
            for(j=0; j<k; j++) {
                z[i]+=priorsum[((i*k)+j)]; //TODO: should be symmetric, so could just sum over i or j
                z[j]+=priorsum[((i*k)+j)];
                gammatot+=priorsum[((i*k)+j)];
            }
        }

        double nusum=0.0;
        for(i=0; i<k; i++) {
            nu[i]=z[i]/gammatot;
            if (nu[i]>1-minp) nu[i]=1-minp; //cap to prevent underflows
            if (nu[i]<minp) nu[i]=minp;

            assert(nu[i]>0.0);
            assert(nu[i]<1.0);
            nusum+=nu[i];
        }

        for(i=0; i<k; i++) nu[i]*=(1/nusum);

        //Now update emission probabilties

        int t;

        for(t=0; t<n; t++){ //for each position

            for(i=0; i<k; i++){
              if ((gammaSum0[(i*n)+t]+gammaSum1[(i*n)+t])==0.0) {
                g[(i*n)+t]=minp; // no observations for this allele, emit minp0
              } else {
                g[(i*n)+t] = gammaSum0[(i*n)+t]/(gammaSum0[(i*n)+t]+gammaSum1[(i*n)+t]);
              }

              if (g[(i*n)+t]<minp){
                g[(i*n)+t]=minp;
              }

              if (g[(i*n)+t]>1-minp){
                g[(i*n)+t]=1-minp;
              }

            }
        }

        //update transition/recombination probabilities
        int d;
        double minrate=0.1, maxrate=100, colsum;
        for(t=0; t<n-1; t++) { //for each position

            if (npPos!=NULL){
                d=pos[t+1]-pos[t];
                assert(d>0); //otherwise positions not sorted? or across chromosomes?
            } else {
                d=10000; //default distance between snps
            }

            minrate=exp(-ngen * minrate * d/100/1000000); //TODO: use parameter here! nGen/ 0.1 cM/Mb is minrate
            maxrate=exp(-ngen * maxrate * d/100/1000000); //TODO: use parameter here! nGen/ 100 cM/Mb is maxrate

            colsum=0.0;
            for(i=0; i<k; i++) colsum+=asum[i*(n-1)+t];

            if (colsum==0.0) colsum=minp;

            if (colsum<=0.0) for(i=0; i<k; i++) fprintf(stderr, "asum[%d,%d]=%f\n", i,t,asum[i*(n-1)+t]);
            assert(colsum>0.0);

            s[t]=exp(-(colsum/Vdims[0]/2)); //update sigma

            assert(s[t]>0.0);

            for(i=0; i<k; i++){
                a[i*(n-1)+t] = asum[i*(n-1)+t]/colsum; //scale to update the a matrix
                //bound parameters
                if (a[i*(n-1)+t]<minp) a[i*(n-1)+t]=minp;
                if (a[i*(n-1)+t]>1.0-minp) a[i*(n-1)+t]=1.0-minp;

                assert(a[i*(n-1)+t]>0.0);
                assert(a[i*(n-1)+t]<1.0);
            }

            colsum=0.0;
            for(i=0; i<k; i++) colsum+=a[i*(n-1)+t];
            for(i=0; i<k; i++) a[i*(n-1)+t] = (1/colsum)*a[i*(n-1)+t]; //scale to sum to one
        }

        fprintf(stderr,"Log likelihood after iteration %d (took %ld seconds): %f - %f (%f)\n", it, (time(NULL) - t0), pl, ll, pl-ll);

        pl=ll;
    }

    Py_DECREF(npasum);
    Py_DECREF(npgammaSum0);
    Py_DECREF(npgammaSum1);

    //return updated matrices with transition and emission parameters
    PyObject *ret=Py_BuildValue("(O,O,O,O,O)",npStart,npAlpha,npSigma,npEmission,npC);

    Py_DECREF(npStart);
    Py_DECREF(npAlpha);
    Py_DECREF(npSigma);
    Py_DECREF(npEmission);
    Py_DECREF(npC);

    return ret;
}

static PyMethodDef imputeff_hmmMethods[] = {
    {"fit", (PyCFunction)fit_impl, METH_VARARGS|METH_KEYWORDS, "fit"},
    // {"logfit", logfit, METH_VARARGS|METH_KEYWORDS, "logfit"},
    {"forward", (PyCFunction)forward_impl, METH_VARARGS|METH_KEYWORDS, "forward"},
    {"forwardHaploid", (PyCFunction)forwardHaploid_impl, METH_VARARGS|METH_KEYWORDS, "forwardHaploid"},
    // {"logforward", logforward, METH_VARARGS|METH_KEYWORDS, "logforward"},
    {"backward", (PyCFunction)backward_impl, METH_VARARGS|METH_KEYWORDS, "backward"},
    {"backwardHaploid", (PyCFunction)backwardHaploid_impl, METH_VARARGS|METH_KEYWORDS, "backwardHaploid"},
    {"forwardHaploidDouble", (PyCFunction)forwardHaploidDouble_impl, METH_VARARGS|METH_KEYWORDS, "forwardHaploidDouble"},
    {"backwardHaploidDouble", (PyCFunction)backwardHaploidDouble_impl, METH_VARARGS|METH_KEYWORDS, "backwardHaploidDouble"},
    // {"logbackward", logbackward, METH_VARARGS|METH_KEYWORDS, "logbackward"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cfstats_impute_hmm =
{
    PyModuleDef_HEAD_INIT,
    "_hmm", /* name of module */
    "cfstats.impute compiled HMM routines",
    -1,
    imputeff_hmmMethods
};

PyMODINIT_FUNC PyInit__hmm(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&cfstats_impute_hmm);
}
