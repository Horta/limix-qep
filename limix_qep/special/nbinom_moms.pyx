cimport cython
from libc.math cimport log, exp, sqrt, fabs, INFINITY
from libc.stdio cimport printf, fflush, stdout

cdef extern from "nbinom_moms_base.h":
    double log_ulike_prior(double x, double N, double K,
                           double mu, double var) nogil
    void get_extrema(double N, double K, double mu, double var,
                     double* left, double* right) nogil
    void moments(double left, double step, int nints,
                 double N, double K, double mu, double var,
                 double* lmom0_res, double* mu_res, double* var_res) nogil

cdef extern from "cephes/cephes.h":
    double norm_logpdf(double x) nogil
    double norm_logcdf(double x) nogil
    double norm_logsf(double x) nogil
    double logaddexp(double x, double y) nogil
    double logaddexps(double x, double y, double sx,
                      double sy) nogil
    double logaddexpss(double x, double y, double sx,
                       double sy, double* sign) nogil
    double binomln(double N, double K) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void compute_heights(double left, double step, int nint,
                          double N, double K, double mu, double var,
                          double[:] heights) nogil:
    cdef:
        int i
    for i in range(nint+1):
        heights[i] = log_ulike_prior(left, N, K, mu, var)
        # printf("height: %.10f pos: %.10f%, ", heights[i], left)
        left += step
    # printf("\n")

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef double compute_weights(double left, double step, int nint,
                          double N, double K, double mu, double var,
                          double[:] heights,
                          double[:] w) nogil:
    cdef:
        double hl, hr, c
        double r
        double acum
        int i
    for i in range(nint):

        r = left + step

        hl = heights[i]
        hr = heights[i+1]
        c = (hr - hl) / step

        llc = log(fabs(c))
        if c > 0:
            w[i] = hl - left*c + logaddexps(r*c, left*c, 1, -1) - llc
        else:
            w[i] = hl - left*c + logaddexps(r*c, left*c, -1, 1) - llc

        if i == 0:
            acum = w[i]
        else:
            acum = logaddexp(acum, w[i])

        left = r

    return acum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double shrink_intervals(double* left,
                        double step, int nint,
                        double N, double K, double mu, double var,
                        double[:] heights, double[:] w) nogil:

    # cdef double* heights = <double*> malloc((nint+1) * sizeof(double))
    # cdef double* w = <double*> malloc(nint * sizeof(double))

    compute_heights(left[0], step, nint, N, K, mu, var, heights)
    cdef:
        double r
        double acum
        int i

    acum = compute_weights(left[0], step, nint, N, K, mu, var,
                           heights, w)

    for i in range(nint):
        w[i] = exp(w[i] - acum)
        # printf("%e ", w[i])
    # printf("\n")

    if (w[0] > 1e-1 or w[w.shape[0]-1] > 1e-1):
        printf("(%d, %d)  (%.5f %.5f) (%f %f) ", <int>N, <int>K, left[0], left[0] + step*nint, mu, var);
        for i in range(0, 2):
            printf("%e ", w[i])
        printf(".. ")
        for i in range(nint-2, nint):
            printf("%e ", w[i])
        printf("\n")

    i = nint
    while w[i-1] <= 1e-256:
        i -= 1
    # printf("i1 %d\n", i); fflush(stdout);
    r = left[0] + step * i

    cdef int j = 0
    while w[j+1] <= 1e-256 and j < i-1:
        j += 1
    # printf("i2 %d\n", j); fflush(stdout);
    left[0] += step * j

    return r

@cython.cdivision(True)
cdef void meaningful_interval(double N, double K, double mu, double var,
              double* left, double* right, double[:] heights, double[:] w) nogil:

    get_extrema(N, K, mu, var, left, right)
    # printf("extrema %.10f %.10f\n", left[0], right[0])

    cdef:
        int nint = w.shape[0]
        double step = (right[0] - left[0]) / nint

    # printf("initial limits: %.3f %.3f\n", left[0], right[0])
    right[0] = shrink_intervals(left, step, nint, N, K, mu, var, heights, w)
    # printf("final limits 1: %.3f %.3f\n", left[0], right[0])

    step = (right[0] - left[0]) / nint
    right[0] = shrink_intervals(left, step, nint, N, K, mu, var, heights, w)
    # printf("final limits 2: %.3f %.3f\n", left[0], right[0])

@cython.cdivision(True)
cpdef pmoments(int nintervals, double N, double K,
            double mu, double var, double[:] height, double[:] w):

    cdef double left
    cdef double right
    meaningful_interval(N, K, mu, var, &left, &right, height, w)
    # # printf("Limits: %.10f %.10f\n", left, right)
    cdef double step = (right - left) / nintervals
    cdef double lmom0, mu_res, var_res
    moments(left, step, nintervals, N, K, mu, var, &lmom0, &mu_res, &var_res)
    return (lmom0, mu_res, var_res)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef void moments_array(int nintervals, double[:] N, double[:] K,
            double[:] mu, double[:] var,
            double[:] height, double[:] w,
            double[:] lmom0, double[:] mu_res, double[:] var_res) nogil:
    cdef:
        int i
        double left, right
        double step

    for i in range(N.shape[0]):
        meaningful_interval(N[i], K[i], mu[i], var[i], &left, &right, height, w)
        step = (right - left) / nintervals
        moments(left, step, nintervals, N[i], K[i], mu[i], var[i],
        &lmom0[i], &mu_res[i], &var_res[i])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef void moments_array2(int nintervals, double[:] N, double[:] K,
             double[:] eta, double[:] tau,
             double[:] height, double[:] w,
             double[:] lmom0, double[:] mu_res, double[:] var_res) nogil:
    cdef:
        int i
        double left, right
        double step
        double mu, var

    for i in range(N.shape[0]):
        mu = eta[i]/tau[i]
        var = 1./tau[i]
        meaningful_interval(N[i], K[i], mu, var, &left, &right, height, w)
        step = (right - left) / nintervals
        moments(left, step, nintervals, N[i], K[i], mu, var,
                &lmom0[i], &mu_res[i], &var_res[i])

# import numpy as np
# cimport numpy as np
from cpython cimport array
import array

cdef int _nintervals
cdef double[:] _height
cdef double[:] _weight

cpdef init(int nintervals):
    global _nintervals, _height, _weight
    _nintervals = nintervals

    _height = array.clone(array.array('d', []), nintervals+1, zero=False)
    _weight = array.clone(array.array('d', []), nintervals, zero=False)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef void moments_array3(double[:] N, double[:] K,
             double[:] eta, double[:] tau,
             double[:] lmom0, double[:] mu_res, double[:] var_res):
    global _nintervals, _height, _weight
    cdef:
        int i
        double left, right
        double step
        double mu, var

    for i in range(N.shape[0]):
        mu = eta[i]/tau[i]
        var = 1./tau[i]
        meaningful_interval(N[i], K[i], mu, var, &left, &right, _height, _weight)
        step = (right - left) / _nintervals
        moments(left, step, _nintervals, N[i], K[i], mu, var,
                &lmom0[i], &mu_res[i], &var_res[i])
