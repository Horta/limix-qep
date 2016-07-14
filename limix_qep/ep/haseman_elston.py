from __future__ import division

from numba import jit

from numpy import ascontiguousarray

@jit
def _haseman_elston_regression(y, K):
    r1 = 0.
    r2 = 0.
    i = 0
    while i < y.shape[0]-1:
        j = i+1
        while j < y.shape[0]:
            r1 += y[i] * y[j] * K[i,j]
            r2 += K[i,j]*K[i,j]
            j += 1
        i += 1
    return r1 / r2

def heritability(y, K, enforce=True):
    y = ascontiguousarray(y, float)
    K = ascontiguousarray(K, float)

    if enforce:
        y = y - y.mean()
        y /= y.std()
        K /= K.diagonal().std()

    h2 = _haseman_elston_regression(y, K)
    return max(0.0, min(h2, 1.0))
