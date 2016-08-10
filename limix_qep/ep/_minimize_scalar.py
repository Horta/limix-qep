from __future__ import division

from math import sqrt

from numpy import finfo
from numpy import inf

from ._bracket import find_bracket
from ._brent import find_minimum as brent_minimum

_sqrt_epsilon = sqrt(finfo(float).eps)


def find_minimum(f, x0=None, x1=None, a=-inf, b=+inf, gfactor=2, glimit=2**8,
                 rtol=_sqrt_epsilon, atol=_sqrt_epsilon, maxiter=500):

    def func(x):
        func.nfev += 1
        return f(x)
    func.nfev = 0

    r = find_bracket(func, x0=x0, x1=x1, a=a, b=b, gfactor=gfactor,
                     glimit=glimit, rtol=rtol, atol=atol, maxiter=maxiter)

    (x0, x1, x2) = r[0]
    (f0, f1, f2) = r[1]

    x0 = brent_minimum(func, x0, x2, f0, f2, x1, f1, rtol, atol, maxiter)[0]
    return x0, func.nfev
