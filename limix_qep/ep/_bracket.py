from __future__ import division

from math import sqrt

from numpy import isfinite
from numpy import finfo
from numpy import sign
from numpy import inf

_sqrt_epsilon = sqrt(finfo(float).eps)


def find_bracket(f, x0=None, x1=None, a=-inf, b=+inf, gfactor=2, glimit=2**8,
                 rtol=_sqrt_epsilon, atol=_sqrt_epsilon, maxiter=500):

    x = sorted([xi for xi in [x0, x1] if xi is not None])

    if len(x) == 0:
        x0 = min(max(0, a), b)
        if x0 - a > b - x0:
            x1 = x0 - (abs(x0) * rtol + atol)
        else:
            x1 = x0 + (abs(x0) * rtol + atol)
    elif len(x) == 1:
        x0 = x[0]
    elif len(x) == 2:
        x0, x1 = x[0], x[1]

    f0 = f(x0)
    f1 = f(x1)
    return _downhill(f, x0, x1, f0, f1, a, b, gfactor, glimit, rtol, atol,
                     maxiter)


def _downhill3(f, x0, x1, x2, f0, f1, l, gfactor, glimit, rtol, atol, maxiter):

    s = sign(x1 - x0)

    if isfinite(l):
        l_ = l - s * (rtol*abs(l) + atol)
    else:
        l_ = s * inf
    sl_ = s * l_

    if s * x2 >= sl_:
        x2 = l

    f2 = f(x2)

    nfails = 0
    nit = 0
    while not(f0 > f1 < f2) and x2 != l and nit < maxiter:
        nit += 1
        tol = rtol * abs(x2) + atol

        x21 = x2 - x1

        if nfails < 2:
            d = _parabolic_step(x2, x1, x0, f2, f1, f0)
            ok0 = abs(d) >= tol
            ok1 = d * s > 0
            ok2 = abs(d) <= abs(glimit * x21)
            if ok0 and ok1 and ok2:
                nfails = 0
            else:
                nfails += 1

        if nfails > 0:
            d = gfactor * x21
            d = s * min(s * d, s * glimit * x21)

        u = x2 + d
        if s * u >= sl_:
            u = l

        fu = f(u)

        x0, x1, x2 = x1, x2, u
        f0, f1, f2 = f1, f2, fu

    if s < 0:
        x2, x1, x0 = x0, x1, x2
        f2, f1, f0 = f0, f1, f2

    return ((x0, x1, x2), (f0, f1, f2), nit)


def _downhill(f, x0, x1, f0, f1, a, b, gfactor, glimit, rtol, atol, maxiter):

    assert a <= x0 <= b
    assert a <= x1 <= b
    assert x0 != x1
    assert gfactor > 1
    assert glimit >= gfactor
    assert maxiter > 0
    assert rtol >= 0
    assert atol >= 0

    # Downhill from x0 to x1.
    if f1 > f0:
        x0, x1 = x1, x0
        f0, f1 = f1, f0

    if x0 < x1:
        if isfinite(a) and x0 <= a + rtol*abs(a) + atol:
            x0 = a

        x2 = x1 + (x1 - x0) * gfactor
        return _downhill3(f, x0, x1, x2, f0, f1, b,
                          gfactor=gfactor, glimit=glimit, rtol=rtol,
                          atol=atol, maxiter=maxiter)

    if isfinite(b) and x0 >= b + rtol*abs(b) + atol:
        x0 = b

    x2 = x1 + (x1 - x0) * gfactor
    return _downhill3(f, x0, x1, x2, f0, f1, a,
                      gfactor=gfactor, glimit=glimit, rtol=rtol,
                      atol=atol, maxiter=maxiter)


def _parabolic_step(x0, x1, x2, f0, f1, f2):
    # x0: best
    # x1: second best
    # x2: third best

    r = (x0 - x1) * (f0 - f2)
    q = (x0 - x2) * (f0 - f1)
    p = (x0 - x2) * q - (x0 - x1) * r
    q = 2.0 * (q - r)
    if 0.0 < q:
        p = - p
    q = abs(q)

    if abs(p) < abs(0.5*q*r):
        d = p / q
    else:
        d = 0

    return d
