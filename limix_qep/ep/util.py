from __future__ import division

import logging

from numpy import sqrt
from numpy import clip


def make_sure_reasonable_conditioning(S):
    max_cond = 1e6
    cond = S.max() / S.min()
    if cond > max_cond:
        eps = (max_cond * S.min() - S.max()) / (1 - max_cond)
        logger = logging.getLogger(__name__)
        logger.warn("The covariance matrix's conditioning number" +
                    " is too high: %e. Summing %e to its eigenvalues and " +
                    "renormalizing for a better conditioning number.",
                    cond, eps)
        m = S.mean()
        S += eps
        S *= m / S.mean()


def shall_use_low_rank_tricks(nsamples, nfeatures):
    return nsamples - nfeatures > 1


def golden_bracket(x):
    # golden ratio
    gs = 0.5 * (3.0 - sqrt(5.0))

    assert x <= 0.5

    left = 1e-3
    right = (x + left * gs - left) / gs
    right = min(right, 1 - 1e-3)

    return (left, right)


def normal_bracket(x, ratio=1 / 2):
    assert x >= 0 and x <= 1

    tleft = x - ratio / 2
    tright = x + ratio / 2

    right = tright - min(tleft, 0)
    left = tleft - max(tright - 1, 0)

    ac = tuple(clip([left, right], 1e-3, 1 - 1e-3))
    if x > ac[0] and x < ac[1]:
        return (ac[0], x, ac[1])
    return (ac[0], ac[0] + (ac[1] - ac[0]) / 2, ac[1])


def ratio_posterior(nsuccesses, ndraws):
    """ :math:`\\frac{\\int_0^1 r \\binom{n}{s} r^s (1 - r)^{n-s} dr}{\\int_0^1
        \\binom{n}{s} r^s (1 - r)^{n-s} dr}`
    """
    import scipy as sp
    import scipy.special
    a = sp.special.beta(nsuccesses + 1 + 1, ndraws - nsuccesses + 1)
    b = sp.special.beta(nsuccesses + 1, ndraws - nsuccesses + 1)
    return a / b


def greek_letter(name):
    names = ['alpha', 'beta', 'bla1', 'bla2', 'epsilon', ]
    for i in range(len(names)):
        if name == names[i]:
            return unichr(0x3b1 + i).encode('utf-8')


def summation_symbol():
    return unichr(0x2211).encode('utf-8')
