from __future__ import division

import logging

from numpy import clip, sqrt


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
    from numpy import logical_and
    from numpy import logical_not
    from numpy import finfo
    from numpy import empty_like

    a = sp.special.beta(nsuccesses + 1 + 1, ndraws - nsuccesses + 1)
    b = sp.special.beta(nsuccesses + 1, ndraws - nsuccesses + 1)

    r = empty_like(a)
    ok = b > 0
    r[ok] = a[ok] / b[ok]
    nok = logical_not(ok)
    r[logical_and(nok, nsuccesses / ndraws < 0.5)] = finfo(float).eps
    r[logical_and(nok, nsuccesses / ndraws >= 0.5)] = 1 - finfo(float).eps
    return r


def create_binomial(nsamples, nfeatures, ntrials, sigg2=0.8, delta=0.2,
                    sige2=1., seed=None):
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    if np.isscalar(ntrials):
        ntrials = np.full(nsamples, ntrials, dtype=int)
    else:
        ntrials = np.asarray(ntrials, int)

    X = np.random.randn(nsamples, nfeatures)
    X -= np.mean(X, 0)
    X /= np.std(X, 0)
    X /= np.sqrt(nfeatures)

    u = np.random.randn(nfeatures) * np.sqrt(sigg2)

    u -= np.mean(u)
    u /= np.std(u)
    u *= np.sqrt(sigg2)

    g1 = np.dot(X, u)
    g2 = np.random.randn(nsamples)
    g2 -= np.mean(g2)
    g2 /= np.std(g2)
    g2 *= np.sqrt(sigg2 * delta)

    g = g1 + g2

    E = np.random.randn(nsamples, np.max(ntrials))
    E *= np.sqrt(sige2)

    Z = g[:, np.newaxis] + E

    Z[Z > 0.] = 1.
    Z[Z <= 0.] = 0.

    y = np.empty(nsamples)
    for i in range(y.shape[0]):
        y[i] = np.sum(Z[i, :ntrials[i]])

    return (y, X)

# K = \sigma_g^2 Q (S + \delta I) Q.T
