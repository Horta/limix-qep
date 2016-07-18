from __future__ import division

from numpy import sqrt
from numpy import exp
from numpy import asarray
from numpy import empty
from numpy import atleast_1d

from limix_math.dist.norm import logcdf

from limix_qep.special.nbinom_moms import moments_array3


class BernoulliPredictor(object):
    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov

    def logpdf(self, y):
        ind = 2*y - 1
        return logcdf(ind * self._mean / sqrt(1 + self._cov))

    def pdf(self, y):
        return exp(self.logpdf(y))

class BinomialPredictor(object):
    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov
        self._tau = 1./cov
        self._eta = self._tau * mean

    def logpdf(self, y, ntrials):
        # ind = 2*y - 1
        # return logcdf(ind * self._mean / np.sqrt(1 + self._cov))
        y = asarray(y, float)
        ntrials = asarray(ntrials, float)
        y = atleast_1d(y)
        ntrials = atleast_1d(ntrials)
        n = len(y)
        lmom0 = empty(n)
        mu1 = empty(n)
        var2 = empty(n)

        moments_array3(ntrials, y, self._eta, self._tau, lmom0, mu1, var2)
        return lmom0

    def pdf(self, y, ntrials):
        return exp(self.logpdf(y, ntrials))
