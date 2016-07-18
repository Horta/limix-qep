from __future__ import absolute_import

import logging

from numpy import log
from numpy import sqrt
from numpy import exp
from numpy import full
from numpy.linalg import lstsq

from limix_math.dist.norm import logcdf
from limix_math.dist.norm import logpdf

import limix_ext as lxt

from .base import EP

class BernoulliPredictor(object):
    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov

    def logpdf(self, y):
        ind = 2*y - 1
        return logcdf(ind * self._mean / sqrt(1 + self._cov))

    def pdf(self, y):
        return exp(self.logpdf(y))

# K = \sigma_g^2 Q S Q.T
class BernoulliEP(EP):
    def __init__(self, y, M, Q, S, QSQt=None):
        super(BernoulliEP, self).__init__(y, M, Q, S, QSQt=QSQt)

        self._logger = logging.getLogger(__name__)

        self._y11 = 2. * self._y - 1.0

    def _init_var(self):
        y = self._y
        tM = self._tM
        ratio = sum(y) / float(y.shape[0])
        self._var = max(1e-3, lxt.lmm.h2(y, tM, self._QSQt(), ratio))

    def _init_beta(self):
        from scipy.stats import norm
        y = self._y
        ratio = sum(y) / float(y.shape[0])
        f = full(len(y), norm(0, 1).isf(1 - ratio))
        self._tbeta = lstsq(self._tM, f)[0]

    def predict(self, m, var, covar):
        (mu, sig2) = self._posterior_normal(m, var, covar)
        return BernoulliPredictor(mu, sig2)

    def _tilted_params(self):
        b = sqrt(self._cavs.tau**2 + self._cavs.tau)
        lb = log(b)
        c = self._y11 * self._cavs.eta / b
        lcdf = self._loghz
        lcdf[:] = logcdf(c)
        lpdf = logpdf(c)
        mu = self._cavs.eta / self._cavs.tau + self._y11 * exp(lpdf - (lcdf + lb))

        sig2 = 1./self._cavs.tau - exp(lpdf - (lcdf + 2*lb)) * (c + exp(lpdf - lcdf))

        return (mu, sig2)

    # \hat z
    def _compute_hz(self):
        b = sqrt(self._cavs.tau**2 + self._cavs.tau)
        c = self._y11 * self._cavs.eta / b
        self._loghz[:] = logcdf(c)
