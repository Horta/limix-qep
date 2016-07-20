from __future__ import absolute_import

import logging

from numpy import log
from numpy import sqrt
from numpy import exp
from numpy import full
from numpy import asarray
from numpy import isfinite
from numpy import all as all_
from numpy.linalg import lstsq

from limix_math.array import issingleton
from limix_math.dist.norm import logcdf
from limix_math.dist.norm import logpdf

from lim.genetics import FastLMM
from lim.genetics.heritability import bern2lat_correction

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
    def __init__(self, y, M, Q0, Q1, S0, QSQt=None):
        super(BernoulliEP, self).__init__(M, Q0, S0, QSQt=QSQt)
        self._logger = logging.getLogger(__name__)

        y = asarray(y, float)
        self._y = y

        if issingleton(y):
            raise ValueError("The phenotype array has a single unique value" +
                             " only.")

        if not all_(isfinite(y)):
            raise ValueError("There are non-finite numbers in phenotype.")

        assert y.shape[0] == M.shape[0], 'Number of individuals mismatch.'
        assert y.shape[0] == Q0.shape[0], 'Number of individuals mismatch.'
        assert y.shape[0] == Q1.shape[0], 'Number of individuals mismatch.'

        self._y11 = 2. * self._y - 1.0
        self._Q1 = Q1

    def initialize_hyperparams(self):
        from scipy.stats import norm
        y = self._y
        ratio = sum(y) / float(len(y))
        latent_mean = norm(0, 1).isf(1 - ratio)
        latent = y / y.std()
        latent = latent - latent.mean() + latent_mean

        Q0 = self._Q
        Q1 = self._Q1
        flmm = FastLMM(full(len(y), latent), QS=[[Q0, Q1], [self._S]])
        flmm.learn()
        gv = flmm.genetic_variance
        nv = flmm.noise_variance
        h2 = gv / (gv + nv)
        h2 = bern2lat_correction(h2, ratio, ratio)

        offset = flmm.offset
        self._var = h2/(1-h2)
        self._tbeta = lstsq(self._tM, full(len(y), offset))[0]

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
        self._hmu[:] = self._cavs.eta / self._cavs.tau \
                       + self._y11 * exp(lpdf - (lcdf + lb))

        self._hvar[:] = 1./self._cavs.tau\
                        - exp(lpdf - (lcdf + 2*lb)) * (c + exp(lpdf - lcdf))

    # \hat z
    def _compute_hz(self):
        b = sqrt(self._cavs.tau**2 + self._cavs.tau)
        c = self._y11 * self._cavs.eta / b
        self._loghz[:] = logcdf(c)
