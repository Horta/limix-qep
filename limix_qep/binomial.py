from __future__ import absolute_import

import logging

from numpy import (all, asarray, clip, exp, full, isfinite, isscalar, log,
                   ones, pi, set_printoptions, sqrt)
from numpy.linalg import lstsq

from lim.genetics.heritability import bern2lat_correction
from limix_math import issingleton
from limix_math.special import normal_logcdf, normal_logpdf
from limix_qep.liknorm import LikNormMoments

from .overdispersion import OverdispersionEP


class BinomialEP(OverdispersionEP):

    def __init__(self, nsuccesses, ntrials, M, Q0, Q1, S0, Q0S0Q0t=None):
        super(BinomialEP, self).__init__(M, Q0, Q1, S0, Q0S0Q0t=Q0S0Q0t)
        self._logger = logging.getLogger(__name__)

        y = asarray(nsuccesses, float)

        if isscalar(ntrials):
            ntrials = full(len(y), ntrials, dtype=float)
        else:
            ntrials = asarray(ntrials, float)

        self._y = y
        self._ntrials = ntrials

        if issingleton(y):
            raise ValueError("The phenotype array has a single unique value" +
                             " only.")

        if not all(isfinite(y)):
            raise ValueError("There are non-finite numbers in phenotype.")

        assert y.shape[0] == M.shape[0], 'Number of individuals mismatch.'
        assert y.shape[0] == Q0.shape[0], 'Number of individuals mismatch.'
        assert y.shape[0] == Q1.shape[0], 'Number of individuals mismatch.'

        # self._y11 = 2. * self._y - 1.0
        self._Q1 = Q1

        self._moments = LikNormMoments(350)
        self.initialize()

    @property
    def genetic_variance(self):
        return self.sigma2_b

    @genetic_variance.setter
    def genetic_variance(self, v):
        self.sigma2_b = v

    @property
    def heritability(self):
        total = self.genetic_variance + self.covariates_variance
        total += self.environmental_variance
        return self.genetic_variance / total

    @heritability.setter
    def heritability(self, v):
        t = self.environmental_variance + self.covariates_variance
        self.genetic_variance = t * (v / (1 - v))

    def initialize(self):
        from scipy.stats import norm
        from lim.genetics.core import FastLMM
        y = self._y
        ratio = sum(y) / float(len(y))
        latent_mean = norm(0, 1).isf(1 - ratio)
        latent = y / y.std()
        latent = latent - latent.mean() + latent_mean

        Q0 = self._Q0
        Q1 = self._Q1
        S0 = self._S0
        covariates = self._M
        flmm = FastLMM(full(len(y), latent), covariates, QS=((Q0, Q1), (S0,)))
        flmm.learn()
        gv = flmm.genetic_variance
        nv = flmm.environmental_variance
        h2 = gv / (gv + nv)
        h2 = bern2lat_correction(h2, ratio, ratio)
        h2 = clip(h2, 0.01, 0.9)

        mean = flmm.mean
        self._tbeta = lstsq(self._tM, full(len(y), mean))[0]
        self.heritability = h2

    @property
    def environmental_variance(self):
        return (pi * pi) / 3

    def _tilted_params(self):
        y = self._y
        ctau = self._cav_tau
        ceta = self._cav_eta
        lmom0 = self._loghz
        self._moments.binomial(y, ones(len(y)), ceta,
                               ctau, lmom0, self._hmu, self._hvar)
