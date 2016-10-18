from __future__ import absolute_import, division, unicode_literals

import logging

import scipy.stats as st

from numpy import (all, asarray, clip, exp, full, isfinite, log, ones, pi,
                   sqrt)
from numpy.linalg import lstsq

from lim.genetics import FastLMM
from limix_math import issingleton
from limix_qep.liknorm import LikNormMoments

from .ep import EP

# K = \sigma_g^2 Q S Q.T


class BernoulliEP(EP):

    def __init__(self, success, M, Q0, Q1, S0, Q0S0Q0t=None):
        super(BernoulliEP, self).__init__(M, Q0, S0, Q0S0Q0t=Q0S0Q0t)
        self._logger = logging.getLogger(__name__)

        success = asarray(success, float)
        self._success = success

        if issingleton(success):
            msg = "The phenotype array has a single unique value only."
            raise ValueError(msg)

        if not all(isfinite(success)):
            raise ValueError("There are non-finite numbers in phenotype.")

        msg = 'Number of individuals mismatch.'
        assert success.shape[0] == M.shape[0], msg
        assert success.shape[0] == Q0.shape[0], msg
        assert success.shape[0] == Q1.shape[0], msg

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
        y = self._success
        ratio = sum(y) / float(len(y))
        latent_mean = st.norm(0, 1).isf(1 - ratio)
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
        h2 = clip(h2, 0.01, 0.9)

        mean = flmm.mean
        self._tbeta = lstsq(self._tM, full(len(y), mean))[0]
        self.heritability = h2

    @property
    def environmental_variance(self):
        return (pi * pi) / 3

    def _tilted_params(self):
        y = self._success
        ctau = self._cav_tau
        ceta = self._cav_eta
        lmom0 = self._loghz
        self._moments.binomial(y, ones(len(y)), ceta,
                               ctau, lmom0, self._hmu, self._hvar)
