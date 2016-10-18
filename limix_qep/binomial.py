from __future__ import absolute_import, division

import logging

from numpy import (all, asarray, clip, exp, full, isfinite, isscalar, log,
                   ones, pi, set_printoptions, sqrt)
from numpy.linalg import lstsq

from limix_math import issingleton
from limix_qep.liknorm import LikNormMoments

from .ep import EP


class BinomialEP(EP):

    def __init__(self, nsuccesses, ntrials, M, Q0, Q1, S0, Q0S0Q0t=None):
        super(BinomialEP, self).__init__(M, Q0, S0, Q0S0Q0t=Q0S0Q0t)
        self._logger = logging.getLogger(__name__)

        nsuccesses = asarray(nsuccesses, float)

        if isscalar(ntrials):
            ntrials = full(len(nsuccesses), ntrials, dtype=float)
        else:
            ntrials = asarray(ntrials, float)

        self._nsuccesses = nsuccesses
        self._ntrials = ntrials

        if issingleton(nsuccesses):
            raise ValueError("The phenotype array has a single unique value" +
                             " only.")

        if not all(isfinite(nsuccesses)):
            raise ValueError("There are non-finite numbers in phenotype.")

        assert nsuccesses.shape[0] == M.shape[
            0], 'Number of individuals mismatch.'
        assert nsuccesses.shape[0] == Q0.shape[
            0], 'Number of individuals mismatch.'
        assert nsuccesses.shape[0] == Q1.shape[
            0], 'Number of individuals mismatch.'

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

        nsuccesses = self._nsuccesses
        ntrials = self._ntrials

        latent = nsuccesses / ntrials
        latent = latent / latent.std()
        latent -= latent.mean()

        Q0 = self._Q0
        Q1 = self._Q1
        S0 = self._S0
        covariates = self._M
        flmm = FastLMM(latent, covariates, QS=((Q0, Q1), (S0,)))
        flmm.learn()
        gv = flmm.genetic_variance
        nv = flmm.environmental_variance
        h2 = gv / (gv + nv)
        h2 = clip(h2, 0.01, 0.9)

        mean = flmm.mean
        self._tbeta = lstsq(self._tM, full(len(ntrials), mean))[0]
        self.heritability = h2

    @property
    def environmental_variance(self):
        return (pi * pi) / 3

    def _tilted_params(self):
        nsuccesses = self._nsuccesses
        ntrials = self._ntrials
        ctau = self._cav_tau
        ceta = self._cav_eta
        lmom0 = self._loghz

        self._moments.binomial(nsuccesses, ntrials, ceta,
                               ctau, lmom0, self._hmu, self._hvar)
        lmom0_in, lmom0_ax = min(lmom0), max(lmom0)
        # print(lmom0_in, lmom0_ax, (ceta / ctau).min(), (ceta / ctau).max())
        # import numpy as np
        # idx = np.argsort(lmom0)
        # print("1: %s" % str(lmom0[idx[:10]]))
        # print("2: %s" % str(nsuccesses[idx[:10]]))
        # print("3: %s" % str(ntrials[idx[:10]]))
        # print("4: %s" % str(ctau[idx[:10]]))
        # print("5: %s" % str((ceta / ctau)[idx[:10]]))

        # print("1: %s" % str(lmom0[:5]))
        # print("2: %s" % str(self._hmu[:5]))
        # print("3: %s" % str(self._hvar[:5]))
        # print("4: %s" % str(nsuccesses[:5]))
        # print("5: %s" % str(ntrials[:5]))
        # print("6: %s" % str(ctau[:5]))
        # print("7: %s" % str((ceta / ctau)[:5]))
