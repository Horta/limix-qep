from __future__ import absolute_import
from __future__ import division

import logging

from numpy import clip
from numpy import full
from numpy import asarray
from numpy import isscalar
from numpy import isfinite
from numpy import all as all_
from numpy import set_printoptions

from numpy.linalg import lstsq

from limix_math import issingleton

from lim.genetics import FastLMM

from .overdispersion import OverdispersionEP

from .util import ratio_posterior
from .util import greek_letter
from .util import summation_symbol


class PoissonEP(OverdispersionEP):

    def __init__(self, y, M, Q0, Q1, S0, Q0S0Q0t=None):
        super(PoissonEP, self).__init__(M, Q0, Q1, S0, Q0S0Q0t=Q0S0Q0t)
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

        from .. import LikNormMoments
        self._moments = LikNormMoments(100)

    def initialize_hyperparams(self):
        from scipy.stats import norm
        y = self._y

        Q0 = self._Q0
        Q1 = self._Q1
        covariates = self._M
        flmm = FastLMM(y, covariates, QS=[[Q0, Q1], [self._S0]])
        flmm.learn()
        gv = flmm.genetic_variance
        nv = flmm.environmental_variance
        h2 = gv / (gv + nv)
        h2 = clip(h2, 0.01, 0.9)

        mean = flmm.mean
        self._tbeta = lstsq(self._tM, full(len(y), mean))[0]
        self.environmental_variance = self.instrumental_variance
        self.pseudo_heritability = h2

    def _tilted_params(self):
        y = self._y
        ctau = self._cavs.tau
        ceta = self._cavs.eta
        lmom0 = self._loghz
        self._moments.compute(y, ceta, ctau, lmom0, self._hmu, self._hvar)

    # \hat z
    def _compute_hz(self):
        self._tilted_params()

    # def model(self):
    #     covariate_effect_sizes = self.beta
    #     fixed_effects_variance = self.beta.var()
    #     real_variance = self.real_variance
    #     noise_ratio = self.noise_ratio
    #     genetic_variance = self.genetic_variance
    #     environmental_variance = self.environmental_variance
    #     instrumental_variance = self.instrumental_variance
    #     environmental_genetic_ratio = self.environmental_genetic_ratio
    #     genetic_ratio = self.genetic_ratio
    #     heritability = self.heritability
    #     return BinomialModel(covariate_effect_sizes, fixed_effects_variance,
    #                          real_variance, noise_ratio, genetic_variance,
    #                          environmental_variance, instrumental_variance,
    #                          environmental_genetic_ratio, genetic_ratio,
    #                          heritability)
