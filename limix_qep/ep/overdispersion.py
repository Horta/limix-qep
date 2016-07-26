from __future__ import absolute_import
from __future__ import division

import logging

import numpy as np

from limix_math.linalg import sum2diag
from limix_math.dist.beta import isf as bisf

from hcache import cached
from .dists import Joint

from .config import HYPERPARAM_EPS
from .config import R_EPS

from scipy.optimize import minimize_scalar

from limix_qep.special.nbinom_moms import moments_array3, init

from numpy import var as variance
from numpy import exp
from numpy import log

from .util import golden_bracket

from .base import EP

_NALPHAS0 = 100
_ALPHAS0 = bisf(0.55, 2., 1.-
                np.linspace(0, 1, _NALPHAS0+1, endpoint=False)[1:])
_NALPHAS1 = 30
_ALPHAS1_EPS = 1e-3


# K = Q (v S + e I) Q.T
class OverdispersionEP(EP):
    """
    .. math::
        K = v Q S Q.T
        M = svd_U svd_S svd_V.T
        \\tilde \\beta = svd_S^{1/2} svd_V.T \\beta
        \\tilde M = svd_U svd_S^{1/2} \\tilde \\beta
        m = M \\beta
        m = \\tilde M \\tilde \\beta
    """
    def __init__(self, M, Q, S, QSQt=None):
        super(OverdispersionEP, self).__init__(M, Q, S, QSQt=QSQt)
        self._logger = logging.getLogger(__name__)

        self._joint = Joint(Q, S)
        self._e = None

        # useful for optimization only
        # that is a hacky
        self.__flip_noise_ratio = False

    def initialize_hyperparams(self):
        raise NotImplementedError

    ############################################################################
    ############################################################################
    ######################## Getters and setters ###############################
    ############################################################################
    ############################################################################
    @property
    def e(self):
        if self._e is None:
            self.initialize_hyperparams()
        return self._e

    @e.setter
    def e(self, value):
        self.clear_cache('_lml_components')
        self.clear_cache('_L')
        self.clear_cache('_update')
        self.clear_cache('_A0')
        self.clear_cache('diagK')
        self.clear_cache('K')
        self._e = max(value, 1e-7)

    @property
    def environmental_variance(self):
        return self.e

    @environmental_variance.setter
    def environmental_variance(self, v):
        self.e = v

    @property
    def instrumental_variance(self):
        return 1.

    def _r_bounds(self):
        r = self.e / (1 + self.e)
        if self.__flip_noise_ratio:
            r = 1 - r
        bounds = golden_bracket(r)
        return bounds

    def _noise_ratio_cost(self, r, flip):
        if flip:
            r = 1 - r
        print("  - Evaluating for ratio: %.5f." % r)
        self.e = r / (1 - r)
        h2 = self.heritability
        self.var = h2 * self.e / (1 - h2)
        self._optimize_beta()
        return -self.lml()

    def _h2_cost(self, h2, flip):
        if flip:
            h2 = 1 - h2
        # self._logger.debug("Evaluating for h2: %e.", h2)
        print("- Evaluating for h2: %.5f." % h2)
        var = self.h2tovar(h2)
        self.var = var

        opt = dict(xatol=R_EPS)
        r = self.e / (self.e + 1)

        minimize_scalar(self._noise_ratio_cost,
                        bounds=golden_bracket(0.5 - abs(0.5 - r)),
                        options=opt, method='Bounded', args=r > 0.5)

        return -self.lml()

    ############################################################################
    ############################################################################
    ############## Key but Intermediary Matrix Definitions #####################
    ############################################################################
    ############################################################################
    @cached
    def _A0(self):
        """:math:`e \\mathrm I`"""
        return self._e

    @cached
    def _A1(self):
        """:math:`(e \\mathrm I + \\tilde{\\mathrm T}^{-1})^{-1}`"""
        ttau = self._sites.tau
        return ttau / (self.e * ttau + 1)

    @cached
    def _A2(self):
        """:math:`\\tilde{\\mathrm T}^{-1} \\mathrm A_1`"""
        ttau = self._sites.tau
        return 1 / (self.e * ttau + 1)

    @cached
    def K(self):
        """:math:`K = (v Q S Q.T + e I)`"""
        return sum2diag(self.var * self._QSQt(), self.e)

    @cached
    def diagK(self):
        return self.var * self._diagQSQt() + self.e
