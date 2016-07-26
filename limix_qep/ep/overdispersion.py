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

from .util import normal_bracket

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

    def initialize_hyperparams(self):
        raise NotImplementedError

    ############################################################################
    ############################################################################
    ######################## Getters and setters ###############################
    ############################################################################
    ############################################################################
    @property
    def noise_ratio(self):
        e = self.environmental_variance
        return e / (e + 1)

    @noise_ratio.setter
    def noise_ratio(self, value):
        value = min(value, 1-1e-7)
        self.environmental_variance = value / (1 - value)

    @property
    def environmental_variance(self):
        if self._e is None:
            self.initialize_hyperparams()
        return self._e

    @environmental_variance.setter
    def environmental_variance(self, v):
        self.clear_cache('_lml_components')
        self.clear_cache('_L')
        self.clear_cache('_update')
        self.clear_cache('_A0')
        self.clear_cache('diagK')
        self.clear_cache('K')
        self._e = max(v, 1e-7)

    @property
    def instrumental_variance(self):
        return 1.

    def _noise_ratio_cost(self, r):
        print("  - Evaluating for ratio: %.5f." % r)
        self.e = r / (1 - r)
        h2 = self.heritability
        self.var = h2 * self.e / (1 - h2)
        self._optimize_beta()
        c = -self.lml()
        print("  - Cost: %.5f" % c)
        return c

    def _h2_cost(self, h2):
        # self._logger.debug("Evaluating for h2: %e.", h2)
        print("- Evaluating for h2: %.5f." % h2)
        var = self.h2tovar(h2)
        self.var = var

        r = self.e / (self.e + 1)
        p = min(abs(self._previous_h2 - h2)*100, 1)
        bracket = normal_bracket(r, p)
        print("    Ratio bracket: %s." % str(bracket))
        try:
            res = minimize_scalar(self._noise_ratio_cost,
                                bracket=bracket,
                                tol=R_EPS, method='Brent')
        except ValueError:
            fa = self._noise_ratio_cost(bracket[0])
            fc = self._noise_ratio_cost(bracket[2])
            if fa < fc:
                r = bracket[0]
            else:
                r = bracket[2]
        else:
            r = res.x

        self._previous_h2 = h2
        self.e = r / (1 - r)
        return -self.lml()

    def optimize(self):
        self._previous_h2 = self.heritability
        super(OverdispersionEP, self).optimize()

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
