from __future__ import absolute_import, division, unicode_literals

import logging
from time import time

from hcache import Cached, cached

from limix_math.linalg import (cho_solve, ddot, dotd, economic_svd, solve,
                               sum2diag)

from ._optimize import find_minimum
from .base import EPBase


# K = v Q ((1-delta)*S + delta*I) Q.T
# sigma2_epsilon = v * delta
# sigma2_b = v * (1-delta)

class OverdispersionEP(EPBase):
    """
    .. math::
        K = v Q ((1-delta)*S + delta*I) Q.T
        M = svd_U svd_S svd_V.T
        \\tilde \\beta = svd_S^{1/2} svd_V.T \\beta
        \\tilde M = svd_U svd_S^{1/2} \\tilde \\beta
        m = M \\beta
        m = \\tilde M \\tilde \\beta
    """

    def __init__(self, M, Q0, Q1, S0, Q0S0Q0t=None):
        super(OverdispersionEP, self).__init__(M, Q0, S0, Q0S0Q0t=Q0S0Q0t)
        self._logger = logging.getLogger(__name__)

        self._Q1 = Q1
        self._delta = 0.5

    @cached
    def _A(self):
        ttau = self._sitelik_tau
        s2 = self.sigma2_epsilon
        return ttau / (ttau * s2 + 1)

    @cached
    def _C(self):
        ttau = self._sitelik_tau
        s2 = self.sigma2_epsilon
        return 1 / (ttau * s2 + 1)

    @property
    def sigma2_epsilon(self):
        return self._v * self._delta

    @sigma2_epsilon.setter
    def sigma2_epsilon(self, v):
        self._delta = v / self._v

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, v):
        self.clear_cache('_lml_components')
        self.clear_cache('_L')
        self.clear_cache('_Q0BiQ0t')
        self.clear_cache('_A')
        self.clear_cache('_C')
        self.clear_cache('_update')
        self._delta = v

    @cached
    def K(self):
        return sum2diag(self.sigma2_b * self._Q0S0Q0t(), self.sigma2_epsilon)

    @cached
    def diagK(self):
        return self.sigma2_b * self._diagQ0S0Q0t() + self.sigma2_epsilon

    def optimize(self):

        self._logger.info("Start of optimization.")

        def function_cost(v):
            self.sigma2_b = v

            def function_cost_delta(delta):
                self.delta = delta
                self._optimize_beta()
                return -self.lml()

            delta, nfev = find_minimum(function_cost_delta, self.delta, a=1e-4,
                                       b=1 - 1e-4, rtol=0, atol=1e-6)

            self.delta = delta
            self._optimize_beta()
            return -self.lml()

        start = time()
        v, nfev = find_minimum(function_cost, self.sigma2_b, a=1e-4,
                               b=1e4, rtol=0, atol=1e-6)

        self.sigma2_b = v

        self._optimize_beta()
        elapsed = time() - start

        msg = "End of optimization (%.3f seconds, %d function calls)."
        self._logger.info(msg, elapsed, nfev)

    def _joint_update(self):
        A = self._A()
        C = self._C()
        K = self.K()
        m = self.m()
        teta = self._sitelik_eta
        QBiQt = self._Q0BiQ0t()

        jtau = self._joint_tau
        jeta = self._joint_eta

        diagK = K.diagonal()
        QBiQtA = ddot(QBiQt, A, left=False)
        jtau[:] = 1 / (diagK - dotd(QBiQtA, K))

        Kteta = K.dot(teta)
        jeta[:] = m - QBiQtA.dot(m) + Kteta - QBiQtA.dot(Kteta)
        jeta *= jtau
        jtau /= C
