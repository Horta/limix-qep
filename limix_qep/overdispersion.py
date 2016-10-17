from __future__ import absolute_import, division, unicode_literals

import logging

from hcache import Cached, cached

from limix_math.linalg import (cho_solve, ddot, dotd, economic_svd, solve,
                               sum2diag)

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

        nsamples = M.shape[0]
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

    def _joint_update(self):
        A = self._A()
        C = self._C()
        K = self.K()
        m = self.m()
        teta = self._sitelik_eta
        QBiQt = self._Q0BiQ0t()

        jtau = self._joint_tau
        jeta = self._joint_eta

        diagCK = C * K.diagonal()
        QBiQtA = ddot(QBiQt, A, left=False)
        jtau[:] = 1 / (diagCK - C * dotd(QBiQtA, K))

        Kteta = K.dot(teta)
        jeta[:] = C * m - C * QBiQtA.dot(m) + C * Kteta - C * QBiQtA.dot(Kteta)
        jeta *= jtau

# def _joint_update(self):
#     K = self.K()
#     m = self.m()
#     A2 = self._A2()
#     QB1Qt = self._Q0B1Q0t()
#
#     jtau = self._joint_tau
#     jeta = self._joint_eta
#
#     diagK = K.diagonal()
#     QB1QtA1 = ddot(QB1Qt, self._A1(), left=False)
#     jtau[:] = 1 / (A2 * diagK - A2 * dotd(QB1QtA1, K))
#
#     Kteta = K.dot(self._sitelik_eta)
#     jeta[:] = A2 * (m - QB1QtA1.dot(m) + Kteta - QB1QtA1.dot(Kteta))
#     jeta *= jtau
