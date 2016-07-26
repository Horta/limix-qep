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
    def e(self):
        if self._e is None:
            self.initialize_hyperparams()
        return self._e

    @e.setter
    def e(self, value):
        self.clear_cache('_lml_components')
        self.clear_cache('_L')
        self.clear_cache('_update')
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
        # golden ratio
        gs = 0.5 * (3.0 - np.sqrt(5.0))
        r_left = 1e-2
        curr_r = self.e / (1 + self.e)
        r_right = (curr_r + r_left * gs - r_left) / gs
        r_right = min(r_right, 0.99)
        if r_right <= r_left:
            r_right = min(r_left + 1e-3, 0.99)

        r_bounds = (r_left, r_right)

        self._logger.debug("r bound: (%.5f, %.5f)", r_left, r_right)

        return r_bounds

    def _nlml(self, r, h2):
        # self._logger.debug("Evaluating for r:%e, h2:%e", r, h2)
        print("Evaluating for r:%.5f, h2:%.5f" %(r, h2))
        self.e = r / (1 - r)
        self.var = h2 * self.e / (1 - h2)
        self._optimize_beta()
        return -self.lml()

    def _optimize_h2(self, h2):
        # self._logger.debug("Evaluating for h2:%.5f", h2)
        print("Evaluating for h2:%.5f" % h2)

        opt = dict(xatol=R_EPS)
        minimize_scalar(self._nlml, bounds=self._r_bounds(),
                        options=opt, method='Bounded', args=(h2,))
        print('----------------------------------------------------------------')
        return -self.lml()

    def optimize(self):

        from time import time

        start = time()

        self._logger.debug("Start of optimization.")
        # self._logger.debug("Initial parameters: h2=%.5f, var=%.5f, e=%.5f," +
        #                    " beta=%s.", self.heritability, self.var, self.e,
        #                    bytes(self.beta))
        print("Initial parameters: h2=%.5f, var=%.5f, e=%.5f, beta=%s." % (self.heritability, self.var, self.e, str(self.beta)))

        opt = dict(xatol=HYPERPARAM_EPS)

        self._h2_bounds()
        r = minimize_scalar(self._optimize_h2, options=opt,
                            bounds=self._h2_bounds(),
                            method='Bounded')
        nfev = r.nfev

        # fun_cost = self._create_fun_cost_both(opt_beta)
        # opt = dict(xatol=_ALPHAS1_EPS, maxiter=_NALPHAS1, disp=disp)
        # res = optimize.minimize_scalar(fun_cost, options=opt,
        #                                 bounds=(_ALPHAS1_EPS,
        #                                           1-_ALPHAS1_EPS),
        #                                   method='Bounded')
        # alpha1 = res.x
        # alpha0 = self._best_alpha0(alpha1, opt_beta)[0]
        #
        # (self.sigg2, self.delta) = _alphas2hyperparams(alpha0, alpha1)
        #
        # nfev = 0
        # if opt_var:
        #     opt = dict(xatol=HYPERPARAM_EPS, disp=disp)
        #
        #     r = minimize_scalar(self._nlml, options=opt,
        #                         bounds=self._h2_bounds(),
        #                         method='Bounded', args=opt_beta)
        #     self.var = self.h2tovar(r.x)
        #     self._logger.debug("Optimizer message: %s.", r.message)
        #     if r.status != 0:
        #         self._logger.warn("Optimizer failed with status %d.", r.status)
        #
        #     nfev = r.nfev
        #
        # if opt_beta:
        #     self._optimize_beta()
        #
        # self._logger.debug("Final parameters: h2=%.5f, var=%.5f, beta=%s",
        #                    self.heritability, self.var, bytes(self.beta))
        #
        self._logger.debug("End of optimization (%.3f seconds" +
                           ", %d function calls).", time() - start, nfev)


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
