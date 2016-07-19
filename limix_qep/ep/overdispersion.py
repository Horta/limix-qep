from __future__ import absolute_import
import logging
import numpy as np
from numpy import dot
# from scipy.linalg import lu_factor
from limix_math.linalg import ddot
from limix_math.linalg import sum2diag
from limix_math.linalg import dotd
# from limix_math.linalg import sum2diag_inplace
# from limix_math.linalg import solve
from limix_math.linalg import cho_solve
# from limix_math.linalg import lu_solve
from limix_math.linalg import stl
# from limix_math.dist.norm import logpdf
# from limix_math.dist.norm import logcdf
from limix_math.dist.beta import isf as bisf
from hcache import cached
# from limix_qep.special.nbinom_moms import moments_array3, init
# from limix_qep.lik import Binomial, Bernoulli
# from limix_math.array import issingleton
from .dists import OverdispersionJoint

from numpy import var as variance

# from scipy import optimize
from scipy.misc import logsumexp

# from .config import MAX_EP_ITER
# from .config import EP_EPS
# from .config import HYPERPARAM_EPS

from .base import EP


def _is_zero(x):
    return abs(x) < 1e-9

_NALPHAS0 = 100
_ALPHAS0 = bisf(0.55, 2., 1.-
                np.linspace(0, 1, _NALPHAS0+1, endpoint=False)[1:])
_NALPHAS1 = 30
_ALPHAS1_EPS = 1e-3


# K = \sigma_g^2 Q (S + \delta I) Q.T
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
        super(OverdispersionEP, self).__init__(M, Q, S, QSQt=None)
        self._logger = logging.getLogger(__name__)

        self._joint = OverdispersionJoint(Q, S)
        self._delta = None

    def _init_hyperparams(self):
        raise NotImplementedError

    # def _LU(self):
    #     Sd = self.sigg2 * (self._S + self.delta)
    #     Q = self._Q
    #     ttau = self._sites.tau
    #     R = ddot(Sd, Q.T, left=True)
    #     R = ddot(ttau, dot(Q, R), left=True)
    #     sum2diag_inplace(R, 1.0)
    #     return lu_factor(R, overwrite_a=True, check_finite=False)

    # def _LUM(self):
    #     m = self._m
    #     ttau = self._sites.tau
    #     teta = self._sites.eta
    #     LU = self._LU()
    #     return lu_solve(LU, ttau * m + teta)

    # def predict(self, m, var, covar):
    #     m = np.atleast_1d(m)
    #     var = np.atleast_2d(var)
    #     covar = np.atleast_2d(covar)
    #
    #     if isinstance(self._outcome_type, Binomial):
    #         return self._predict_binom(m, var, covar)
    #
    #     # if covar.ndim == 1:
    #     #     assert isnumber(var) and isnumber(m)
    #     # elif covar.ndim == 2:
    #     #     assert len(var) == covar.shape[0]
    #     # else:
    #     #     raise ValueError("covar has a wrong layout.")
    #
    #     A1 = self._A1()
    #     L = self._L()
    #     Q = self._Q
    #     mu = m + dot(covar, self._AtmuLm())
    #
    #     A1cov = ddot(A1, covar.T, left=True)
    #     part3 = dotd(A1cov.T, dot(Q, cho_solve(L, dot(Q.T, A1cov))))
    #     sig2 = var.diagonal() - dotd(A1cov.T, covar.T) + part3
    #
    #     if isinstance(self._outcome_type, Bernoulli):
    #         return BernoulliPredictor(mu, sig2)
    #     else:
    #         return BinomialPredictor(mu, sig2)
    #     #
    #     # if covar.ndim == 1:
    #     #     p = dict()
    #     #     p[1] = np.exp(logcdf(mu / np.sqrt(1 + sig2)))
    #     #     p[0] = 1 - p[1]
    #     # else:
    #     #     v = np.exp(logcdf(mu / np.sqrt(1 + sig2)))
    #     #     p = [dict([(0, 1-vi), (1, vi)]) for vi in v]
    #     #
    #     # return p

    # def _predict_binom(self, m, var, covar):
    #     m = np.atleast_1d(m)
    #     var = np.atleast_2d(var)
    #     covar = np.atleast_2d(covar)
    #
    #     mu = m + covar.dot(self._LUM())
    #
    #     LU = self._LU()
    #     ttau = self._sites.tau
    #
    #     sig2 = var.diagonal() - dotd(covar, lu_solve(LU, ddot(ttau, covar.T, left=True)))
    #
    #     return BinomialPredictor(mu, sig2)

    @property
    def delta(self):
        if self._delta is None:
            self._init_hyperparams()
        return self._delta

    @delta.setter
    def delta(self, value):
        self.clear_cache('_lml_components')
        self.clear_cache('_L')
        self.clear_cache('_vardotdQSQt')
        self.clear_cache('_update')
        self.clear_cache('_AtmuLm')

        self.clear_cache('_QtAQ')
        # self.clear_cache('_B')
        # self.clear_cache('_update')
        # self.clear_cache('_AtmuLm')

        self._delta = max(value, 1e-4)


    @cached
    def K(self):
        """:math:`K = v (Q S Q.T + \\delta I)`"""
        return sum2diag(self.var * self._QSQt(), self.var * self.delta)

    ############################################################################
    ############################################################################
    ######################## Getters and setters ###############################
    ############################################################################
    ############################################################################
    def h2(self):
        var = self.var
        varc = variance(self.m())
        delta = self.delta
        return var / (var + var*delta + varc + 1.)

    def h2tovar(self, h2):
        varc = variance(self.m())
        delta = self.delta
        return h2 * (1 + varc) / (1 - h2 - delta*h2)

    # @property
    # def delta(self):
    #     return self._delta
    # @delta.setter
    # def delta(self, value):
    #     if isinstance(self._outcome_type, Bernoulli):
    #         assert value == 0.
    #     self.clear_cache('_lml_components')
    #     self.clear_cache('_L')
    #     self.clear_cache('_QtAQ')
    #     # self.clear_cache('_B')
    #     self.clear_cache('_update')
    #     self.clear_cache('_AtmuLm')
    #     self._delta = value

    @cached
    def _lml_components(self):
        self._update()
        Q = self._Q
        S = self._S
        m = self._m
        var = self.var
        ttau = self._sites.tau
        teta = self._sites.eta
        ctau = self._cavs.tau
        ceta = self._cavs.eta
        cmu = self._cavs.mu
        A = self._A()

        varS = var * S
        tctau = ttau + ctau
        A1m = A1*m

        L = self._L()

        p1 = - np.sum(np.log(np.diagonal(L)))
        p1 += - 0.5 * np.log(varS).sum()
        p1 += 0.5 * np.log(A1).sum()

        p3 = np.sum(teta*teta/self._iA0T())
        A0A0pT_teta = teta / self._iAAT()
        QtA0A0pT_teta = dot(Q.T, A0A0pT_teta)
        L = self._L()
        L_0 = stl(L, QtA0A0pT_teta)
        p3 += dot(L_0.T, L_0)
        vv = 2*np.log(np.abs(teta))
        p3 -= np.exp(logsumexp(vv - np.log(tctau)))
        p3 *= 0.5

        p4 = 0.5 * np.sum(ceta * (ttau * cmu - 2*teta) / tctau)

        f0 = None
        L_1 = cho_solve(L, QtA0A0pT_teta)
        f0 = A1 * dot(Q, L_1)
        p5 = dot(m, A0A0pT_teta) - dot(m, f0)

        p6 = - 0.5 * np.sum(m * A1m) +\
            0.5 * dot(A1m, dot(Q, cho_solve(L, dot(Q.T, A1m))))

        p7 = 0.5 * (np.log(ttau + ctau).sum() - np.log(ctau).sum())

        p7 -= 0.5 * np.log(ttau).sum()

        p8 = self._loghz.sum()

        return (p1, p3, p4, p5, p6, p7, p8, f0, A0A0pT_teta)

    def lml(self):
        (p1, p3, p4, p5, p6, p7, p8, _, _) = self._lml_components()
        return p1 + p3 + p4 + p5 + p6 + p7 + p8

    # @cached
    # def _update(self):
    #     self._logger.debug('Main EP loop has started.')
    #     m = self._m
    #     Q = self._Q
    #     S = self._S
    #     sigg2 = self.sigg2
    #     delta = self.delta
    #
    #     ttau = self._sites.tau
    #     teta = self._sites.eta
    #
    #     SQt = self._SQt()
    #     sigg2dotdQSQt = self._sigg2dotdQSQt()
    #
    #     # before2 = time()
    #     if not self._params_initialized:
    #         # before3 = time()
    #         outcome_type = self._outcome_type
    #         y = self._y
    #
    #         if not self._use_low_rank_trick():
    #             self._K = self._Q.dot(self._SQt())
    #             if self._sigg2 is None:
    #                 print("Initial h2 sigg2 guess: %.5f %.5f" % (self.h2(), self.sigg2))
    #                 self.sigg2 = max(1e-3, self.h2tosigg2(heritability(y, self._K)))
    #         else:
    #             # tirar isso daqui
    #             if self._sigg2 is None:
    #                 self.sigg2 = self.h2tosigg2(heritability(y, self._Q.dot(self._SQt())))
    #
    #         if isinstance(outcome_type, Bernoulli) and self._beta is None:
    #             ratio = sum(y) / float(y.shape[0])
    #             f = sp.stats.norm(0,1).isf(1-ratio)
    #             self._beta = np.linalg.lstsq(self._M, np.full(y.shape[0], f))[0]
    #         else:
    #             self._beta = np.zeros(self._M.shape[1])
    #             self._joint.initialize(m, sigg2, delta)
    #
    #         # self._time_elapsed['eploop_init'] += time() - before3
    #         # self._calls['eploop_init'] += 1
    #         self._params_initialized = True
    #     else:
    #         self._joint.update(m, sigg2, delta, S, Q, self._L(),
    #                                  ttau, teta, self._A1(), 1./self._iA0T(),
    #                                  sigg2dotdQSQt, SQt, self._K)
    #     i = 0
    #     NMAX = 10
    #     while i < NMAX:
    #         self._logger.debug('Iteration %d.', i)
    #         self._psites.tau[:] = ttau
    #         self._psites.eta[:] = teta
    #
    #         self._logger.debug('step 1')
    #         self._cavs.update(self._joint.tau, self._joint.eta, ttau, teta)
    #         self._logger.debug('step 2')
    #         (hmu, hsig2) = self._tilted_params()
    #         if not np.all(np.isfinite(hsig2)) or np.any(hsig2 == 0.):
    #             raise Exception('Error: not np.all(np.isfinite(hsig2))' +
    #                             ' or np.any(hsig2 == 0.).')
    #         self._logger.debug('step 3')
    #         self._sites.update(self._cavs.tau, self._cavs.eta, hmu, hsig2)
    #         self.clear_cache('_L')
    #         self.clear_cache('_QtAQ')
    #         self.clear_cache('_AtmuLm')
    #         self._joint.update(m, sigg2, delta, S, Q, self._L(),
    #                                  ttau, teta, self._A1(), 1./self._iA0T(),
    #                                  sigg2dotdQSQt, SQt, self._K)
    #         self._logger.debug('step 4')
    #         tdiff = np.abs(self._psites.tau - ttau)
    #         ediff = np.abs(self._psites.eta - teta)
    #         aerr = tdiff.max() + ediff.max()
    #         self._logger.debug('step 5')
    #
    #         if self._psites.tau.min() <= 0. or (0. in self._psites.eta):
    #             rerr = np.inf
    #         else:
    #             rtdiff = tdiff/np.abs(self._psites.tau)
    #             rediff = ediff/np.abs(self._psites.eta)
    #             rerr = rtdiff.max() + rediff.max()
    #
    #         i += 1
    #         self._logger.debug('step 6')
    #         if aerr < 2*_EP_EPS or rerr < 2*_EP_EPS:
    #             break
    #
    #     if i + 1 == _MAX_ITER:
    #         self._logger.warn('Maximum number of EP iterations has'+
    #                           ' been attained.')
    #
    #     self._logger.debug('Main EP loop has finished '+
    #                       'performing %d iterations.', i+1)

    def _best_alpha0(self, alpha1, opt_beta):
        min_cost = np.inf
        alpha0_min = None
        for i in range(_NALPHAS0):
            (var, delta) = _alphas2hyperparams(_ALPHAS0[i], alpha1)

            self.var = var
            self.delta = delta
            self._update()
            if opt_beta:
                self._optimize_beta()

            cost = -self.lml()
            if cost < min_cost:
                min_cost = cost
                alpha0_min = _ALPHAS0[i]

        return (alpha0_min, min_cost)

    def _create_fun_cost_both(self, opt_beta):
        def fun_cost(alpha1):
            (_, min_cost) = self._best_alpha0(alpha1, opt_beta)
            return min_cost
        return fun_cost

    ############################################################################
    ############################################################################
    ############## Key but Intermediary Matrix Definitions #####################
    ############################################################################
    ############################################################################
    def _iA0T(self):
        return 1./(self.var * self.delta) + self._sites.tau

    def _AAA(self):
        vd = self.var * self.delta
        return vd / (1 + vd * self._sites.tau)

    def _iAAT(self):
        return 1. + self._sites.tau * self.var * self.delta

    # def _A0(self):
        # return 1./(self.var * self.delta)

    def _A(self):
        return self._sites.tau / self._iAAT()


    # @cached
    # def _AtmuLm(self):
    #     return self._A1tmuL() - self._A1mL()

    # def _A1tmuL(self):
    #     A1 = self._A1()
    #     L = self._L()
    #     Q = self._Q
    #     A1tmu = A1*self._sites.tau
    #     return A1tmu - A1*dot(Q, cho_solve(L, dot(Q.T, A1tmu)))
    #
    # def _A1mL(self):
    #     m = self._m
    #     A1 = self._A1()
    #     A1m = A1*m
    #     L = self._L()
    #     Q = self._Q
    #     return A1m - A1*dot(Q, cho_solve(L, dot(Q.T, A1m)))


def _alphas2hyperparams(alpha0, alpha1):
    a0 = 1-alpha0
    a1 = 1-alpha1
    var = alpha0 / (a0*a1)
    delta = (a0 * alpha1)/alpha0
    return (var, delta)
