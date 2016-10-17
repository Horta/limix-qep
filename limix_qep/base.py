from __future__ import absolute_import, division, unicode_literals

import logging
from time import time

from hcache import Cached, cached
from numpy import var as variance
from numpy import (abs, all, any, asarray, atleast_1d, atleast_2d, clip,
                   diagonal, dot, empty, empty_like, errstate, inf, isfinite,
                   log, maximum, set_printoptions, sqrt, sum, zeros,
                   zeros_like)
from numpy.linalg import LinAlgError, multi_dot
from scipy.linalg import cho_factor

from limix_math.linalg import (cho_solve, ddot, dotd, economic_svd, solve,
                               sum2diag)

from ._optimize import find_minimum
from .util import make_sure_reasonable_conditioning

MAX_EP_ITER = 10
EP_EPS = 1e-4
HYPERPARAM_EPS = 1e-5


class EPBase(Cached):
    """
    .. math::
        K = v Q S Q0.T
        M = svd_U svd_S svd_V.T
        \\tilde \\beta = svd_S^{1/2} svd_V.T \\beta
        \\tilde M = svd_U svd_S^{1/2} \\tilde \\beta
        m = M \\beta
        m = \\tilde M \\tilde \\beta
    """

    def __init__(self, M, Q0, S0, Q0S0Q0t=None):
        Cached.__init__(self)
        self._logger = logging.getLogger(__name__)
        self._ep_params_initialized = False

        if not all(isfinite(Q0)) or not all(isfinite(S0)):
            raise ValueError("There are non-finite numbers in the provided" +
                             " eigen decomposition.")

        if S0.min() <= 0:
            raise ValueError("The provided covariance matrix is not" +
                             " positive-definite because the minimum" +
                             " eigvalue is %f." % S0.min())

        make_sure_reasonable_conditioning(S0)

        self._covariate_setup(M)
        self._S0 = S0
        self._Q0 = Q0
        self.__Q0S0Q0t = Q0S0Q0t

        nsamples = M.shape[0]
        self._previous_sitelik_tau = zeros(nsamples)
        self._previous_sitelik_eta = zeros(nsamples)

        self._sitelik_tau = zeros(nsamples)
        self._sitelik_eta = zeros(nsamples)

        self._cav_tau = zeros(nsamples)
        self._cav_eta = zeros(nsamples)

        self._joint_tau = zeros(nsamples)
        self._joint_eta = zeros(nsamples)

        self._v = None
        self.__tbeta = None

        self._loghz = empty(nsamples)
        self._hmu = empty(nsamples)
        self._hvar = empty(nsamples)

    # def fixed_ep(self):
    #
    #     (p1, p3, p4, _, _, p7, p8, p9) = self._lml_components()
    #
    #     lml_const = p1 + p3 + p4 + p7 + p8 + p9
    #
    #     A1 = self._A1()
    #     A2teta = self._A2() * self._sites.eta
    #     Q0B1Q0t = self._Q0B1Q0t()
    #     beta_nom = self._optimal_beta_nom()
    #
    #     return FixedBaseEP(lml_const, A1, A2teta, Q0B1Q0t, beta_nom)

    def _covariate_setup(self, M):
        self._M = M
        SVD = economic_svd(M)
        self._svd_U = SVD[0]
        self._svd_S12 = sqrt(SVD[1])
        self._svd_V = SVD[2]
        self._tM = ddot(self._svd_U, self._svd_S12, left=False)
        self.__tbeta = None

    # def _posterior_normal(self, m, var, covar):
    #     m = atleast_1d(m)
    #     var = atleast_2d(var)
    #     covar = atleast_2d(covar)
    #
    #     A1 = self._A1()
    #     Q0B1Q0t = self._Q0B1Q0t()
    #
    #     mu = m - covar.dot(self._r())
    #
    #     A1covar = ddot(A1, covar.T, left=True)
    #     sig2 = var.diagonal() - dotd(A1covar.T, covar.T)\
    #         + dotd(A1covar.T, Q0B1Q0t.dot(A1covar))
    #
    #     return (mu, sig2)

    def _init_ep_params(self):
        assert self._ep_params_initialized is False
        self._logger.info("EP parameters initialization.")
        self._joint_initialize()
        self._sitelik_initialize()
        self._ep_params_initialized = True

    def initialize(self):
        raise NotImplementedError

    def _joint_initialize(self):
        self._joint_tau[:] = 1 / self.diagK()
        self._joint_eta[:] = self.m()
        self._joint_eta[:] *= self._joint_tau

    def _sitelik_initialize(self):
        self._sitelik_tau[:] = 0.
        self._sitelik_eta[:] = 0.

    @cached
    def K(self):
        """:math:`K = v Q S Q0.T`"""
        return self.genetic_variance * self._Q0S0Q0t()

    @cached
    def diagK(self):
        return self.genetic_variance * self._diagQ0S0Q0t()

    def _diagQ0S0Q0t(self):
        return self._Q0S0Q0t().diagonal()

    @cached
    def m(self):
        """:math:`m = M \\beta`"""
        return dot(self._tM, self._tbeta)

    @property
    def genetic_variance(self):
        return self._v

    @property
    def total_variance(self):
        tv = self.covariates_variance + self.genetic_variance
        tv += self.environmental_variance
        return tv

    @property
    def covariates_variance(self):
        return variance(self.m())

    @property
    def environmental_variance(self):
        raise NotImplementedError

    @environmental_variance.setter
    def environmental_variance(self, v):
        raise NotImplementedError

    @property
    def heritability(self):
        return self.genetic_variance / self.total_variance

    @heritability.setter
    def heritability(self, v):
        t = self.covariates_variance + self.environmental_variance
        self.genetic_variance = t * (v / (1 - v))

    @genetic_variance.setter
    def genetic_variance(self, value):
        # self.clear_cache('_r')
        self.clear_cache('_lml_components')
        self.clear_cache('_L')
        self.clear_cache('_Q0BiQ0t')
        self.clear_cache('_update')
        self.clear_cache('K')
        self.clear_cache('diagK')
        self._v = max(value, 1e-7)

    @property
    def _tbeta(self):
        return self.__tbeta

    @_tbeta.setter
    def _tbeta(self, value):
        # self.clear_cache('_r')
        self.clear_cache('_lml_components')
        self.clear_cache('m')
        self.clear_cache('_update')
        if self.__tbeta is None:
            self.__tbeta = asarray(value, float).copy()
        else:
            self.__tbeta[:] = value

    @property
    def beta(self):
        return solve(self._svd_V.T, self._tbeta / self._svd_S12)

    @beta.setter
    def beta(self, value):
        self._tbeta = self._svd_S12 * dot(self._svd_V.T, value)

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        self._covariate_setup(value)
        # self.clear_cache('_r')
        self.clear_cache('m')
        self.clear_cache('_lml_components')
        self.clear_cache('_update')

    @cached
    def _lml_components(self):
        self._update()
        S0 = self._S0
        m = self.m()
        ttau = self._sitelik_tau
        teta = self._sitelik_eta
        ctau = self._cav_tau
        ceta = self._cav_eta
        tctau = ttau + ctau
        # cmu = self._cavs.mu
        # TODO: MUDAR ISSO AQUI
        cmu = ceta / ctau
        A = self._A()
        C = self._C()
        L = self._L()
        QBiQt = self._Q0BiQ0t()

        gS0 = self.genetic_variance * S0
        # eC = self.environmental_variance * C
        eC = 0

        w1 = -sum(log(diagonal(L))) + (- sum(log(gS0)) / 2 + log(A).sum() / 2)
        w2 = sum(teta * eC * teta)
        w2 += dot(C * teta, dot(QBiQt, C * teta))
        w2 -= sum((teta * teta) / tctau)
        w2 /= 2

        w3 = sum(ceta * (ttau * cmu - 2 * teta) / tctau) / 2

        w4 = sum(m * C * teta) - sum(m * A * dot(QBiQt, C * teta))

        Am = A * m
        w5 = -sum(m * A * m) / 2 + dot(Am, dot(QBiQt, Am)) / 2

        w6 = -sum(log(ttau)) + sum(log(tctau)) - sum(log(ctau))
        w6 /= 2

        w7 = sum(self._loghz)

        return (w1, w2, w3, w4, w5, w6, w7)

    def lml(self):
        return sum(self._lml_components())

    @cached
    def _update(self):
        if not self._ep_params_initialized:
            self._init_ep_params()

        self._logger.info('EP loop has started.')

        pttau = self._previous_sitelik_tau
        pteta = self._previous_sitelik_eta

        ttau = self._sitelik_tau
        teta = self._sitelik_eta

        jtau = self._joint_tau
        jeta = self._joint_eta

        ctau = self._cav_tau
        ceta = self._cav_eta

        i = 0
        while i < MAX_EP_ITER:
            pttau[:] = ttau
            pteta[:] = teta

            ctau[:] = jtau - ttau
            ceta[:] = jeta - teta
            self._tilted_params()

            if not all(isfinite(self._hvar)) or any(self._hvar == 0.):
                raise Exception('Error: not all(isfinite(hsig2))' +
                                ' or any(hsig2 == 0.).')

            self._sitelik_update()
            # self.clear_cache('_r')
            self.clear_cache('_L')
            self.clear_cache('_A')
            self.clear_cache('_C')
            self.clear_cache('_Q0BiQ0t')

            self._joint_update()

            tdiff = abs(pttau - ttau)
            ediff = abs(pteta - teta)
            aerr = tdiff.max() + ediff.max()

            if pttau.min() <= 0. or (0. in pteta):
                rerr = inf
            else:
                rtdiff = tdiff / abs(pttau)
                rediff = ediff / abs(pteta)
                rerr = rtdiff.max() + rediff.max()

            i += 1
            if aerr < 2 * EP_EPS or rerr < 2 * EP_EPS:
                break

        if i + 1 == MAX_EP_ITER:
            self._logger.warn('Maximum number of EP iterations has' +
                              ' been attained.')

        self._logger.info('EP loop has performed %d iterations.', i)

    def _joint_update(self):
        A = self._A()
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

    def _sitelik_update(self):
        hmu = self._hmu
        hvar = self._hvar
        tau = self._cav_tau
        eta = self._cav_eta
        self._sitelik_tau[:] = maximum(1.0 / hvar - tau, 1e-16)
        self._sitelik_eta[:] = hmu / hvar - eta

    def _optimal_beta_nom(self):
        A = self._A()
        C = self._C()
        QBiQt = self._Q0BiQ0t()
        teta = self._sitelik_eta
        return C * teta - A * dot(QBiQt, C * teta)

    def _optimal_tbeta_denom(self):
        QBiQt = self._Q0BiQ0t()
        AM = ddot(self._A(), self._tM, left=True)
        return dot(self._tM.T, AM) - multi_dot([AM.T, QBiQt, AM])

    def _optimal_tbeta(self):
        self._update()

        if all(abs(self._M) < 1e-15):
            return zeros_like(self._tbeta)

        u = dot(self._tM.T, self._optimal_beta_nom())
        Z = self._optimal_tbeta_denom()

        try:
            with errstate(all='raise'):
                self._tbeta = solve(Z, u)

        except (LinAlgError, FloatingPointError):
            self._logger.warn('Failed to compute the optimal beta.' +
                              ' Zeroing it.')
            self.__tbeta[:] = 0.

        return self.__tbeta

    def _optimize_beta(self):
        ptbeta = empty_like(self._tbeta)

        step = inf
        i = 0
        while step > 1e-7 and i < 5:
            ptbeta[:] = self._tbeta
            self._optimal_tbeta()
            step = sum((self._tbeta - ptbeta)**2)
            i += 1

    def optimize(self):

        self._logger.info("Start of optimization.")

        def function_cost(v):
            self.genetic_variance = v
            self._optimize_beta()
            return -self.lml()

        start = time()
        v, nfev = find_minimum(function_cost, self.genetic_variance, a=1e-4,
                               b=1e4, rtol=0, atol=1e-6)

        self.genetic_variance = v

        self._optimize_beta()
        elapsed = time() - start

        msg = "End of optimization (%.3f seconds, %d function calls)."
        self._logger.info(msg, elapsed, nfev)

    @cached
    def _A(self):
        return self._sitelik_tau

    @cached
    def _C(self):
        return 1

    @cached
    def _S0Q0t(self):
        """:math:`S Q^t`"""
        return ddot(self._S0, self._Q0.T, left=True)

    def _Q0S0Q0t(self):
        """:math:`\\mathrm Q \\mathrm S \\mathrm Q^t`"""
        if self.__Q0S0Q0t is None:
            Q0 = self._Q0
            self.__Q0S0Q0t = dot(Q0, self._S0Q0t())
        return self.__Q0S0Q0t

    @cached
    def _Q0BiQ0t(self):
        Q0 = self._Q0
        return Q0.dot(cho_solve(self._L(), Q0.T))

    def _B(self):
        Q0 = self._Q0
        A = self._A()
        Q0tAQ0 = dot(Q0.T, ddot(A, Q0, left=True))
        return sum2diag(Q0tAQ0, 1. / (self.genetic_variance * self._S0))

    @cached
    def _L(self):
        return cho_factor(self._B(), lower=True)[0]
    #
    # @cached
    # def _r(self):
    #     teta=self._sites.eta
    #     A1=self._A1()
    #     Q0B1Q0t=self._Q0B1Q0t()
    #     K=self.K()
    #
    #     u=self.m() + K.dot(teta)
    #     A1u=A1 * u
    #     return A1u - A1 * Q0B1Q0t.dot(A1u) - teta
