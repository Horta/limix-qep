from __future__ import absolute_import, division, unicode_literals

import logging
from time import time

from hcache import Cached, cached
from numpy import var as variance
from numpy import (abs, all, any, asarray, atleast_1d, atleast_2d, clip,
                   diagonal, dot, empty, empty_like, inf, isfinite, log,
                   maximum, set_printoptions, sqrt, sum, zeros)
from numpy.linalg import multi_dot
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

        nsamples = M.shape[0]

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

        self._previous_sitelik_tau = zeros(nsamples)
        self._previous_sitelik_eta = zeros(nsamples)

        self._sitelik_tau = zeros(nsamples)
        self._sitelik_eta = zeros(nsamples)

        self._cav_tau = zeros(nsamples)
        self._cav_eta = zeros(nsamples)

        self._joint_tau = zeros(nsamples)
        self._joint_eta = zeros(nsamples)

        # genetic variance
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

    def _posterior_normal(self, m, var, covar):
        m = atleast_1d(m)
        var = atleast_2d(var)
        covar = atleast_2d(covar)

        A1 = self._A1()
        Q0B1Q0t = self._Q0B1Q0t()

        mu = m - covar.dot(self._r())

        A1covar = ddot(A1, covar.T, left=True)
        sig2 = var.diagonal() - dotd(A1covar.T, covar.T)\
            + dotd(A1covar.T, Q0B1Q0t.dot(A1covar))

        return (mu, sig2)

    def _init_ep_params(self):
        assert self._ep_params_initialized is False
        self._logger.info("EP parameters initialization.")
        self._joint_initialize()
        self._sitelik_initialize()
        self._ep_params_initialized = True

    def initialize_hyperparams(self):
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
        if self._v is None:
            self.initialize_hyperparams()
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
        # self.clear_cache('_lml_components')
        # self.clear_cache('_L')
        # self.clear_cache('_Q0B1Q0t')
        # self.clear_cache('_update')
        # self.clear_cache('K')
        # self.clear_cache('diagK')
        self._v = max(value, 1e-7)

    @property
    def _tbeta(self):
        if self.__tbeta is None:
            self.initialize_hyperparams()
        return self.__tbeta

    @_tbeta.setter
    def _tbeta(self, value):
        # self.clear_cache('_r')
        # self.clear_cache('_lml_components')
        # self.clear_cache('m')
        # self.clear_cache('_update')
        if self.__tbeta is None:
            self.__tbeta = asarray(value, float).copy()
        else:
            self.__tbeta[:] = value

    @property
    def beta(self):
        if self.__tbeta is None:
            self.initialize_hyperparams()
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
        self.clear_cache('_r')
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
        # cmu = self._cavs.mu
        # TODO: MUDAR ISSO AQUI
        cmu = ceta / ctau
        A0 = self._A0()
        A1 = self._A1()
        A2 = self._A2()

        vS0 = self.genetic_variance * S0
        tctau = ttau + ctau
        A1m = A1 * m
        A2m = A2 * m
        Q0B1Q0t = self._Q0B1Q0t()
        A2teta = A2 * teta

        L = self._L()

        p1 = - sum(log(diagonal(L))) - log(vS0).sum() / 2

        p3 = (teta * A0 * teta / (ttau * A0 + 1)).sum()
        p3 += (A2teta * Q0B1Q0t.dot(A2teta)).sum()
        p3 -= ((teta * teta) / tctau).sum()
        p3 /= 2

        p4 = (ceta * (ttau * cmu - 2 * teta) / tctau).sum() / 2

        A1mQ0B1Q0t = A1m.dot(Q0B1Q0t)

        p5 = A2m.dot(teta) - A1mQ0B1Q0t.dot(A2 * teta)

        p6 = - A1m.dot(m) + A1mQ0B1Q0t.dot(A1m)
        p6 /= 2

        p7 = (log(tctau).sum() - log(ctau).sum()) / 2

        p8 = self._loghz.sum()

        p9 = log(A2).sum() / 2

        return (p1, p3, p4, p5, p6, p7, p8, p9)

    def lml(self):
        return sum(self._lml_components())

    @cached
    def _update(self):
        if not self._ep_params_initialized:
            self._init_ep_params()

        self._logger.info('EP loop has started.')
        m = self.m()

        ttau = self._sitelik_tau
        teta = self._sitelik_eta

        jtau = self._joint_tau
        jeta = self._joint_eta

        hmu = self._hmu
        hvar = self._hvar

        i = 0
        while i < MAX_EP_ITER:
            self._previous_sitelik_tau[:] = ttau
            self._previous_sitelik_eta[:] = teta

            # self._cav_update()
            # self._cavs.update(self._joint.tau, self._joint.eta, ttau, teta)

            self._cav_tau[:] = jtau - ttau
            self._cav_eta[:] = jeta - teta
            self._tilted_params()

            if not all(isfinite(hvar)) or any(hvar == 0.):
                raise Exception('Error: not all(isfinite(hsig2))' +
                                ' or any(hsig2 == 0.).')

            # self._sites.update(self._cavs.tau, self._cavs.eta, hmu, hvar)
            self._sitelik_update(hmu, hvar)
            self.clear_cache('_r')
            self.clear_cache('_L')
            self.clear_cache('_A1')
            self.clear_cache('_A2')
            self.clear_cache('_Q0B1Q0t')

            # self._joint.update(m, teta, self._A1(), self._A2(),
            #                    self._Q0B1Q0t(), self.K())
            import pytest
            pytest.set_trace()
            self._joint_update()

            tdiff = abs(self._psites.tau - ttau)
            ediff = abs(self._psites.eta - teta)
            aerr = tdiff.max() + ediff.max()

            if self._psites.tau.min() <= 0. or (0. in self._psites.eta):
                rerr = inf
            else:
                rtdiff = tdiff / abs(self._psites.tau)
                rediff = ediff / abs(self._psites.eta)
                rerr = rtdiff.max() + rediff.max()

            i += 1
            # self._logger.info('EP step size: %e.', max(aerr, rerr))
            if aerr < 2 * EP_EPS or rerr < 2 * EP_EPS:
                break

        if i + 1 == MAX_EP_ITER:
            self._logger.warn('Maximum number of EP iterations has' +
                              ' been attained.')

        self._logger.info('EP loop has performed %d iterations.', i)

    def _joint_update(self):
        K = self.K()
        m = self.m()
        A2 = self._A2()
        QB1Qt = self._Q0B1Q0t()

        jtau = self._joint_tau
        jeta = self._joint_eta

        diagK = K.diagonal()
        QB1QtA1 = ddot(QB1Qt, self._A1(), left=False)
        jtau[:] = 1 / (A2 * diagK - A2 * dotd(QB1QtA1, K))

        Kteta = K.dot(self._sitelik_eta)
        jeta[:] = A2 * (m - QB1QtA1.dot(m) + Kteta - QB1QtA1.dot(Kteta))
        jeta *= jtau

    def _sitelik_update(self, hmu, hvar):
        tau = self._cav_tau
        eta = self._cav_eta
        tau[:] = maximum(1.0 / hvar - tau, 1e-16)
        eta[:] = hmu / hvar - eta

    # def _optimal_beta_nom(self):
    #     A1 = self._A1()
    #     A2 = self._A2()
    #     Q0B1Q0t = self._Q0B1Q0t()
    #     teta = self._sites.eta
    #     return A2 * teta - A1 * Q0B1Q0t.dot(A2 * teta)
    #
    # def _optimal_tbeta_denom(self):
    #     Q0B1Q0t = self._Q0B1Q0t()
    #     A1M = ddot(self._A1(), self._tM, left=True)
    #     return dot(self._tM.T, A1M) - multi_dot([A1M.T, Q0B1Q0t, A1M])
    #
    # def _optimal_tbeta(self):
    #     self._update()
    #
    #     if all(np.abs(self._M) < 1e-15):
    #         return np.zeros_like(self._tbeta)
    #
    #     u = dot(self._tM.T, self._optimal_beta_nom())
    #     Z = self._optimal_tbeta_denom()
    #
    #     try:
    #         with np.errstaall_ = 'raise'):
    #             self._tbeta=solve(Z, u)
    #
    #     except (np.linalg.LinAlgError, FloatingPointError):
    #         self._logger.warn('Failed to compute the optimal beta.' +
    #                           ' Zeroing it.')
    #         self.__tbeta[:]=0.
    #
    #     return self.__tbeta
    #
    # def _optimize_beta(self):
    #     # self._logger.info("Beta optimization.")
    #     ptbeta=empty_like(self._tbeta)
    #
    #     step=inf
    #     i=0
    #     while step > 1e-7 and i < 5:
    #         ptbeta[:]=self._tbeta
    #         self._optimal_tbeta()
    #         step=np.sum((self._tbeta - ptbeta)**2)
    #         # self._logger.info("Beta step: %e.", step)
    #         i += 1
    #
    #     # self._logger.info("Beta optimization performed %d steps " +
    #     #                    "to find %s.", i, bytes(self._tbeta))
    #
    # def _h2_cost(self, h2):
    #     h2=clip(h2, 1e-3, 1 - 1e-3)
    #     self._logger.info("Evaluating for h2: %e.", h2)
    #     var=self.h2tovar(h2)
    #     self.genetic_variance=var
    #     self._optimize_beta()
    #     return -self.lml()
    #
    # def optimize(self):
    #
    #     start=time()
    #
    #     self._logger.info("Start of optimization.")
    #     # self._logger.info(self.__str__())
    #     # self._logger.info("Initial parameters: h2=%.5f, var=%.5f, beta=%s.",
    #     #                    self.heritability, self.genetic_variance,
    #     #                    bytes(self.beta))
    #
    #     nfev=0
    #
    #     h2=self.heritability
    #     atol=1e-6
    #     eps=1e-4
    #     h2, nfev=find_minimum(self._h2_cost, h2, a = eps, b = 1 - eps, rtol = 0,
    #                             atol = atol)
    #     # r = minimize_scalar(self._h2_cost,
    #     #                     bracket=normal_bracket(h2),
    #     #                     method='Brent', tol=HYPERPARAM_EPS)
    #
    #     self.genetic_variance=self.h2tovar(h2)
    #
    #     # if not r.success:
    #     #     self._logger.warn("Optimizer has failed: %s.", str(r))
    #
    #     # nfev = r.nfev
    #
    #     self._optimize_beta()
    #
    #     # self._logger.info("Final parameters: h2=%.5f, var=%.5f, beta=%s",
    #     #                    self.heritability, self.genetic_variance,
    #     #                    bytes(self.beta))
    #
    #     # self._logger.info(self.__str__())
    #     msg="End of optimization (%.3f seconds, %d function calls)."
    #     self._logger.info(msg, time() - start, nfev)
    #
    # @cached
    # def _A0(self):
    #     """:math:`v \\delta \\mathrm I`"""
    #     return 0.0
    #
    # @cached
    # def _A1(self):
    #     """:math:`(v \\delta \\mathrm I + \\tilde{\\mathrm T}^{-1})^{-1}`"""
    #     return self._sites.tau
    #
    # @cached
    # def _A2(self):
    #     """:math:`\\tilde{\\mathrm T}^{-1} \\mathrm A_1`"""
    #     return 1.0
    #
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
    #
    # @cached
    # def _Q0B1Q0t(self):
    #     Q0=self._Q0
    #     return Q0.dot(cho_solve(self._L(), Q0.T))
    #
    # def _B(self):
    #     """:math:`\\mathrm B = \\mathrm Q^t \\mathrm A_1 \\mathrm Q +
    #               \\mathrm S^{-1} v^{-1}`"""
    #     Q0=self._Q0
    #     A1=self._A1()
    #     Q0tA1Q0=dot(Q0.T, ddot(A1, Q0, left=True))
    #     return sum2diag(Q0tA1Q0, 1. / (self.genetic_variance * self._S0))
    #
    # @cached
    # def _L(self):
    #     """:math:`\\mathrm L \\mathrm L^T =
    #         \\mathrm{Chol}\\{ \\mathrm B \\}`"""
    #     return cho_factor(self._B(), lower = True)[0]
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
