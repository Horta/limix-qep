from __future__ import absolute_import, division, unicode_literals

import logging
from math import fsum
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


class EP(Cached):
    r"""
    Let :math:`\mathrm Q \mathrm S \mathrm Q^{\intercal}` be the economic
    eigendecomposition of the genetic covariance matrix.
    Let :math:`\mathrm U\mathrm S\mathrm V^{\intercal}` be the singular value
    decomposition of the user-provided covariates :math:`\mathrm M`. We define

    .. math::

        \mathrm K = v ((1-\delta)\mathrm Q \mathrm S \mathrm Q^{\intercal} +
                    \delta \mathrm I)

    as the covariance of the prior distribution. As such,
    :math:`v` and :math:`\delta` refer to :py:attr:`_v` and :py:attr:`_delta`
    class attributes, respectively. We also use the following variables for
    convenience:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \sigma_b^2          & = & v (1-\delta) \\
            \sigma_{\epsilon}^2 & = & v \delta
        \end{eqnarray}

    The covariate effect-sizes is given by :math:`\boldsymbol\beta`, which
    implies

    .. math::

        \mathbf m = \mathrm M \boldsymbol\beta

    The prior is thus defined as

    .. math::

        \mathcal N(\mathbf z ~|~ \mathbf m; \mathrm K)

    and the marginal likelihood is given by

    .. math::

        p(\mathbf y) = \int \prod_i p(y_i | g(\mathrm E[y_i | z_i])=z_i)
            \mathcal N(\mathbf z ~|~ \mathbf m, \mathrm K) \mathrm d\mathbf z

    However, the singular value decomposition of the covariates allows us to
    automatically remove dependence between covariates, which would create
    infinitly number of :math:`\boldsymbol\beta` that lead to global optima.
    Let us define

    .. math::

        \tilde{\boldsymbol\beta} = \mathrm S^{1/2} \mathrm V^{\intercal}
                                    \boldsymbol\beta

    as the covariate effect-sizes we will effectively work with during the
    optimization process. Let us also define the

    .. math::

        \tilde{\mathrm M} = \mathrm U \mathrm S^{1/2}

    as the redundance-free covariates. Naturally,

    .. math::

        \mathbf m = \tilde{\mathrm M} \tilde{\boldsymbol\beta}

    In summary, we will optimize :math:`\tilde{\boldsymbol{\beta}}`, even
    though the user will be able to retrieve the corresponding
    :math:`\boldsymbol{\beta}`.


    Let

    .. math::

        \mathrm{KL}[p(y_i|z_i) p_{-}(z_i|y_i)_{\text{EP}} ~|~
            p(y_i|z_i)_{\text{EP}} p_{-}(z_i|y_i)_{\text{EP}}]

    be the KL divergence we want to minimize at each EP iteration.
    The left-hand side can be described as
    :math:`\hat c_i \mathcal N(z_i | \hat \mu_i; \hat \sigma_i^2)`


    Args:
        M (array_like): :math:`\mathrm M` covariates.
        Q (array_like): :math:`\mathrm Q` of the economic
                        eigendecomposition.
        S (array_like): :math:`\mathrm S` of the economic
                        eigendecomposition.
        QSQt (array_like): :math:`\mathrm Q \mathrm S
                        \mathrm Q^{\intercal}` in case this has already
                        been computed.


    Attributes:
        _v (float): Total variance :math:`v` from the prior distribution.
        _delta (float): Fraction of the total variance due to the identity
                        matrix :math:`\mathrm I`.
        _loghz (array_like): This is :math:`\log(\hat c)` for each site.
        _hmu (array_like): This is :math:`\hat \mu` for each site.
        _hvar (array_like): This is :math:`\hat \sigma^2` for each site.

    """

    def __init__(self, M, Q, S, fixed_delta, QSQt=None):
        Cached.__init__(self)
        self._logger = logging.getLogger(__name__)

        if not all(isfinite(Q)) or not all(isfinite(S)):
            raise ValueError("There are non-finite numbers in the provided" +
                             " eigen decomposition.")

        if S.min() <= 0:
            raise ValueError("The provided covariance matrix is not" +
                             " positive-definite because the minimum" +
                             " eigvalue is %f." % S.min())

        make_sure_reasonable_conditioning(S)

        self._covariate_setup(M)
        self._S = S
        self._Q = Q
        self.__QSQt = QSQt

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
        self._delta = 0
        self._fixed_delta = fixed_delta
        self.__tbeta = None

        self._loghz = empty(nsamples)
        self._hmu = empty(nsamples)
        self._hvar = empty(nsamples)

    def _covariate_setup(self, M):
        self._M = M
        SVD = economic_svd(M)
        self._svd_U = SVD[0]
        self._svd_S12 = sqrt(SVD[1])
        self._svd_V = SVD[2]
        self._tM = ddot(self._svd_U, self._svd_S12, left=False)
        self.__tbeta = None

    @cached
    def _init_ep_params(self):
        self._logger.info("EP parameters initialization.")
        self._joint_initialize()
        self._sitelik_initialize()

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
        r"""Returns :math:`\mathrm K`."""
        return sum2diag(self.sigma2_b * self._QSQt(), self.sigma2_epsilon)

    @cached
    def diagK(self):
        r"""Returns the diagonal of :math:`\mathrm K`."""
        return self.sigma2_b * self._diagQSQt() + self.sigma2_epsilon

    def _diagQSQt(self):
        return self._QSQt().diagonal()

    @cached
    def m(self):
        r"""Returns :math:`\mathbf m = \mathrm M \boldsymbol\beta`."""
        return dot(self._tM, self._tbeta)

    @property
    def covariates_variance(self):
        r"""Variance explained by the covariates.

        It is defined as

        .. math::

            \sum_{s=1}^p \left\{ \sum_{i=1}^n \left( \mathrm M_{i,s}\beta_s -
                \sum_{j=1}^n\frac{\mathrm M_{j,s}\beta_s}{n} \right)^2 \Big/ n
            \right\}

        where :math:`p` is the number of covariates and :math:`n` is the number
        of individuals. One can show that it amounts to
        :math:`\sum_s \beta_s^2` whenever the columns of :math:`\mathrm M`
        are normalized to have mean and standard deviation equal to zero and
        one, respectively.
        """
        return fsum(variance(self.M * self.beta, axis=0))

    @property
    def sigma2_b(self):
        r"""Returns :math:`v (1-\delta)`."""
        return self.v

    @property
    def sigma2_epsilon(self):
        r"""Returns :math:`v \delta`."""
        return self.v * self.delta

    @property
    def delta(self):
        r"""Returns :math:`\delta`."""
        return self._delta

    @delta.setter
    def delta(self, v):
        r"""Set :math:`\delta`."""
        self.clear_cache('K')
        self.clear_cache('diagK')
        self.clear_cache('_update')
        self.clear_cache('_lml_components')
        self.clear_cache('_L')
        self.clear_cache('_A')
        self.clear_cache('_C')
        self.clear_cache('_QBiQt')
        self._delta = max(v, 1e-7)

    @property
    def v(self):
        r"""Returns :math:`v`."""
        return self._v

    @v.setter
    def v(self, v):
        r"""Set :math:`v`."""
        self.clear_cache('K')
        self.clear_cache('diagK')
        self.clear_cache('_update')
        self.clear_cache('_lml_components')
        self.clear_cache('_L')
        self.clear_cache('_A')
        self.clear_cache('_C')
        self.clear_cache('_QBiQt')
        self._v = max(v, 1e-7)

    @property
    def _tbeta(self):
        return self.__tbeta

    @_tbeta.setter
    def _tbeta(self, value):
        self.clear_cache('_lml_components')
        self.clear_cache('m')
        self.clear_cache('_update')
        if self.__tbeta is None:
            self.__tbeta = asarray(value, float).copy()
        else:
            self.__tbeta[:] = value

    @property
    def beta(self):
        r"""Returns :math:`\boldsymbol\beta`."""
        return solve(self._svd_V.T, self._tbeta / self._svd_S12)

    @beta.setter
    def beta(self, value):
        self._tbeta = self._svd_S12 * dot(self._svd_V.T, value)

    @property
    def M(self):
        r"""Returns :math:`\mathrm M`."""
        return self._M

    @M.setter
    def M(self, value):
        self._covariate_setup(value)
        self.clear_cache('m')
        self.clear_cache('_lml_components')
        self.clear_cache('_update')

    @cached
    def _lml_components(self):
        self._update()
        S = self._S
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
        QBiQt = self._QBiQt()

        gS = self.sigma2_b * S
        eC = self.sigma2_epsilon * C

        w1 = -sum(log(diagonal(L))) + (- sum(log(gS)) / 2 + log(A).sum() / 2)

        # w2 = sum(teta * eC * teta)
        # w2 += dot(C * teta, dot(QBiQt, C * teta))
        # w2 -= sum((teta * teta) / tctau)
        # w2 /= 2

        w2 = eC * teta
        w2 += ddot(C, dot(QBiQt, C * teta), left=True)
        w2 -= teta / tctau
        w2 = dot(teta, w2) / 2

        # w3 = sum(ceta * (ttau * cmu - 2 * teta) / tctau) / 2
        w3 = dot(ceta, (ttau * cmu - 2 * teta) / tctau) / 2

        Am = A * m
        w4 = dot(m * C, teta) - dot(Am, dot(QBiQt, C * teta))

        # w5 = -sum(m * A * m) / 2 + dot(Am, dot(QBiQt, Am)) / 2
        w5 = -dot(Am, m) / 2 + dot(Am, dot(QBiQt, Am)) / 2

        w6 = -sum(log(ttau)) + sum(log(tctau)) - sum(log(ctau))
        w6 /= 2

        w7 = sum(self._loghz)

        return (w1, w2, w3, w4, w5, w6, w7)

    def lml(self):
        return fsum(self._lml_components())

    @cached
    def _update(self):
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
            self.clear_cache('_lml_components')
            self.clear_cache('_L')
            self.clear_cache('_A')
            self.clear_cache('_C')
            self.clear_cache('_QBiQt')

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
        C = self._C()
        K = self.K()
        m = self.m()
        teta = self._sitelik_eta
        QBiQt = self._QBiQt()

        jtau = self._joint_tau
        jeta = self._joint_eta

        diagK = K.diagonal()
        QBiQtA = ddot(QBiQt, A, left=False)
        jtau[:] = 1 / (diagK - dotd(QBiQtA, K))

        Kteta = K.dot(teta)
        jeta[:] = m - QBiQtA.dot(m) + Kteta - QBiQtA.dot(Kteta)
        jeta *= jtau
        jtau /= C

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
        QBiQt = self._QBiQt()
        teta = self._sitelik_eta
        return C * teta - A * dot(QBiQt, C * teta)

    def _optimal_tbeta_denom(self):
        QBiQt = self._QBiQt()
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
            self.v = v
            if not self._fixed_delta:
                def function_cost_delta(delta):
                    self.delta = delta
                    self._optimize_beta()
                    return -self.lml()
                d, nfev = find_minimum(function_cost_delta, self.delta,
                                       a=1e-4, b=1 - 1e-4, rtol=0, atol=1e-6)
                self.delta = d
            self._optimize_beta()
            return -self.lml()

        start = time()
        v, nfev = find_minimum(function_cost, self.v, a=1e-4,
                               b=1e4, rtol=0, atol=1e-6)

        self.v = v

        self._optimize_beta()
        elapsed = time() - start

        msg = "End of optimization (%.3f seconds, %d function calls)."
        self._logger.info(msg, elapsed, nfev)

    @cached
    def _A(self):
        r"""Returns :math:`\mathcal A = \tilde{\mathrm T} \mathcal C^{-1}`."""
        ttau = self._sitelik_tau
        s2 = self.sigma2_epsilon
        return ttau / (ttau * s2 + 1)

    @cached
    def _C(self):
        r"""Returns :math:`\mathcal C = \sigma_{\epsilon}^2 \tilde{\mathrm T} +
            \mathrm I`."""
        ttau = self._sitelik_tau
        s2 = self.sigma2_epsilon
        return 1 / (ttau * s2 + 1)

    @cached
    def _SQt(self):
        r"""Returns :math:`\mathrm S \mathrm Q^\intercal`."""
        return ddot(self._S, self._Q.T, left=True)

    def _QSQt(self):
        r"""Returns :math:`\mathrm Q \mathrm S \mathrm Q^\intercal`."""
        if self.__QSQt is None:
            Q = self._Q
            self.__QSQt = dot(Q, self._SQt())
        return self.__QSQt

    @cached
    def _QBiQt(self):
        Q = self._Q
        return Q.dot(cho_solve(self._L(), Q.T))

    @cached
    def _L(self):
        r"""Returns the Cholesky factorization of :math:`\mathcal B`.

        .. math::

            \mathcal B = \mathrm Q^{\intercal}\mathcal A\mathrm Q
                (\sigma_b^2 \mathrm S)^{-1}
        """
        Q = self._Q
        A = self._A()
        QtAQ = dot(Q.T, ddot(A, Q, left=True))
        B = sum2diag(QtAQ, 1. / (self.sigma2_b * self._S))
        return cho_factor(B, lower=True)[0]
