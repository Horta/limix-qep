from __future__ import absolute_import
from __future__ import division

import logging
import numpy as np

from numpy import inf
from numpy import dot
from numpy import log
from numpy import sqrt
from numpy import empty
from numpy import asarray
from numpy import empty_like
from numpy import atleast_1d
from numpy import atleast_2d
from numpy import var as variance
from numpy.linalg import multi_dot

from scipy.linalg import cho_factor
from scipy.optimize import minimize_scalar

from limix_math.linalg import ddot
from limix_math.linalg import sum2diag
from limix_math.linalg import dotd
from limix_math.linalg import solve
from limix_math.linalg import cho_solve

from hcache import Cached, cached

from limix_math.linalg import economic_svd

from .dists import SiteLik
from .dists import Joint
from .dists import Cavity

# from .fixed_ep import FixedEP

from .config import MAX_EP_ITER
from .config import EP_EPS
from .config import HYPERPARAM_EPS

from .util import make_sure_reasonable_conditioning
from .util import golden_bracket


class EP(Cached):
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
        Cached.__init__(self)
        self._logger = logging.getLogger(__name__)
        self._ep_params_initialized = False

        nsamples = M.shape[0]

        if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(S)):
            raise ValueError("There are non-finite numbers in the provided" +
                             " eigen decomposition.")

        if S.min() <= 0:
            raise ValueError("The provided covariance matrix is not" +
                            " positive-definite because the minimum eigvalue" +
                            " is %f." % S.min())

        make_sure_reasonable_conditioning(S)

        self._covariate_setup(M)
        self._S = S
        self._Q = Q
        self.__QSQt = QSQt

        self._psites = SiteLik(nsamples)
        self._sites = SiteLik(nsamples)
        self._cavs = Cavity(nsamples)
        self._joint = Joint(Q, S)

        self._var = None
        self.__tbeta = None

        self._loghz = empty(nsamples)
        self._hmu = empty(nsamples)
        self._hvar = empty(nsamples)

        # useful for optimization only
        # that is a hacky
        self._flip_heritability = False

    def _covariate_setup(self, M):
        self._M = M
        SVD = economic_svd(M)
        self._svd_U = SVD[0]
        self._svd_S12 = sqrt(SVD[1])
        self._svd_V = SVD[2]
        self._tM = ddot(self._svd_U, self._svd_S12, left=False)
        self.__tbeta = None

    def fixed_ep(self):
        self._update()
        # (p1, p3, p4, _, _, p7, p8, f0, A0A0pT_teta) =\
        #     self._lml_components()
        #
        # lml_nonbeta_part = p1 + p3 + p4 + p7 + p8
        # Q = self._Q
        # L = self._L()
        # A = self._A1()
        # opt_bnom = self._opt_beta_nom()
        # vv1 = FixedEP(lml_nonbeta_part, A0A0pT_teta, f0,\
        #                 A, L, Q, opt_bnom)
        #
        # return vv1

    def _posterior_normal(self, m, var, covar):
        m = atleast_1d(m)
        var = atleast_2d(var)
        covar = atleast_2d(covar)

        A1 = self._A1()
        QB1Qt = self._QB1Qt()

        mu = m - covar.dot(self._r())

        A1covar = ddot(A1, covar.T, left=True)
        sig2 = var.diagonal() - dotd(A1covar.T, covar.T)\
               + dotd(A1covar.T, QB1Qt.dot(A1covar))

        return (mu, sig2)

    def _init_ep_params(self):
        assert self._ep_params_initialized is False
        self._logger.debug("EP parameters initialization.")
        self._joint.initialize(self.m(), self.diagK())
        self._sites.initialize()
        self._ep_params_initialized = True

    def initialize_hyperparams(self):
        raise NotImplementedError

    @cached
    def K(self):
        """:math:`K = v Q S Q.T`"""
        return self.var * self._QSQt()

    @cached
    def diagK(self):
        return self.var * self._diagQSQt()

    def _diagQSQt(self):
        return self._QSQt().diagonal()

    @cached
    def m(self):
        """:math:`m = M \\beta`"""
        return dot(self._tM, self._tbeta)


    ############################################################################
    ############################################################################
    ######################## Getters and setters ###############################
    ############################################################################
    ############################################################################
    @property
    def total_variance(self):
        return (self.covariates_variance + self.genetic_variance +
                self.environmental_variance + self.instrumental_variance)

    @property
    def covariates_variance(self):
        return variance(self.m())

    @property
    def genetic_variance(self):
        return self.var

    @genetic_variance.setter
    def genetic_variance(self, v):
        self.var = v

    @property
    def environmental_variance(self):
        return 1.

    @environmental_variance.setter
    def environmental_variance(self, v):
        raise NotImplementedError

    @property
    def instrumental_variance(self):
        return 0.

    @property
    def heritability(self):
        return self.genetic_variance / (self.genetic_variance +
                                        self.environmental_variance +
                                        self.covariates_variance)

    def h2tovar(self, h2):
        varc = variance(self.m())
        return h2 * (1 + varc) / (1 - h2)

    @property
    def _tbeta(self):
        if self.__tbeta is None:
            self.initialize_hyperparams()
        return self.__tbeta

    @_tbeta.setter
    def _tbeta(self, value):
        self.clear_cache('_r')
        self.clear_cache('_lml_components')
        self.clear_cache('m')
        self.clear_cache('_update')
        if self.__tbeta is None:
            self.__tbeta = asarray(value, float).copy()
        else:
            self.__tbeta[:] = value

    @property
    def beta(self):
        if self.__tbeta is None:
            self.initialize_hyperparams()
        return solve(self._svd_V.T, self._tbeta/self._svd_S12)

    @beta.setter
    def beta(self, value):
        self._tbeta = self._svd_S12 * dot(self._svd_V.T, value)

    @property
    def var(self):
        if self._var is None:
            self.initialize_hyperparams()
        return self._var

    @var.setter
    def var(self, value):
        self.clear_cache('_r')
        self.clear_cache('_lml_components')
        self.clear_cache('_L')
        self.clear_cache('_QB1Qt')
        self.clear_cache('_update')
        self.clear_cache('K')
        self.clear_cache('diagK')
        self._var = max(value, 1e-7)

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

    ############################################################################
    ############################################################################
    ###################### Log Marginal Likelihood #############################
    ############################################################################
    ############################################################################
    @cached
    def _lml_components(self):
        self._update()
        # Q = self._Q
        S = self._S
        m = self.m()
        var = self.var
        ttau = self._sites.tau
        teta = self._sites.eta
        ctau = self._cavs.tau
        ceta = self._cavs.eta
        cmu = self._cavs.mu
        A0 = self._A0()
        A1 = self._A1()
        A2 = self._A2()

        varS = var * S
        tctau = ttau + ctau
        A1m = A1*m
        A2m = A2*m
        QB1Qt = self._QB1Qt()
        A2teta = A2 * teta

        L = self._L()

        p1 = - np.sum(log(np.diagonal(L))) - log(varS).sum() / 2

        p3 = (teta * A0 * teta / (ttau * A0 + 1)).sum()
        p3 += (A2teta * QB1Qt.dot(A2teta)).sum()
        p3 -= ((teta * teta) / tctau).sum()
        p3 /= 2

        p4 = (ceta * (ttau * cmu - 2*teta) / tctau).sum() / 2

        A1mQB1Qt = A1m.dot(QB1Qt)

        p5 = A2m.dot(teta) - A1mQB1Qt.dot(A2*teta)

        p6 = - A1m.dot(m) + A1mQB1Qt.dot(A1m)
        p6 /= 2

        p7 = (log(tctau).sum() - log(ctau).sum()) / 2

        p8 = self._loghz.sum()

        p9 = log(A2).sum() / 2
        #
        # p7 -= 0.5 * np.log(ttau).sum()


        # p3 = np.sum(teta*teta*self._AAA())
        # A0A0pT_teta = teta / self._iAAT()
        # QtA0A0pT_teta = dot(Q.T, A0A0pT_teta)
        # L = self._L()
        # L_0 = stl(L, QtA0A0pT_teta)
        # p3 += dot(L_0.T, L_0)
        # vv = 2*np.log(np.abs(teta))
        # p3 -= np.exp(logsumexp(vv - np.log(tctau)))
        # p3 *= 0.5
        #
        # p4 = 0.5 * np.sum(ceta * (ttau * cmu - 2*teta) / tctau)
        #
        # f0 = None
        # L_1 = cho_solve(L, QtA0A0pT_teta)
        # f0 = A * dot(Q, L_1)
        # p5 = dot(m, A0A0pT_teta) - dot(m, f0)
        #
        # p6 = - 0.5 * np.sum(m * Am) +\
        #     0.5 * dot(Am, dot(Q, cho_solve(L, dot(Q.T, Am))))
        #
        # p7 = 0.5 * (np.log(ttau + ctau).sum() - np.log(ctau).sum())
        #
        # p7 -= 0.5 * np.log(ttau).sum()
        #
        # p8 = self._loghz.sum()

        return (p1, p3, p4, p5, p6, p7, p8, p9)
        # return (p1, p3, p4, p5, p6, p7, p8, p9, f0, A0A0pT_teta)

    def lml(self):
        # (p1, p3, p4, p5, p6, p7, p8, _, _) = self._lml_components()
        (p1, p3, p4, p5, p6, p7, p8, p9) = self._lml_components()
        # print(p1, p3, p4, p5, p6, p7, p8, p9)
        return p1 + p3 + p4 + p5 + p6 + p7 + p8 + p9

    ############################################################################
    ############################################################################
    ########################### Main EP loop ###################################
    ############################################################################
    ############################################################################
    @cached
    def _update(self):
        if not self._ep_params_initialized:
            self._init_ep_params()

        self._logger.debug('EP loop has started.')
        m = self.m()

        ttau = self._sites.tau
        teta = self._sites.eta

        hmu = self._hmu
        hvar = self._hvar

        i = 0
        while i < MAX_EP_ITER:
            self._psites.tau[:] = ttau
            self._psites.eta[:] = teta

            self._cavs.update(self._joint.tau, self._joint.eta, ttau, teta)
            self._tilted_params()

            if not np.all(np.isfinite(hvar)) or np.any(hvar == 0.):
                raise Exception('Error: not np.all(np.isfinite(hsig2))' +
                                ' or np.any(hsig2 == 0.).')

            self._sites.update(self._cavs.tau, self._cavs.eta, hmu, hvar)
            self.clear_cache('_r')
            self.clear_cache('_L')
            self.clear_cache('_A1')
            self.clear_cache('_A2')
            self.clear_cache('_QB1Qt')


            self._joint.update(m, teta, self._A1(), self._A2(), self._QB1Qt(),
                               self.K())

            tdiff = np.abs(self._psites.tau - ttau)
            ediff = np.abs(self._psites.eta - teta)
            aerr = tdiff.max() + ediff.max()

            if self._psites.tau.min() <= 0. or (0. in self._psites.eta):
                rerr = np.inf
            else:
                rtdiff = tdiff/np.abs(self._psites.tau)
                rediff = ediff/np.abs(self._psites.eta)
                rerr = rtdiff.max() + rediff.max()

            i += 1
            self._logger.debug('EP step size: %e.', max(aerr, rerr))
            if aerr < 2*EP_EPS or rerr < 2*EP_EPS:
                break

        if i + 1 == MAX_EP_ITER:
            self._logger.warn('Maximum number of EP iterations has'+
                              ' been attained.')

        # self._logger.debug('EP loop has performed %d iterations.', i)
        # print('EP loop has performed %d iterations.' % (i,))


    ############################################################################
    ############################################################################
    ################## Optimization of hyperparameters #########################
    ############################################################################
    ############################################################################
    def _optimal_tbeta_nom(self):
        A1 = self._A1()
        A2 = self._A2()
        QB1Qt = self._QB1Qt()
        teta = self._sites.eta
        return A2 * teta - A1 * QB1Qt.dot(A2 * teta)

    def _optimal_tbeta_denom(self):
        QB1Qt = self._QB1Qt()
        A1M = ddot(self._A1(), self._tM, left=True)
        return dot(self._tM.T, A1M) - multi_dot([A1M.T, QB1Qt, A1M])

    def _optimal_tbeta(self):
        self._update()

        if np.all(np.abs(self._M) < 1e-15):
            return np.zeros_like(self._tbeta)

        u = dot(self._tM.T, self._optimal_tbeta_nom())
        Z = self._optimal_tbeta_denom()

        try:
            with np.errstate(all='raise'):
                self._tbeta = solve(Z, u)

        except (np.linalg.LinAlgError, FloatingPointError):
            self._logger.warn('Failed to compute the optimal beta.'+
                              ' Zeroing it.')
            self.__tbeta[:] = 0.

        return self.__tbeta

    def _optimize_beta(self):
        self._logger.debug("Beta optimization.")
        # print("Beta optimization.")
        ptbeta = empty_like(self._tbeta)

        step = inf
        i = 0
        while step > HYPERPARAM_EPS and i < 5:
            ptbeta[:] = self._tbeta
            self._optimal_tbeta()
            step = np.sum((self._tbeta - ptbeta)**2)
            self._logger.debug("Beta step: %e.", step)
            # print("Beta step: %.5f." % step)
            i += 1

        # print("Beta optimization took %d steps." % i)
        self._logger.debug("Beta optimization performed %d steps " +
                           "to find %s.", i, bytes(self._tbeta))

    def _h2_cost(self, h2, flip):
        if flip:
            h2 = 1 - h2
        self._logger.debug("Evaluating for h2: %e.", h2)
        var = self.h2tovar(h2)
        self.var = var
        self._optimize_beta()
        return -self.lml()

    def optimize(self):

        from time import time

        start = time()

        self._logger.debug("Start of optimization.")
        self._logger.debug("Initial parameters: h2=%.5f, var=%.5f, beta=%s.",
                           self.heritability, self.var, bytes(self.beta))

        nfev = 0
        opt = dict(xatol=HYPERPARAM_EPS)

        h2 = self.heritability
        flip = h2 > 0.5
        r = minimize_scalar(self._h2_cost, options=opt,
                            bounds=golden_bracket(0.5 - abs(0.5 - h2)),
                            method='Bounded',
                            args=flip)

        if flip:
            r.x = 1 - r.x

        self.var = self.h2tovar(r.x)

        self._logger.debug("Optimizer message: %s.", r.message)
        if r.status != 0:
            self._logger.warn("Optimizer failed with status %d.", r.status)

        nfev = r.nfev

        self._optimize_beta()

        self._logger.debug("Final parameters: h2=%.5f, var=%.5f, beta=%s",
                           self.heritability, self.var, bytes(self.beta))

        self._logger.debug("End of optimization (%.3f seconds" +
                           ", %d function calls).", time() - start, nfev)

    ############################################################################
    ############################################################################
    ############## Key but Intermediary Matrix Definitions #####################
    ############################################################################
    ############################################################################
    @cached
    def _A0(self):
        """:math:`v \\delta \\mathrm I`"""
        return 0.0

    @cached
    def _A1(self):
        """:math:`(v \\delta \\mathrm I + \\tilde{\\mathrm T}^{-1})^{-1}`"""
        return self._sites.tau

    @cached
    def _A2(self):
        """:math:`\\tilde{\\mathrm T}^{-1} \\mathrm A_1`"""
        return 1.0

    @cached
    def _SQt(self):
        """:math:`S Q^t`"""
        return ddot(self._S, self._Q.T, left=True)

    def _QSQt(self):
        """:math:`\\mathrm Q \\mathrm S \\mathrm Q^t`"""
        if self.__QSQt is None:
            Q = self._Q
            self.__QSQt = dot(Q, self._SQt())
        return self.__QSQt

    @cached
    def _QB1Qt(self):
        Q = self._Q
        return Q.dot(cho_solve(self._L(), Q.T))

    def _B(self):
        """:math:`\\mathrm B = \\mathrm Q^t \\mathrm A_1 \\mathrm Q +
                  \\mathrm S^{-1} v^{-1}`"""
        Q = self._Q
        A1 = self._A1()
        QtA1Q = dot(Q.T, ddot(A1, Q, left=True))
        return sum2diag(QtA1Q, 1./(self.var * self._S))

    @cached
    def _L(self):
        """:math:`\\mathrm L \\mathrm L^T = \\mathrm{Chol}\\{ \\mathrm B \\}`"""
        return cho_factor(self._B(), lower=True)[0]

    @cached
    def _r(self):
        teta = self._sites.eta
        A1 = self._A1()
        QB1Qt = self._QB1Qt()
        K = self.K()

        u = self.m() + K.dot(teta)
        A1u = A1 * u
        return A1u - A1*QB1Qt.dot(A1u) - teta
