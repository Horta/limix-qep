from __future__ import absolute_import
import logging
import numpy as np

from numpy import dot
from numpy import sqrt
from numpy import empty
from numpy import asarray
from numpy import empty_like
from numpy import atleast_1d
from numpy import atleast_2d
from numpy import var as variance

from limix_math.linalg import ddot
from limix_math.linalg import sum2diag
from limix_math.linalg import dotd
from limix_math.linalg import solve
from limix_math.linalg import cho_solve
from limix_math.linalg import stl

from hcache import Cached, cached

from limix_math.array import issingleton
from limix_math.linalg import economic_svd

from .dists import SiteLik
from .dists import Joint
from .dists import Cavity
from .fixed_ep import FixedEP
from .config import _MAX_ITER, _EP_EPS
from .config import _HYPERPARAM_EPS
from scipy.optimize import minimize_scalar
from scipy.misc import logsumexp
import scipy as sp

from .util import make_sure_reasonable_conditioning


class EP(Cached):
    """
    .. math::
        K = \\sigma_g^2 Q S Q.T
        M = svd_U svd_S svd_V.T
        \\tilde \\beta = svd_S^{1/2} svd_V.T \\beta
        \\tilde M = svd_U svd_S^{1/2} \\tilde \\beta
        m = M \\beta
        m = \\tilde M \\tilde \\beta
    """
    def __init__(self, y, M, Q, S, QSQt=None):
        Cached.__init__(self)
        self._logger = logging.getLogger(__name__)

        if issingleton(y):
            raise ValueError("The phenotype array has a single unique value" +
                             " only.")

        if not np.all(np.isfinite(y)):
            raise ValueError("There are non-finite numbers in phenotype.")

        if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(S)):
            raise ValueError("There are non-finite numbers in the provided" +
                             " eigen decomposition.")

        assert y.shape[0] == M.shape[0], 'Number of individuals mismatch.'
        assert y.shape[0] == Q.shape[0], 'Number of individuals mismatch.'

        if S.min() <= 0:
            raise ValueError("The provided covariance matrix is not" +
                            " positive-definite because the minimum eigvalue" +
                            " is %f." % S.min())

        make_sure_reasonable_conditioning(S)

        self._y = asarray(y, float)
        self._covariate_setup(M)
        self._S = S
        self._Q = Q
        self.__QSQt = QSQt
        self._K = None

        nsamples = y.shape[0]

        self._psites = SiteLik(nsamples)
        self._sites = SiteLik(nsamples)
        self._cavs = Cavity(nsamples)
        self._joint = Joint(Q, S)

        self._sigg2 = None
        self._delta = 0.
        self.__tbeta = None

        self._loghz = empty(nsamples)

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
        (p1, p3, p4, _, _, p7, p8, f0, A0A0pT_teta) =\
            self._lml_components()

        lml_nonbeta_part = p1 + p3 + p4 + p7 + p8
        Q = self._Q
        L = self._L()
        A = self._A()
        opt_bnom = self._opt_beta_nom()
        vv1 = FixedEP(lml_nonbeta_part, A0A0pT_teta, f0,\
                        A, L, Q, opt_bnom)

        return vv1

    def _posterior_normal(self, m, var, covar):
        m = atleast_1d(m)
        var = atleast_2d(var)
        covar = atleast_2d(covar)

        A = self._A()
        L = self._L()
        Q = self._Q
        mu = m + dot(covar, self._AtmuLm())

        Acov = ddot(A, covar.T, left=True)
        part3 = dotd(Acov.T, dot(Q, cho_solve(L, dot(Q.T, Acov))))
        sig2 = var.diagonal() - dotd(Acov.T, covar.T) + part3

        return (mu, sig2)

    @cached
    def _init_ep_params(self):
        self._logger.debug("EP parameters initialization.")
        m = self.m()
        sigg2 = self.sigg2
        self._joint.initialize(m, sigg2)
        self._sites.initialize()

    def _init_sigg2(self):
        raise NotImplementedError

    def _init_beta(self):
        raise NotImplementedError

    @cached
    def K(self):
        return self.sigg2 * self._QSQt()

    @cached
    def m(self):
        return dot(self._tM, self._tbeta)


    ############################################################################
    ############################################################################
    ######################## Getters and setters ###############################
    ############################################################################
    ############################################################################
    def h2(self):
        sigg2 = self.sigg2
        varc = variance(self.m())
        return sigg2 / (sigg2 + varc + 1.)

    def h2tosigg2(self, h2):
        varc = variance(self.m())
        return h2 * (1 + varc) / (1 - h2)

    @property
    def _tbeta(self):
        return self.__tbeta

    @_tbeta.setter
    def _tbeta(self, value):
        self.clear_cache('_lml_components')
        self.clear_cache('m')
        self.clear_cache('_update')
        self.clear_cache('_AtmuLm')
        if self.__tbeta is None:
            self.__tbeta = asarray(value, float).copy()
        else:
            self.__tbeta[:] = value

    @property
    def beta(self):
        if self.__tbeta is None:
            self._init_beta()
        return solve(self._svd_V.T, self._tbeta/self._svd_S12)

    @beta.setter
    def beta(self, value):
        if self.__tbeta is None:
            self._init_beta()
        self._tbeta = self._svd_S12 * dot(self._svd_V.T, value)

    @property
    def sigg2(self):
        if self._sigg2 is None:
            self._init_sigg2()
        return self._sigg2

    @sigg2.setter
    def sigg2(self, value):
        self.clear_cache('_lml_components')
        self.clear_cache('_L')
        self.clear_cache('_sigg2dotdQSQt')
        self.clear_cache('_update')
        self.clear_cache('_AtmuLm')
        self._sigg2 = max(value, 1e-4)

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        self._covariate_setup(value)
        self.clear_cache('m')
        self.clear_cache('_lml_components')
        self.clear_cache('_update')
        self.clear_cache('_AtmuLm')

    ############################################################################
    ############################################################################
    ###################### Log Marginal Likelihood #############################
    ############################################################################
    ############################################################################
    @cached
    def _lml_components(self):
        self._update()
        Q = self._Q
        S = self._S
        m = self.m()
        sigg2 = self.sigg2
        ttau = self._sites.tau
        teta = self._sites.eta
        ctau = self._cavs.tau
        ceta = self._cavs.eta
        cmu = self._cavs.mu
        A = self._A()

        sigg2S = sigg2 * S
        tctau = ttau + ctau
        Am = A*m

        L = self._L()

        p1 = - np.sum(np.log(np.diagonal(L)))
        p1 += - 0.5 * np.log(sigg2S).sum()
        p1 += 0.5 * np.log(A).sum()

        p3 = 0.0
        A0A0pT_teta = teta
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
        f0 = A * dot(Q, L_1)
        p5 = dot(m, A0A0pT_teta) - dot(m, f0)

        p6 = - 0.5 * np.sum(m * Am) +\
            0.5 * dot(Am, dot(Q, cho_solve(L, dot(Q.T, Am))))

        p7 = 0.5 * (np.log(ttau + ctau).sum() - np.log(ctau).sum())

        p7 -= 0.5 * np.log(ttau).sum()

        p8 = self._loghz.sum()

        return (p1, p3, p4, p5, p6, p7, p8, f0, A0A0pT_teta)

    def lml(self):
        (p1, p3, p4, p5, p6, p7, p8, _, _) = self._lml_components()
        return p1 + p3 + p4 + p5 + p6 + p7 + p8

    ############################################################################
    ############################################################################
    ########################### Main EP loop ###################################
    ############################################################################
    ############################################################################
    @cached
    def _update(self):
        self._init_ep_params()

        self._logger.debug('EP loop has started.')
        m = self.m()
        Q = self._Q
        S = self._S
        sigg2 = self.sigg2

        ttau = self._sites.tau
        teta = self._sites.eta

        SQt = self._SQt()
        sigg2dotdQSQt = self._sigg2dotdQSQt()

        i = 0
        while i < 10:
            self._psites.tau[:] = ttau
            self._psites.eta[:] = teta

            self._cavs.update(self._joint.tau, self._joint.eta, ttau, teta)
            (hmu, hsig2) = self._tilted_params()

            if not np.all(np.isfinite(hsig2)) or np.any(hsig2 == 0.):
                raise Exception('Error: not np.all(np.isfinite(hsig2))' +
                                ' or np.any(hsig2 == 0.).')

            self._sites.update(self._cavs.tau, self._cavs.eta, hmu, hsig2)
            self.clear_cache('_L')
            self.clear_cache('_QtAQ')
            self.clear_cache('_AtmuLm')

            self._joint.update(m, sigg2, S, Q, self._L(),
                                     teta, self._A(),
                                     sigg2dotdQSQt, SQt, self._K)

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
            if aerr < 2*_EP_EPS or rerr < 2*_EP_EPS:
                break

        if i + 1 == _MAX_ITER:
            self._logger.warn('Maximum number of EP iterations has'+
                              ' been attained.')

        self._logger.debug('EP loop has performed %d iterations.', i+1)


    ############################################################################
    ############################################################################
    ################## Optimization of hyperparameters #########################
    ############################################################################
    ############################################################################
    def _optimal_tbeta_nom(self):
        A = self._A()
        L = self._L()
        Q = self._Q
        teta = self._sites.eta

        d = cho_solve(L, dot(Q.T, teta))
        return teta - A * dot(Q, d)

    def _optimal_tbeta_denom(self):
        A = self._A()
        tM = self._tM
        Q = self._Q
        L = self._L()
        AM = ddot(A, tM, left=True)
        QtAM = dot(Q.T, AM)
        return dot(tM.T, AM) - dot(AM.T, dot(Q, cho_solve(L, QtAM)))

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
        ptbeta = empty_like(self._tbeta)

        step = -1.
        i = 0
        while step < _HYPERPARAM_EPS and i < 5:
            ptbeta[:] = self._tbeta
            self._optimal_tbeta()
            step = np.sum((self._tbeta - ptbeta)**2)
            self._logger.debug("Beta step: %e.", step)
            i += 1

        self._logger.debug("Beta optimization performed %d steps " +
                           "to find %s.", i, bytes(self._tbeta))

    def _nlml(self, h2, opt_beta):
        sigg2 = self.h2tosigg2(h2)
        self._logger.debug("Evaluating for h2:%e, sigg2:%e", h2, sigg2)
        self.sigg2 = sigg2
        if opt_beta:
            self._optimize_beta()
        return -self.lml()

    def _h2_bounds(self):
        # golden ratio
        gs = 0.5 * (3.0 - np.sqrt(5.0))
        sigg2_left = 1e-4
        h2_left = sigg2_left / (sigg2_left + 1)
        curr_h2 = self.h2()
        h2_right = (curr_h2 + h2_left * gs - h2_left) / gs
        h2_right = min(h2_right, 0.967)
        h2_bounds = (h2_left, h2_right)

        self._logger.debug("H2 bound: (%.5f, %.5f)", h2_left, h2_right)

        return h2_bounds

    def optimize(self, opt_beta=True, opt_sigg2=True, disp=False):

        self._logger.debug("Start of optimization.")
        self._logger.debug("Initial parameters: sigg2=%e, beta=%s).",
                           self.sigg2, bytes(self.beta))

        if opt_sigg2:
            opt = dict(xatol=_HYPERPARAM_EPS, disp=disp)

            r = minimize_scalar(self._nlml, options=opt,
                                bounds=self._h2_bounds(),
                                method='Bounded', args=opt_beta)
            self.sigg2 = self.h2tosigg2(r.x)
            self._logger.debug("%d function evaluations request by the" +
                               " optimizer.", r.nfev)
            self._logger.debug("Optimizer message: %s.", r.message)
            if r.status != 0:
                self._logger.warn("Optimizer failed with status %d.", r.status)

        if opt_beta:
            self._optimize_beta()

        self._logger.debug("End of optimization.")

    ############################################################################
    ############################################################################
    ############## Key but Intermediary Matrix Definitions #####################
    ############################################################################
    ############################################################################
    def _A(self):
        """ $\tilde T$ """
        return self._sites.tau

    @cached
    def _QtAQ(self):
        """ $Q^t A Q$ """
        Q = self._Q
        A = self._A()
        return dot(Q.T, ddot(A, Q, left=True))

    @cached
    def _SQt(self):
        """ $S Q^t$ """
        return ddot(self._S, self._Q.T, left=True)

    @cached
    def _dotdQSQt(self):
        """:math:`\\mathrm{Diag}\\{Q S Q^t\\}`"""
        return dotd(self._Q, self._SQt())

    def _QSQt(self):
        """ $QSQ^t$ """
        if self.__QSQt is None:
            Q = self._Q
            self.__QSQt = dot(Q, self._SQt())
        return self.__QSQt

    @cached
    def _sigg2dotdQSQt(self):
        """:math:`\\sigma_g^2 \\mathrm{Diag}\\{ Q S Q^t \\}`"""
        return self.sigg2 * self._dotdQSQt()

    @cached
    def _AtmuLm(self):
        """:math:`\\mathrm{AL}(\\eta) - \\mathrm{AL}(\\tilde T \\mathrm m)`

        .. math::
            \\tilde \\eta - \\tilde T Q B^{-1} Q^t \\tilde \\eta
            \\tilde T \\mathrm m   - \\tilde T Q B^{-1} Q^t \\tilde T m
        """
        A = self._A()
        m = self.m()
        L = self._L()
        Q = self._Q
        teta = self._sites.eta
        Am = A*m

        # $\tilde \eta - \tilde T Q B^{-1} Q^t \tilde \eta$
        AtmuL = teta - A*dot(Q, cho_solve(L, dot(Q.T, teta)))

        # $\tilde T m - \tilde T Q B^{-1} Q^t \tilde T m$
        AmL = Am - A*dot(Q, cho_solve(L, dot(Q.T, Am)))

        return AtmuL - AmL

    def _B(self):
        """:math:`B = Q^t \\tilde T Q + S^{-1} \\sigma_g^{-2}`"""
        return sum2diag(self._QtAQ(), 1./(self.sigg2 * self._S))

    @cached
    def _L(self):
        """:math:`L L^T = \\mathrm{Chol}\\{ Q^t A Q + S^{-1} \\sigma_g^{-2} \\}`"""
        return sp.linalg.cho_factor(self._B(), lower=True)[0]
