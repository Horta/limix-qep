from __future__ import absolute_import
import logging
import numpy as np
from numpy import dot
from scipy.linalg import lu_factor
from limix_math.linalg import ddot, sum2diag, dotd, sum2diag_inplace
from limix_math.linalg import solve, cho_solve, lu_solve
from limix_math.linalg import stl
from limix_math.dist.norm import logpdf, logcdf
from limix_math.dist.beta import isf as bisf
from hcache import Cached, cached
from limix_qep.special.nbinom_moms import moments_array3, init
from limix_qep.lik import Binomial, Bernoulli
from limix_math.array import issingleton
from .dists import SiteLik
from .dists import Joint
from .dists import Cavity
from .fixed_ep import FixedEP
from .config import _MAX_ITER, _EP_EPS, _PARAM_EPS
from scipy import optimize
from scipy.misc import logsumexp
import scipy as sp
import scipy.stats
from .haseman_elston import heritability
import limix_ext as lxt

_MAX_COND = 1e6
def _is_zero(x):
    return abs(x) < 1e-9

_NALPHAS0 = 100
_ALPHAS0 = bisf(0.55, 2., 1.-
                np.linspace(0, 1, _NALPHAS0+1, endpoint=False)[1:])
_NALPHAS1 = 30
_ALPHAS1_EPS = 1e-3
_DEFAULT_NINTP = 100

def _process_S(S):
    cond = np.max(S) / np.min(S)
    if cond > _MAX_COND:
        eps = (_MAX_COND * np.min(S) - np.max(S)) / (1 - _MAX_COND)
        logger = logging.getLogger(__name__)
        logger.info("The covariance matrix's conditioning number" +
                    " is too high: %e. Summing %e to its eigenvalues and " +
                    "renormalizing for a better conditioning number.",
                    cond, eps)
        m = np.mean(S)
        S += eps
        S *= m / np.mean(S)

class BernPredictorEP(object):
    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov

    def logpdf(self, y):
        ind = 2*y - 1
        return logcdf(ind * self._mean / np.sqrt(1 + self._cov))

    def pdf(self, y):
        return np.exp(self.logpdf(y))

class BinomPredictorEP(object):
    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov
        self._tau = 1./cov
        self._eta = self._tau * mean

    def logpdf(self, y, ntrials):
        # ind = 2*y - 1
        # return logcdf(ind * self._mean / np.sqrt(1 + self._cov))
        y = np.asarray(y, float)
        ntrials = np.asarray(ntrials, float)
        y = np.atleast_1d(y)
        ntrials = np.atleast_1d(ntrials)
        n = len(y)
        lmom0 = np.empty(n)
        mu1 = np.empty(n)
        var2 = np.empty(n)

        moments_array3(ntrials, y, self._eta, self._tau, lmom0, mu1, var2)
        return lmom0

    def pdf(self, y, ntrials):
        return np.exp(self.logpdf(y, ntrials))


# K = \sigma_g^2 Q (S + \delta I) Q.T
class EP(Cached):
    def __init__(self, y, M, Q, S, outcome_type=None,
                 nintp=_DEFAULT_NINTP):
        Cached.__init__(self)
        self._logger = logging.getLogger(__name__)
        outcome_type = Bernoulli() if outcome_type is None else outcome_type
        self._outcome_type = outcome_type
        assert y.shape[0] == M.shape[0], 'Number of individuals mismatch.'
        assert y.shape[0] == Q.shape[0], 'Number of individuals mismatch.'

        if issingleton(y):
            raise ValueError("The phenotype array has a single unique value" +
                             " only.")

        if S.min() <= 0:
            raise Exception("The provided covariance matrix is not" +
                            " positive-definite because the minimum eigvalue" +
                            " is %f." % S.min())

        _process_S(S)

        self._nchol = 0

        self._params_initialized = False
        self._nsamples = y.shape[0]
        self._y = np.asarray(y, float)
        self._M = M
        self._S = S

        self._update_calls = 0

        self._Q = Q

        self._psites = SiteLik(self._nsamples)
        self._sites = SiteLik(self._nsamples)

        self._cavs = Cavity(self._nsamples)

        self._joint = Joint(Q, S)

        self._sigg2 = None
        self._beta = None

        if isinstance(outcome_type, Bernoulli):
            self._y11 = 2. * self._y - 1.0
            self._delta = 0.
        else:
            self._delta = 1.
            init(nintp)
            self._mu1 = np.empty(self._nsamples)
            self._var2 = np.empty(self._nsamples)

        self._loghz = np.empty(self._nsamples)

        self._prev_sigg2 = np.inf
        self._prev_delta = np.inf
        self._prev_beta = np.full(M.shape[1], np.inf)

        self._K = None
        if not self._use_low_rank_trick():
            self._K = self._Q.dot(self._SQt())

        self._nfuncost = 0

        self._covariate_setup()
        self._last_ep_error = np.inf
        self._logger.debug('An EP object has been initialized'+
                           ' with outcome type %s.', type(outcome_type))

    def _covariate_setup(self):
        M = self._M
        v = M.std(0)
        m = M.mean(0)
        ok = np.full(M.shape[1], True, dtype=bool)

        idx = np.where(v == 0.)[0]
        if len(idx) > 1:
            ok[idx[1:]] = False
        ok[np.logical_and(v == 0., m == 0.)] = False

        self._Mok = ok

    def _use_low_rank_trick(self):
        return self._Q.shape[0] - self._S.shape[0] > 1

    def _LU(self):
        Sd = self.sigg2 * (self._S + self.delta)
        Q = self._Q
        ttau = self._sites.tau
        R = ddot(Sd, Q.T, left=True)
        R = ddot(ttau, dot(Q, R), left=True)
        sum2diag_inplace(R, 1.0)
        return lu_factor(R, overwrite_a=True, check_finite=False)

    def _LUM(self):
        m = self._m
        ttau = self._sites.tau
        teta = self._sites.eta
        LU = self._LU()
        return lu_solve(LU, ttau * m + teta)

    def predict(self, m, var, covar):
        m = np.atleast_1d(m)
        var = np.atleast_2d(var)
        covar = np.atleast_2d(covar)

        if isinstance(self._outcome_type, Binomial):
            return self._predict_binom(m, var, covar)

        # if covar.ndim == 1:
        #     assert isnumber(var) and isnumber(m)
        # elif covar.ndim == 2:
        #     assert len(var) == covar.shape[0]
        # else:
        #     raise ValueError("covar has a wrong layout.")

        A1 = self._A1()
        L1 = self._L1()
        Q = self._Q
        mu = m + dot(covar, self._A1tmuLm())

        A1cov = ddot(A1, covar.T, left=True)
        part3 = dotd(A1cov.T, dot(Q, cho_solve(L1, dot(Q.T, A1cov))))
        sig2 = var.diagonal() - dotd(A1cov.T, covar.T) + part3

        if isinstance(self._outcome_type, Bernoulli):
            return BernPredictorEP(mu, sig2)
        else:
            return BinomPredictorEP(mu, sig2)
        #
        # if covar.ndim == 1:
        #     p = dict()
        #     p[1] = np.exp(logcdf(mu / np.sqrt(1 + sig2)))
        #     p[0] = 1 - p[1]
        # else:
        #     v = np.exp(logcdf(mu / np.sqrt(1 + sig2)))
        #     p = [dict([(0, 1-vi), (1, vi)]) for vi in v]
        #
        # return p

    def _predict_binom(self, m, var, covar):
        m = np.atleast_1d(m)
        var = np.atleast_2d(var)
        covar = np.atleast_2d(covar)

        mu = m + covar.dot(self._LUM())

        LU = self._LU()
        ttau = self._sites.tau

        sig2 = var.diagonal() - dotd(covar, lu_solve(LU, ddot(ttau, covar.T, left=True)))

        return BinomPredictorEP(mu, sig2)

    def K(self):
        Q = self._Q
        S = self._S
        sigg2 = self._sigg2
        delta = self._delta
        left = sigg2 * dot(Q, ddot(S, Q.T, left=True))
        return sum2diag(left, sigg2 * delta)

    @property
    def M(self):
        return self._M
    @M.setter
    def M(self, value):
        self._M = value
        self.clear_cache('_m')
        self.clear_cache('_lml_components')
        self.clear_cache('_update')
        self.clear_cache('_A1tmuLm')
        self._beta = np.zeros(value.shape[1])
        self._covariate_setup()

    def h2(self, **kwargs):
        sigg2 = self.sigg2
        varc = np.var(dot(self.M, self.beta))
        delta = self.delta

        ot = self._outcome_type
        if isinstance(ot, Bernoulli):
            assert delta == 0.
            return sigg2 / (sigg2 + varc + 1.)
        elif isinstance(ot, Binomial):
            ign_samp_noise = False
            if 'ign_samp_noise' in kwargs:
                ign_samp_noise = kwargs['ign_samp_noise']
            if ign_samp_noise:
                return sigg2 / (sigg2 + sigg2*delta + varc)
            return sigg2 / (sigg2 + sigg2*delta + varc + 1.)

        assert False

    def h2tosigg2(self, h2):
        # sigg2 = self.sigg2
        varc = np.var(dot(self.M, self.beta))
        delta = self.delta

        ot = self._outcome_type
        if isinstance(ot, Bernoulli):
            assert delta == 0.
            # return h2 = sigg2 / (sigg2 + varc + 1.)
            return h2 * (1 + varc) / (1 - h2)
        elif isinstance(ot, Binomial):
            # return h2 = sigg2 / (sigg2 + sigg2*delta + varc + 1.)
            return h2 * (1 + varc) / (1 - h2 - delta*h2)

        assert False

    @property
    def beta(self):
        if self._beta is None:
            self._beta = np.empty(self._M.shape[1])
        return self._beta.copy()
    @beta.setter
    def beta(self, value):
        self.clear_cache('_lml_components')
        self.clear_cache('_m')
        # self.clear_cache('_L1')
        # self.clear_cache('_B1')
        self.clear_cache('_update')
        self.clear_cache('_A1tmuLm')
        if self._beta is None:
            self._beta = np.empty(self._M.shape[1])
        self._beta[:] = value

    @property
    def sigg2(self):
        if self._sigg2 is None:
            if self._K is None:
                self._sigg2 = 1.0
            else:
                p = sum(self._y) / float(self._y.shape[0])
                self._sigg2 = max(1e-3, lxt.lmm.h2(self._y, self._M, self._K, p))
                # self._sigg2 = max(1e-3, self.h2tosigg2(heritability(self._y, self._K)))
            print("Initial h2 sigg2 guess: %.5f %.5f" % (self.h2(), self._sigg2))
        return self._sigg2
    @sigg2.setter
    def sigg2(self, value):
        self.clear_cache('_lml_components')
        self.clear_cache('_L1')
        # self.clear_cache('_B1')
        self.clear_cache('_sigg2dotdQSQt')
        self.clear_cache('_update')
        self.clear_cache('_A1tmuLm')
        self._sigg2 = max(value, 1e-4)

    @property
    def delta(self):
        return self._delta
    @delta.setter
    def delta(self, value):
        if isinstance(self._outcome_type, Bernoulli):
            assert value == 0.
        self.clear_cache('_lml_components')
        self.clear_cache('_L1')
        self.clear_cache('_QtA1Q')
        # self.clear_cache('_B1')
        self.clear_cache('_update')
        self.clear_cache('_A1tmuLm')
        self._delta = value

    def _A0T(self):
        if _is_zero(self.delta):
            return np.zeros(self._Q.shape[0])
        v = self.sigg2 * self.delta
        return 1./(1./v + self._sites.tau)

    def _A0A0T(self):
        if _is_zero(self.delta):
            return np.ones(self._Q.shape[0])
        v = self.sigg2 * self.delta
        return (1./v)/(1./v + self._sites.tau)

    def _A0(self):
        return 1./(self.sigg2 * self.delta)

    def _A1(self):
        ttau = self._sites.tau
        return self._A0A0T() * ttau

    def _A2(self):
        A0 = self._A0()
        ttau = self._sites.tau
        return 1./(A0 + ttau)

    def _B0(self):
        A0 = self._A0()
        Q = self._Q
        sigg2 = self._sigg2
        S = self._S
        B0 = sum2diag(dot(Q.T, A0 * Q), S/sigg2)
        return B0

    def _L0(self):
        L0 = np.linalg.cholesky(self._B0())
        return L0

    @cached
    def _QtA1Q(self):
        Q = self._Q
        A1 = self._A1()
        # before = time()
        return_ = dot(Q.T, ddot(A1, Q, left=True))
        # self._time_elapsed['_QtA1Q'] += time() - before
        # self._calls['_QtA1Q'] += 1
        return return_

    @cached
    def _SQt(self):
        return ddot(self._S, self._Q.T, left=True)

    @cached
    def _dotdQSQt(self):
        Q = self._Q
        return dotd(Q, self._SQt())

    @cached
    def _sigg2dotdQSQt(self):
        return self.sigg2 * self._dotdQSQt()

    @cached
    def _A1tmuLm(self):
        return self._A1tmuL() - self._A1mL()

    def _A1tmuL(self):
        A1 = self._A1()
        L1 = self._L1()
        Q = self._Q
        if _is_zero(self.delta):
            A1tmu = self._sites.eta
        else:
            A1tmu = A1*self._sites.tau
        return A1tmu - A1*dot(Q, cho_solve(L1, dot(Q.T, A1tmu)))

    def _A1mL(self):
        m = self._m
        A1 = self._A1()
        A1m = A1*m
        L1 = self._L1()
        Q = self._Q
        return A1m - A1*dot(Q, cho_solve(L1, dot(Q.T, A1m)))

    def _B1(self):
        sigg2 = self._sigg2
        S = self._S
        sigg2S = sigg2 * S
        B1 = sum2diag(self._QtA1Q(), 1./sigg2S)
        return B1

    @cached
    def _L1(self):
        B1 = self._B1()
        self._nchol += 1
        L1 = sp.linalg.cho_factor(B1, lower=True)[0]
        return L1

    @cached
    def _lml_components(self):
        self._update()
        Q = self._Q
        S = self._S
        m = self._m
        sigg2 = self.sigg2
        ttau = self._sites.tau
        teta = self._sites.eta
        ctau = self._cavs.tau
        ceta = self._cavs.eta
        cmu = self._cavs.mu
        A1 = self._A1()

        sigg2S = sigg2 * S
        tctau = ttau + ctau
        A1m = A1*m

        L1 = self._L1()

        p1 = - np.sum(np.log(np.diagonal(L1)))
        p1 += - 0.5 * np.log(sigg2S).sum()
        p1 += 0.5 * np.log(A1).sum()

        A0T = self._A0T()
        A0A0T = self._A0A0T()

        p3 = np.sum(teta*teta*A0T)
        A0A0pT_teta = A0A0T * teta
        QtA0A0pT_teta = dot(Q.T, A0A0pT_teta)
        L1 = self._L1()
        L1_0 = stl(L1, QtA0A0pT_teta)
        p3 += dot(L1_0.T, L1_0)
        vv = 2*np.log(np.abs(teta))
        p3 -= np.exp(logsumexp(vv - np.log(tctau)))
        p3 *= 0.5

        p4 = 0.5 * np.sum(ceta * (ttau * cmu - 2*teta) / tctau)

        f0 = None
        L1_1 = cho_solve(L1, QtA0A0pT_teta)
        f0 = A1 * dot(Q, L1_1)
        p5 = dot(m, A0A0pT_teta) - dot(m, f0)

        p6 = - 0.5 * np.sum(m * A1m) +\
            0.5 * dot(A1m, dot(Q, cho_solve(L1, dot(Q.T, A1m))))

        p7 = 0.5 * (np.log(ttau + ctau).sum() - np.log(ctau).sum())

        p7 -= 0.5 * np.log(ttau).sum()

        p8 = self._loghz.sum()

        return (p1, p3, p4, p5, p6, p7, p8, f0, A0A0pT_teta)

    def lml(self):
        (p1, p3, p4, p5, p6, p7, p8, _, _) = self._lml_components()
        return p1 + p3 + p4 + p5 + p6 + p7 + p8

    def fixed_ep(self):
        self._update()
        (p1, p3, p4, _, _, p7, p8, f0, A0A0pT_teta) =\
            self._lml_components()

        lml_nonbeta_part = p1 + p3 + p4 + p7 + p8
        Q = self._Q
        L1 = self._L1()
        A1 = self._A1()
        opt_bnom = self._opt_beta_nom()
        vv1 = FixedEP(lml_nonbeta_part, A0A0pT_teta, f0,\
                        A1, L1, Q, opt_bnom)

        return vv1

    def site_means(self):
        return self._sites.eta / self._sites.tau

    def site_vars(self):
        return 1. / self._sites.tau

    def site_zs(self):
        lzh = self._loghz
        smus = self.site_means()
        svars = self.site_vars()
        cmus = self._cavs.mu
        cvars = self._cavs.sig2
        lzs = lzh + np.log(2*np.pi)/2 + np.log(svars + cvars)/2\
            + (smus - cmus)**2/(2*(svars + cvars))
        return np.exp(lzs)

    @cached
    def _update(self):
        self._logger.debug('Main EP loop has started.')
        m = self._m
        Q = self._Q
        S = self._S
        sigg2 = self.sigg2
        delta = self.delta

        ttau = self._sites.tau
        teta = self._sites.eta

        SQt = self._SQt()
        sigg2dotdQSQt = self._sigg2dotdQSQt()

        # before2 = time()
        if not self._params_initialized:
            # before3 = time()
            outcome_type = self._outcome_type
            y = self._y

            if not self._use_low_rank_trick():
                self._K = self._Q.dot(self._SQt())
                if self._sigg2 is None:
                    print("Initial h2 sigg2 guess: %.5f %.5f" % (self.h2(), self.sigg2))
                    self.sigg2 = max(1e-3, self.h2tosigg2(heritability(y, self._K)))
            else:
                # tirar isso daqui
                if self._sigg2 is None:
                    self.sigg2 = self.h2tosigg2(heritability(y, self._Q.dot(self._SQt())))

            if isinstance(outcome_type, Bernoulli) and self._beta is None:
                ratio = sum(y) / float(y.shape[0])
                f = sp.stats.norm(0,1).isf(1-ratio)
                self._beta = np.linalg.lstsq(self._M, np.full(y.shape[0], f))[0]
            else:
                self._beta = np.zeros(self._M.shape[1])
                self._joint.initialize(m, sigg2, delta)

            # self._time_elapsed['eploop_init'] += time() - before3
            # self._calls['eploop_init'] += 1
            self._params_initialized = True
        else:
            self._joint.update(m, sigg2, delta, S, Q, self._L1(),
                                     ttau, teta, self._A1(), self._A0T(),
                                     sigg2dotdQSQt, SQt, self._K)
        self._update_calls += 1

        i = 0
        NMAX = 10
        while i < NMAX:
            self._logger.debug('Iteration %d.', i)
            self._psites.tau[:] = ttau
            self._psites.eta[:] = teta

            self._logger.debug('step 1')
            self._cavs.update(self._joint.tau, self._joint.eta, ttau, teta)
            self._logger.debug('step 2')
            (hmu, hsig2) = self._tilted_params()
            if not np.all(np.isfinite(hsig2)) or np.any(hsig2 == 0.):
                raise Exception('Error: not np.all(np.isfinite(hsig2))' +
                                ' or np.any(hsig2 == 0.).')
            self._logger.debug('step 3')
            self._sites.update(self._cavs.tau, self._cavs.eta, hmu, hsig2)
            self.clear_cache('_L1')
            self.clear_cache('_QtA1Q')
            self.clear_cache('_A1tmuLm')
            self._joint.update(m, sigg2, delta, S, Q, self._L1(),
                                     ttau, teta, self._A1(), self._A0T(),
                                     sigg2dotdQSQt, SQt, self._K)
            self._logger.debug('step 4')
            tdiff = np.abs(self._psites.tau - ttau)
            ediff = np.abs(self._psites.eta - teta)
            aerr = tdiff.max() + ediff.max()
            self._logger.debug('step 5')

            if self._psites.tau.min() <= 0. or (0. in self._psites.eta):
                rerr = np.inf
            else:
                rtdiff = tdiff/np.abs(self._psites.tau)
                rediff = ediff/np.abs(self._psites.eta)
                rerr = rtdiff.max() + rediff.max()

            i += 1
            self._logger.debug('step 6')
            # self._ttau[self.sigg2].append(self._sites.tau.copy())
            self._last_ep_error = min(aerr, rerr)
            if aerr < 2*_EP_EPS or rerr < 2*_EP_EPS:
                break

        if i + 1 == _MAX_ITER:
            self._logger.warn('Maximum number of EP iterations has'+
                              ' been attained.')

        self._logger.debug('Main EP loop has finished '+
                          'performing %d iterations.', i+1)

    def _tilted_params(self):
        otype = self._outcome_type

        if isinstance(otype, Bernoulli):
            return self._tilted_params_bernoulli()
        elif isinstance(otype, Binomial):
            return self._tilted_params_binomial()
        else:
            assert False, 'Wrong provided likelihood.'

    # \hat z
    def _compute_hz(self):
        if isinstance(self._outcome_type, Bernoulli):
            b = np.sqrt(self._cavs.tau**2 + self._cavs.tau)
            c = self._y11 * self._cavs.eta / b
            self._loghz[:] = logcdf(c)
        else:
            self._tilted_params_binomial()

    def _tilted_params_bernoulli(self):
        # self._calls['tilted_params_bernoulli'] += 1
        # before = time()
        b = np.sqrt(self._cavs.tau**2 + self._cavs.tau)
        lb = np.log(b)
        c = self._y11 * self._cavs.eta / b
        lcdf = self._loghz
        lcdf[:] = logcdf(c)
        lpdf = logpdf(c)
        mu = self._cavs.eta / self._cavs.tau + self._y11 * np.exp(lpdf - (lcdf + lb))

        sig2 = 1./self._cavs.tau - np.exp(lpdf - (lcdf + 2*lb)) * (c + np.exp(lpdf - lcdf))

        # self._time_elapsed['tilted_params_bernoulli'] += time() - before

        return (mu, sig2)

    def _tilted_params_binomial(self):

        binom = self._outcome_type
        N = binom.ntrials

        ctau = self._cavs.tau
        ceta = self._cavs.eta
        K = self._y
        mu1 = self._mu1
        var2 = self._var2

        lmom0 = self._loghz
        # self._nbinom_moms.moments_array2(N, K, ceta, ctau, lmom0, mu1, var2)
        moments_array3(N, K, ceta, ctau, lmom0, mu1, var2)

        mu = mu1
        sig2 = var2

        return (mu, sig2)

    @property
    @cached
    def _m(self):
        return dot(self._M, self.beta)

    def _opt_beta_nom(self):
        A0T = self._A0T()
        A1 = self._A1()
        L1 = self._L1()
        Q = self._Q
        ttau = self._sites.tau
        teta = self._sites.eta

        D = ttau * A0T
        c = teta - D * teta
        d = cho_solve(L1, dot(Q.T, c))
        u = c - A1 * dot(Q, d)
        return u

    def _opt_beta_denom(self):
        A1 = self._A1()
        M = self._M
        ok = self._Mok
        Q = self._Q
        L1 = self._L1()
        A1M = ddot(A1, M[:,ok], left=True)
        QtA1M = dot(Q.T, A1M)
        return dot(M[:,ok].T, A1M) - dot(A1M.T, dot(Q, cho_solve(L1, QtA1M)))

    def _optimal_beta(self):
        self._update()

        if np.all(np.abs(self._M) < 1e-15):
            return np.zeros_like(self._beta)

        M = self._M

        u = self._opt_beta_nom()
        u = dot(M.T, u)

        Z = self._opt_beta_denom()

        try:
            ok = self._Mok
            obeta = np.zeros_like(self._beta)
            with np.errstate(all='raise'):
                obeta[ok] = solve(Z, u[ok])
        except (np.linalg.LinAlgError, FloatingPointError):
            self._logger.warn('Failed to compute the optimal beta.'+
                              ' Zeroing it.')
            obeta = np.zeros_like(self._beta)

        return obeta

    def _optimize_beta(self):
        # self._calls['beta'] += 1
        # before = time()
        pbeta = np.empty_like(self._beta)

        max_ii = 100
        max_ii = 1
        ii = 0
        while ii < max_ii:
            ii += 1
            pbeta[:] = self._beta
            self.beta = self._optimal_beta()
            if np.sum((self.beta - pbeta)**2) < _PARAM_EPS:
                break

        # self._logger.debug('Number of iterations for beta optimization: %d.',
        #                   ii)
        # self._time_elapsed['beta'] += time() - before

    def _best_alpha0(self, alpha1, opt_beta):
        min_cost = np.inf
        alpha0_min = None
        for i in range(_NALPHAS0):
            (sigg2, delta) = _alphas2hyperparams(_ALPHAS0[i], alpha1)

            self.sigg2 = sigg2
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

    def _create_fun_cost_sigg2(self, opt_beta):
        def fun_cost(h2):
            # before = time()
            print("Trying h2 sigg2: %.5f %.5f" % (h2, self.h2tosigg2(h2)))
            self.sigg2 = self.h2tosigg2(h2)
            # self._update()
            if opt_beta:
                self._optimize_beta()
            cost = -self.lml()
            # self._time_elapsed['sigg2'] += time() - before
            # self._calls['sigg2'] += 1
            return cost
        return fun_cost

    def _optimize_step1(self, opt_beta=True, opt_sigg2=True, opt_delta=None,
                        disp=False):
        self._logger.debug("Starting optimization step 1.")

        if opt_delta is None:
            opt_delta = isinstance(self._outcome_type, Binomial)

        if not opt_delta and opt_sigg2:
            opt = dict(xatol=_PARAM_EPS, disp=disp)
            gs = 0.5 * (3.0 - np.sqrt(5.0))
            sigg2_left = 1e-4
            h2_left = sigg2_left / (sigg2_left + 1)
            curr_h2 = self.h2()
            h2_right = (curr_h2 + h2_left * gs - h2_left) / gs
            h2_right = min(h2_right, 0.967)


            bounds_h2 = [h2_left, h2_right]
            print("H2 bound: (%.5f, %.5f)" % (h2_left, h2_right))
            fun_cost = self._create_fun_cost_sigg2(opt_beta)
            res = optimize.minimize_scalar(fun_cost,
                                              options=opt,
                                              bounds=bounds_h2,
                                              method='Bounded')
            self.sigg2 = self.h2tosigg2(res.x)
        elif opt_delta and opt_sigg2:
            fun_cost = self._create_fun_cost_both(opt_beta)
            opt = dict(xatol=_ALPHAS1_EPS, maxiter=_NALPHAS1, disp=disp)
            res = optimize.minimize_scalar(fun_cost, options=opt,
                                              bounds=(_ALPHAS1_EPS,
                                                      1-_ALPHAS1_EPS),
                                              method='Bounded')
            alpha1 = res.x
            alpha0 = self._best_alpha0(alpha1, opt_beta)[0]

            (self.sigg2, self.delta) = _alphas2hyperparams(alpha0, alpha1)
        elif opt_delta and not opt_sigg2:
            assert False
            # fun_cost = self._create_fun_cost_step1_keep_sigg2(disp, opt_beta)
            # opt = dict(xatol=_PARAM_EPS, maxiter=30, disp=disp)
            # res = optimize.minimize_scalar(fun_cost, options=opt,
            #                                   bounds=bounds_delta,
            #                                   method='Bounded')
            # self.delta = res.x

        if opt_beta:
            self._optimize_beta()

        self._logger.debug("End of optimization step 1.")

    def optimize(self, opt_beta=True, opt_sigg2=True, opt_delta=None,
                 disp=False):

        self._logger.debug("Starting optimization.")
        self._optimize_step1(opt_beta, opt_sigg2, opt_delta, disp)
        self._logger.debug("Parameters: sigg2=%e, delta=%e).",
                           self.sigg2, self.delta)
        self._logger.debug("End of optimization.")

def _alphas2hyperparams(alpha0, alpha1):
    a0 = 1-alpha0
    a1 = 1-alpha1
    sigg2 = alpha0 / (a0*a1)
    delta = (a0 * alpha1)/alpha0
    return (sigg2, delta)
