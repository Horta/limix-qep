import logging
import numpy as np
from numpy import dot
from limix_math.linalg import ddot, sum2diag
from limix_math.linalg import solve, cho_solve
from limix_math.linalg import trace2
from limix_math.dist.norm import logpdf, logcdf
from limix_math.dist.beta import isf as bisf
from hcache import Cached, cached
# from limix_qep.special.nbinom_moms import NBinomMoms
from limix_qep.special.nbinom_moms import moments_array3, init
from limix_qep.lik import Binomial, Bernoulli
from dists import SiteLik
from dists import Joint
from dists import Cavity
from fixed_ep import FixedEP
from config import _MAX_ITER, _EP_EPS, _PARAM_EPS
from scipy import optimize
from scipy.misc import logsumexp

class PrecisionError(Exception):
    pass

_MAX_COND = 1e6
def _is_zero(x):
    return abs(x) < 1e-9

_NALPHAS0 = 100
_ALPHAS0 = bisf(0.55, 2., 1.-
                np.linspace(0, 1, _NALPHAS0+1, endpoint=False)[1:])
# _ALPHAS0 = st.beta(0.55, 2.).isf(1.-
#                         np.linspace(0, 1, _NALPHAS0+1, endpoint=False)[1:])
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
    return S


# K = \sigma_g^2 Q (S + \delta I) Q.T
class EP(Cached):
    def __init__(self, y, M, Q, S, outcome_type=None, X=None,
                 nintp=_DEFAULT_NINTP):
        Cached.__init__(self)
        self._logger = logging.getLogger(__name__)
        outcome_type = Bernoulli() if outcome_type is None else outcome_type
        self._outcome_type = outcome_type
        assert y.shape[0] == M.shape[0], 'Number of individuals mismatch.'
        assert y.shape[0] == Q.shape[0], 'Number of individuals mismatch.'
        if S.min() <= 0:
            raise Exception("The provided covariance matrix is not" +
                            " positive-definite because the minimum eigvalue" +
                            " is %f." % S.min())

        S = _process_S(S)

        self._params_initialized = False
        self._nsamples = y.shape[0]
        self._y = np.asarray(y, float)
        self._M = M
        self._S = S

        self._Q = Q
        self._X = X

        self._psites = SiteLik(self._nsamples)
        self._sites = SiteLik(self._nsamples)

        self._cavs = Cavity(self._nsamples)

        self._joint = Joint(Q, S)

        if isinstance(outcome_type, Bernoulli):
            self._y11 = 2. * self._y - 1.0
            self._delta = 0.
        else:
            self._delta = 1.
            # self._nbinom_moms = NBinomMoms(nintp)
            init(nintp)
            self._mu1 = np.empty(self._nsamples)
            self._var2 = np.empty(self._nsamples)

        self._beta = np.zeros(M.shape[1])
        self._sigg2 = 1.0
        self._loghz = np.empty(self._nsamples)
        self._freeze_this_thing = False

        self._nfuncost = 0
        self._covariate_setup()
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

    # --------------------------------------------------------#
    # ---------------------- Interface ---------------------- #
    # --------------------------------------------------------#
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
        self.clear_cache('_L1')
        self.clear_cache('_B1')
        self.clear_cache('_update')
        self._beta = np.zeros(value.shape[1])
        self._covariate_setup()

    @property
    def h2(self):
        sigg2 = self.sigg2
        varc = np.var(dot(self.M, self.beta))
        delta = self.delta

        ot = self._outcome_type
        if isinstance(ot, Bernoulli):
            assert delta == 0.

        return sigg2 / (sigg2 + sigg2*delta + varc + 1.)

    @property
    def beta(self):
        return self._beta.copy()
    @beta.setter
    def beta(self, value):
        self.clear_cache('_lml_components')
        self.clear_cache('_m')
        self.clear_cache('_L1')
        self.clear_cache('_B1')
        self.clear_cache('_update')
        self._beta[:] = value

    @property
    def sigg2(self):
        return self._sigg2
    @sigg2.setter
    def sigg2(self, value):
        self.clear_cache('_lml_components')
        self.clear_cache('_L1')
        self.clear_cache('_B1')
        self.clear_cache('_update')
        self._sigg2 = value

    @property
    def delta(self):
        return self._delta
    @delta.setter
    def delta(self, value):
        if isinstance(self._outcome_type, Bernoulli):
            assert value == 0.
        self.clear_cache('_lml_components')
        self.clear_cache('_L1')
        self.clear_cache('_B1')
        self.clear_cache('_update')
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

    def _QtA1Q(self):
        Q = self._Q
        A1 = self._A1()
        return dot(Q.T, ddot(A1, Q, left=True))

    @cached
    def _B1(self):
        sigg2 = self._sigg2
        S = self._S
        sigg2S = sigg2 * S
        B1 = sum2diag(self._QtA1Q(), 1./sigg2S)
        return B1

    @cached
    def _L1(self):
        L1 = np.linalg.cholesky(self._B1())
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
        L1_0 = np.linalg.solve(L1, QtA0A0pT_teta)
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

    def lml_grad(self):
        self._update()
        A0 = self._A0()
        A1 = self._A1()
        L1 = self._L1()

        Q = self._Q
        S = self._S
        delta = self.delta
        sigg2 = self.sigg2

        m = self._m

        ttau = self._sites.tau
        teta = self._sites.eta

        C = A0 / (A0 + ttau)
        Cteta = C * teta

        p1 = Cteta - A1 * dot(Q, cho_solve(L1, dot(Q.T, Cteta)))

        p2 = A1 * m - A1 * dot(Q, cho_solve(L1, dot(Q.T, A1 * m)))

        A1Q = ddot(A1, Q, left=True)
        SQt = ddot(S, Q.T, left=True)
        L1_QtA1 = cho_solve(L1, ddot(Q.T, A1, left=False))
        L1_QtA1Q = dot(L1_QtA1, Q)
        p3_sigg2 = trace2(A1Q, SQt) + np.sum(A1 * delta) - trace2(dot(ddot(S, A1Q.T, left=True), Q), L1_QtA1Q) -\
                delta * trace2(A1Q, L1_QtA1)

        p3_delta = sigg2 * np.sum(A1) - sigg2 * trace2(A1Q, L1_QtA1)

        Qtp2 = dot(Q.T, p2)
        deltap2 = delta * p2
        dKsigg2p2 = dot(Q, ddot(S, Qtp2, left=True)) + deltap2

        Qtp1 = dot(Q.T, p1)
        deltap1 = delta * p1
        dKsigg2p1 = dot(Q, ddot(S, Qtp1, left=True)) + deltap1

        dsigg2 = 0.5 * dot(p2, dKsigg2p2) - dot(p2, dKsigg2p1) + 0.5 * dot(p1, dKsigg2p1) -\
            0.5 * p3_sigg2

        dKdeltap1 = sigg2 * p1
        dKdeltap2 = sigg2 * p2

        ddelta = 0.5 * dot(p2, dKdeltap2) - dot(p2, dKdeltap1) + 0.5 * dot(p1, dKdeltap1) -\
            0.5 * p3_delta

        return (ddelta, dsigg2)


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

    def get_site_likelihoods(self):
        return self._sites.copy()

    def set_site_likelihoods(self, s):
        self._sites.tau[:] = s.tau
        self._sites.eta[:] = s.eta

        self.clear_cache('_lml_components')
        self.clear_cache('_L1')
        self.clear_cache('_B1')
        self.clear_cache('_update')

        m = self._m
        sigg2 = self.sigg2
        delta = self.delta
        S = self._S
        Q = self._Q
        self._joint.update(m, sigg2, delta, S, Q, self._L1(),
                           s.tau, s.eta, self._A1(), self._A0T())
        self._cavs.update(self._joint.tau, self._joint.eta, s.tau, s.eta)
        self._compute_hz()

    # --------------------------------------------------------#
    # ---------------------- MAIN LOOP ---------------------- #
    # --------------------------------------------------------#
    @cached
    def _update(self):
        if self._freeze_this_thing:
            return

        self._logger.debug('Main EP loop has started.')
        m = self._m
        Q = self._Q
        S = self._S
        sigg2 = self.sigg2
        delta = self.delta

        ttau = self._sites.tau
        teta = self._sites.eta

        if not self._params_initialized:
            self._joint.initialize(m, sigg2, delta)
            self._params_initialized = True
        else:
            self._joint.update(m, sigg2, delta, S, Q, self._L1(),
                                     ttau, teta, self._A1(), self._A0T())

        i = 0
        while i < _MAX_ITER:

            self._psites.tau[:] = ttau
            self._psites.eta[:] = teta

            self._cavs.update(self._joint.tau, self._joint.eta, ttau, teta)

            (hmu, hsig2) = self._tilted_params()
            if not np.all(np.isfinite(hsig2)) or np.any(hsig2 == 0.):
                raise Exception('Error: not np.all(np.isfinite(hsig2))' +
                                ' or np.any(hsig2 == 0.).')

            self._sites.update(self._cavs.tau, self._cavs.eta, hmu, hsig2)
            self.clear_cache('_L1')
            self.clear_cache('_B1')
            self._joint.update(m, sigg2, delta, S, Q, self._L1(),
                                     ttau, teta, self._A1(), self._A0T())

            tdiff = np.abs(self._psites.tau - ttau)
            ediff = np.abs(self._psites.eta - teta)
            aerr = tdiff.max() + ediff.max()


            if self._psites.tau.min() <= 0. or (0. in self._psites.eta):
                rerr = np.inf
            else:
                rtdiff = tdiff/np.abs(self._psites.tau)
                rediff = ediff/np.abs(self._psites.eta)
                rerr = rtdiff.max() + rediff.max()

            if aerr < 2*_EP_EPS or rerr < 2*_EP_EPS:
                break

            i += 1


        if i + 1 == _MAX_ITER:
            self._logger.warn('Maximum number of EP iterations has'+
                              ' been attained.')

        self._logger.debug('Main EP loop has finished '+
                          'performing %d iterations.', i+1)

    # -------------------------------------------------------#
    # ------------------- TILTED METHODS ------------------- #
    # -------------------------------------------------------#
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
        b = np.sqrt(self._cavs.tau**2 + self._cavs.tau)
        lb = np.log(b)
        c = self._y11 * self._cavs.eta / b
        lcdf = self._loghz
        lcdf[:] = logcdf(c)
        lpdf = logpdf(c)
        mu = self._cavs.eta / self._cavs.tau + self._y11 * np.exp(lpdf - (lcdf + lb))

        sig2 = 1./self._cavs.tau - np.exp(lpdf - (lcdf + 2*lb)) * (c + np.exp(lpdf - lcdf))

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

    def _opt_beta_denom_old(self):
        A1 = self._A1()
        M = self._M
        Q = self._Q
        L1 = self._L1()
        A1M = ddot(A1, M, left=True)
        QtA1M = dot(Q.T, A1M)
        return dot(M.T, A1M) - dot(A1M.T, dot(Q, cho_solve(L1, QtA1M)))

    def _opt_beta_denom(self):
        A1 = self._A1()
        M = self._M
        ok = self._Mok
        Q = self._Q
        L1 = self._L1()
        A1M = ddot(A1, M[:,ok], left=True)
        QtA1M = dot(Q.T, A1M)
        return dot(M[:,ok].T, A1M) - dot(A1M.T, dot(Q, cho_solve(L1, QtA1M)))

    # -----------------------------------------------------------#
    # ---------------------- OPTIMIZATION ---------------------- #
    # -----------------------------------------------------------#
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
            obeta[ok] = solve(Z, u[ok])
        except np.linalg.LinAlgError:
            self._logger.warn('Failed to compute the optimal beta.'+
                              ' Zeroing it.')
            obeta = np.zeros_like(self._beta)

        return obeta

    def _optimize_beta(self):
        pbeta = np.empty_like(self._beta)

        max_ii = 100
        ii = 0
        while ii < max_ii:
            ii += 1
            pbeta[:] = self._beta
            self.beta = self._optimal_beta()
            if np.sum((self.beta - pbeta)**2) < _PARAM_EPS:
                break

        self._logger.debug('Number of iterations for beta optimization: %d.',
                          ii)

    def _best_alpha0(self, alpha1, opt_beta):
        min_cost = np.inf
        alpha0_min = None
        for i in xrange(_NALPHAS0):

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
        def fun_cost(sigg2):
            self.sigg2 = sigg2
            self._update()
            if opt_beta:
                self._optimize_beta()
            cost = -self.lml()
            return cost
        return fun_cost

    def _optimize_step1(self, opt_beta=True, opt_sigg2=True, opt_delta=None,
                        disp=False):
        self._logger.debug("Starting optimization step 1.")

        if opt_delta is None:
            opt_delta = isinstance(self._outcome_type, Binomial)

        if not opt_delta and opt_sigg2:
            opt = dict(xatol=_PARAM_EPS, disp=disp)
            bounds_sigg2 = (1e-4, 30.0)
            fun_cost = self._create_fun_cost_sigg2(opt_beta)
            res = optimize.minimize_scalar(fun_cost,
                                              options=opt,
                                              bounds=bounds_sigg2,
                                              method='Bounded')
            self.sigg2 = res.x
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
        self._logger.debug("End of optimization.")

def _alphas2hyperparams(alpha0, alpha1):
    a0 = 1-alpha0
    a1 = 1-alpha1
    sigg2 = alpha0 / (a0*a1)
    delta = (a0 * alpha1)/alpha0
    return (sigg2, delta)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    np.random.seed(5)
    ntrials = 3
    n = 5
    p = n+4

    M = np.ones((n, 1)) * 0.4
    G = np.random.randint(3, size=(n, p))
    G = np.asarray(G, dtype=float)
    G -= G.mean(axis=0)
    G /= G.std(axis=0)
    G /= np.sqrt(p)

    K = dot(G, G.T) + np.eye(n)*0.1
    (S, Q) = np.linalg.eigh(K)

    y = np.array([1., 0., 1., 1., 1.])
    ep = EP(y, M, Q, S)
    ep.optimize()
