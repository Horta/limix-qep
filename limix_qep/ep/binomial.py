from __future__ import absolute_import
from __future__ import division

import logging

from numpy import clip
from numpy import full
from numpy import asarray
from numpy import isscalar
from numpy import isfinite
from numpy import all as all_
from numpy import set_printoptions

from numpy.linalg import lstsq

from limix_math.array import issingleton

from lim.genetics import FastLMM

from .overdispersion import OverdispersionEP

from limix_qep.special.nbinom_moms import moments_array3, init

from .util import ratio_posterior
from .util import greek_letter
from .util import summation_symbol

# class BernoulliPredictor(object):
#     def __init__(self, mean, cov):
#         self._mean = mean
#         self._cov = cov
#
#     def logpdf(self, y):
#         ind = 2*y - 1
#         return logcdf(ind * self._mean / sqrt(1 + self._cov))
#
#     def pdf(self, y):
#         return exp(self.logpdf(y))

# K = v (Q S Q.T + \delta I)
class BinomialEP(OverdispersionEP):
    def __init__(self, y, ntrials, M, Q0, Q1, S0, Q0S0Q0t=None):
        super(BinomialEP, self).__init__(M, Q0, Q1, S0, Q0S0Q0t=Q0S0Q0t)
        self._logger = logging.getLogger(__name__)

        y = asarray(y, float)

        if isscalar(ntrials):
            ntrials = full(len(y), ntrials, dtype=float)
        else:
            ntrials = asarray(ntrials, float)

        self._y = y
        self._ntrials = ntrials

        if issingleton(y):
            raise ValueError("The phenotype array has a single unique value" +
                             " only.")

        if not all_(isfinite(y)):
            raise ValueError("There are non-finite numbers in phenotype.")

        assert y.shape[0] == M.shape[0], 'Number of individuals mismatch.'
        assert y.shape[0] == Q0.shape[0], 'Number of individuals mismatch.'
        assert y.shape[0] == Q1.shape[0], 'Number of individuals mismatch.'

        init(100)

    def initialize_hyperparams(self):
        from scipy.stats import norm
        y = self._y
        ntrials = self._ntrials

        ratios = ratio_posterior(y, ntrials)
        latent = norm(0, 1).isf(1 - ratios)

        Q0 = self._Q0
        Q1 = self._Q1
        flmm = FastLMM(latent, QS=[[Q0, Q1], [self._S0]])
        flmm.learn()
        gv = flmm.genetic_variance
        nv = flmm.noise_variance
        h2 = gv / (gv + nv)
        h2 = clip(h2, 0.01, 0.9)

        offset = flmm.offset
        self._tbeta = lstsq(self._tM, full(len(y), offset))[0]
        self.environmental_variance = self.instrumental_variance
        self.pseudo_heritability = h2

    # def predict(self, m, var, covar):
    #     (mu, sig2) = self._posterior_normal(m, var, covar)
    #     return BernoulliPredictor(mu, sig2)

    def _tilted_params(self):
        N = self._ntrials
        K = self._y
        ctau = self._cavs.tau
        ceta = self._cavs.eta
        lmom0 = self._loghz
        moments_array3(N, K, ceta, ctau, lmom0, self._hmu, self._hvar)

    # \hat z
    def _compute_hz(self):
        self._tilted_params()
        # b = sqrt(self._cavs.tau**2 + self._cavs.tau)
        # c = self._y11 * self._cavs.eta / b
        # self._loghz[:] = logcdf(c)















    # def _tilted_params(self):
    #     otype = self._outcome_type
    #
    #     if isinstance(otype, Bernoulli):
    #         return self._tilted_params_bernoulli()
    #     elif isinstance(otype, Binomial):
    #         return self._tilted_params_binomial()
    #     else:
    #         assert False, 'Wrong provided likelihood.'
    #
    # # \hat z
    # def _compute_hz(self):
    #     if isinstance(self._outcome_type, Bernoulli):
    #         b = np.sqrt(self._cavs.tau**2 + self._cavs.tau)
    #         c = self._y11 * self._cavs.eta / b
    #         self._loghz[:] = logcdf(c)
    #     else:
    #         self._tilted_params_binomial()
    #
    # def _tilted_params_bernoulli(self):
    #     # self._calls['tilted_params_bernoulli'] += 1
    #     # before = time()
    #     b = np.sqrt(self._cavs.tau**2 + self._cavs.tau)
    #     lb = np.log(b)
    #     c = self._y11 * self._cavs.eta / b
    #     lcdf = self._loghz
    #     lcdf[:] = logcdf(c)
    #     lpdf = logpdf(c)
    #     mu = self._cavs.eta / self._cavs.tau + self._y11 * np.exp(lpdf - (lcdf + lb))
    #
    #     sig2 = 1./self._cavs.tau - np.exp(lpdf - (lcdf + 2*lb)) * (c + np.exp(lpdf - lcdf))
    #
    #     # self._time_elapsed['tilted_params_bernoulli'] += time() - before
    #
    #     return (mu, sig2)
    #
    # def _tilted_params_binomial(self):
    #
    #     binom = self._outcome_type
    #     N = binom.ntrials
    #
    #     ctau = self._cavs.tau
    #     ceta = self._cavs.eta
    #     K = self._y
    #     mu1 = self._mu1
    #     var2 = self._var2
    #
    #     lmom0 = self._loghz
    #     # self._nbinom_moms.moments_array2(N, K, ceta, ctau, lmom0, mu1, var2)
    #     moments_array3(N, K, ceta, ctau, lmom0, mu1, var2)
    #
    #     mu = mu1
    #     sig2 = var2
    #
    #     return (mu, sig2)




    # def _create_fun_cost_sigg2(self, opt_beta):
    #     def fun_cost(h2):
    #         # before = time()
    #         print("Trying h2 sigg2: %.5f %.5f" % (h2, self.h2tosigg2(h2)))
    #         self.sigg2 = self.h2tosigg2(h2)
    #         # self._update()
    #         if opt_beta:
    #             self._optimize_beta()
    #         cost = -self.lml()
    #         # self._time_elapsed['sigg2'] += time() - before
    #         # self._calls['sigg2'] += 1
    #         return cost
    #     return fun_cost
    #
    # def _optimize_step1(self, opt_beta=True, opt_sigg2=True, opt_delta=None,
    #                     disp=False):
    #     self._logger.debug("Starting optimization step 1.")
    #
    #     if opt_delta is None:
    #         opt_delta = isinstance(self._outcome_type, Binomial)
    #
    #     if not opt_delta and opt_sigg2:
    #         opt = dict(xatol=_PARAM_EPS, disp=disp)
    #         gs = 0.5 * (3.0 - np.sqrt(5.0))
    #         sigg2_left = 1e-4
    #         h2_left = sigg2_left / (sigg2_left + 1)
    #         curr_h2 = self.heritability
    #         h2_right = (curr_h2 + h2_left * gs - h2_left) / gs
    #         h2_right = min(h2_right, 0.967)
    #
    #
    #         bounds_h2 = [h2_left, h2_right]
    #         print("H2 bound: (%.5f, %.5f)" % (h2_left, h2_right))
    #         fun_cost = self._create_fun_cost_sigg2(opt_beta)
    #         res = optimize.minimize_scalar(fun_cost,
    #                                           options=opt,
    #                                           bounds=bounds_h2,
    #                                           method='Bounded')
    #         self.sigg2 = self.h2tosigg2(res.x)
    #     elif opt_delta and opt_sigg2:
    #         fun_cost = self._create_fun_cost_both(opt_beta)
    #         opt = dict(xatol=_ALPHAS1_EPS, maxiter=_NALPHAS1, disp=disp)
    #         res = optimize.minimize_scalar(fun_cost, options=opt,
    #                                           bounds=(_ALPHAS1_EPS,
    #                                                   1-_ALPHAS1_EPS),
    #                                           method='Bounded')
    #         alpha1 = res.x
    #         alpha0 = self._best_alpha0(alpha1, opt_beta)[0]
    #
    #         (self.sigg2, self.delta) = _alphas2hyperparams(alpha0, alpha1)
    #     elif opt_delta and not opt_sigg2:
    #         assert False
    #         # fun_cost = self._create_fun_cost_step1_keep_sigg2(disp, opt_beta)
    #         # opt = dict(xatol=_PARAM_EPS, maxiter=30, disp=disp)
    #         # res = optimize.minimize_scalar(fun_cost, options=opt,
    #         #                                   bounds=bounds_delta,
    #         #                                   method='Bounded')
    #         # self.delta = res.x
    #
    #     if opt_beta:
    #         self._optimize_beta()
    #
    #     self._logger.debug("End of optimization step 1.")
    #
    # def optimize(self, opt_beta=True, opt_sigg2=True, opt_delta=None,
    #              disp=False):
    #
    #     self._logger.debug("Starting optimization.")
    #     self._optimize_step1(opt_beta, opt_sigg2, opt_delta, disp)
    #     self._logger.debug("Parameters: sigg2=%e, delta=%e).",
    #                        self.sigg2, self.delta)
    #     self._logger.debug("End of optimization.")

    def __str__(self):
        set_printoptions(precision=3, threshold=10)
        s = """
Phenotype definition:
  y_i = {sum}_{{j=1}}^{{n}} Indicator(f_i + {epsilon}_{{i,j}} > 0), where f_i is
        the latent phenotype of the i-th individual and {epsilon}_{{i,j}} is
        distributed according to Normal(0, 1).

Definitions:
  {epsilon}_{{i,j}}: instrumental signal (noise)

Input data:
  y: {y}
  d: {ntrials}""".format(y=bytes(self._y), ntrials=bytes(self._ntrials),
                         epsilon=greek_letter('epsilon'),
                         sum=summation_symbol())

        set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                         precision=8, suppress=False, threshold=1000,
                         formatter=None)
        return s + "\n" + super(BinomialEP, self).__str__()
