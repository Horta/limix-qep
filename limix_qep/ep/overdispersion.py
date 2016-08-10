from __future__ import absolute_import
from __future__ import division

import logging

from scipy.optimize import minimize_scalar

from hcache import cached

from limix_math.linalg import sum2diag

from .config import HYPERPARAM_EPS
from .dists import Joint
from .base import EP
from .bracket import bracket

from numpy import set_printoptions
from numpy import exp
from numpy import clip
from numpy import argmin
from numpy import zeros
from numpy import array
from numpy import log
from numpy import sign


# K = Q (v S + e I) Q.T
class OverdispersionEP(EP):
    """
    .. math::
        K = Q (v S + e I) Q.T
        M = svd_U svd_S svd_V.T
        \\tilde \\beta = svd_S^{1/2} svd_V.T \\beta
        \\tilde M = svd_U svd_S^{1/2} \\tilde \\beta
        m = M \\beta
        m = \\tilde M \\tilde \\beta
    """
    def __init__(self, M, Q0, Q1, S0, Q0S0Q0t=None):
        super(OverdispersionEP, self).__init__(M, Q0, S0, Q0S0Q0t=Q0S0Q0t)
        self._logger = logging.getLogger(__name__)

        nsamples = M.shape[0]
        self._joint = Joint(nsamples)
        self._Q1 = Q1
        self._e = None
        self._calls_environmental_genetic_delta_cost = 0
        self._calls_real_variance_cost = 0

        self._environmental_genetic_ratios = []
        self._real_variances = []
        self._delta_zero_cost = None
        self._delta_one_cost = None
        self._bracket_delta_x = (None, None, None)
        self._bracket_delta_f = (None, None, None)
        self._delta_bounds = (1e-9, 1 - 1e-9)

    def initialize_hyperparams(self):
        raise NotImplementedError

    ############################################################################
    ############################################################################
    ######################## Getters and setters ###############################
    ############################################################################
    ############################################################################
    @property
    def noise_ratio(self):
        e = self.environmental_variance
        i = self.instrumental_variance
        return e / (e + i)

    @noise_ratio.setter
    def noise_ratio(self, value):
        value = min(value, 1-1e-7)
        i = self.instrumental_variance
        self.environmental_variance = i * value / (1 - value)

    @property
    def environmental_variance(self):
        if self._e is None:
            self.initialize_hyperparams()
        return self._e

    @environmental_variance.setter
    def environmental_variance(self, value):
        self.clear_cache('_lml_components')
        self.clear_cache('_L')
        self.clear_cache('_update')
        self.clear_cache('_A0')
        self.clear_cache('diagK')
        self.clear_cache('K')
        self._e = max(value, 1e-7)

    @property
    def instrumental_variance(self):
        return 1.

    @property
    def real_variance(self):
        return self.genetic_variance + self.environmental_variance

    @real_variance.setter
    def real_variance(self, value):
        c = value / (self.genetic_variance + self.environmental_variance)
        self.genetic_variance *= c
        self.environmental_variance *= c

    @property
    def environmental_genetic_ratio(self):
        e = self.environmental_variance
        v = self.genetic_variance
        return e / (e + v)

    @environmental_genetic_ratio.setter
    def environmental_genetic_ratio(self, value):
        t = self.real_variance
        self.genetic_variance = t * (1 - value)
        self.environmental_variance = t * value

    ############################################################################
    ############################################################################
    #################### Optimization related methods ##########################
    ############################################################################
    ############################################################################
    def _get_optimal_delta(self, real_variance):
        rvs = array(self._real_variances)
        i = argmin(abs(rvs - real_variance))
        return self._environmental_genetic_ratios[i]

    def _delta_direction(self, real_variance):
        if (len(self._real_variances) <= 1 or
                len(self._environmental_genetic_ratios) <= 1):
            return +1

        preal = self._real_variances[-2]
        pdelta = self._environmental_genetic_ratios[-2]

        real = self._real_variances[-1]
        delta = self._environmental_genetic_ratios[-1]

        direction = sign(delta - pdelta) * sign(real - preal)
        return sign(real_variance - real) * direction

    def _delta_step(self, real_variance):
        if (len(self._real_variances) <= 1
                or len(self._environmental_genetic_ratios) <= 1):
            return 0.01

        preal = self._real_variances[-2]
        pdelta = self._environmental_genetic_ratios[-2]

        real = self._real_variances[-1]
        delta = self._environmental_genetic_ratios[-1]

        if abs(real - preal) < 1e-7:
            return 1e-4

        c = abs(delta - pdelta) / abs(real - preal)
        return max(1e-4, abs(real_variance - real) * c)

    # def _environmental_genetic_delta_cost(self, logit_delta):
    #     delta = 1/(1 + exp(-logit_delta))
    #     self._logger.debug("_environmental_genetic_cost:logit: %.15f.", logit_delta)
    #     self._logger.debug("_environmental_genetic_cost:delta: %.15f.", delta)
    #     self.environmental_genetic_ratio = delta
    #     self._optimize_beta()
    #     c = -self.lml()
    #     self._logger.debug("_environmental_genetic_cost:cost: %.15f", c)
    #     self._calls_environmental_genetic_delta_cost += 1
    #     return c

    def _eg_delta_cost(self, delta):
        self.environmental_genetic_ratio = delta
        self._optimize_beta()
        c = -self.lml()
        self._logger.debug("_environmental_genetic_cost:cost: %.15f", c)
        self._calls_environmental_genetic_delta_cost += 1
        return c

    def _environmental_genetic_delta_cost(self, delta):
        self._environmental_genetic_ratios_tmp.append(delta)
        self._logger.debug("_environmental_genetic_cost:delta: %.15f.", delta)

        if len(self._environmental_genetic_ratios_tmp) > 8 and abs(self._environmental_genetic_ratios_tmp[-1]-self._environmental_genetic_ratios_tmp[-2]) < 1e-6 and abs(self._environmental_genetic_ratios_tmp[-2]-self._environmental_genetic_ratios_tmp[-3]) < 1e-6:
            import ipdb; ipdb.set_trace()
        # if abs(delta-0.002190636952218) < 1e-10:
            # import ipdb; ipdb.set_trace()

        if self._bracket_delta_x[0] == delta:
            return self._bracket_delta_f[0]
        if self._bracket_delta_x[1] == delta:
            return self._bracket_delta_f[1]
        if self._bracket_delta_x[2] == delta:
            return self._bracket_delta_f[2]

        delta_bounds = self._delta_bounds

        if delta <= delta_bounds[0]:
            if self._delta_zero_cost is None:
                self._delta_zero_cost = self._eg_delta_cost(delta_bounds[0])
            return self._delta_zero_cost + delta_bounds[0] - delta

        elif delta >= delta_bounds[1]:
            if self._delta_one_cost is not None:
                self._delta_one_cost = self._eg_delta_cost(delta_bounds[1])
            return self._delta_one_cost - delta_bounds[1] + delta

        return self._eg_delta_cost(delta)

    def _real_variance_cost(self, lnrv):
        rv = exp(lnrv)
        self._logger.debug("_real_variance_cost:rv: %.5f.", rv)

        dd = self._delta_direction(rv)
        ds = self._delta_step(rv)

        self.real_variance = rv
        self._environmental_genetic_ratios_tmp = []

        delta_bounds = self._delta_bounds

        delta0 = self.environmental_genetic_ratio
        step = dd * ds

        delta1 = clip(delta0+step, delta_bounds[0], delta_bounds[1])

        self._delta_zero_cost = None
        self._delta_one_cost = None

        self._logger.debug("BRACKET BEGIN")
        xa, xb, xc, fa, fb, fc, _ =\
            bracket(self._environmental_genetic_delta_cost, delta0, delta1,
                    bounds=delta_bounds)
        self._logger.debug("BRACKET END")

        # swap so xa < xc can be assumed
        if (xa > xc):
            xc, xa = xa, xc
            fc, fa = fa, fc

        if not ((xa < xb) and (xb < xc)):
            self._logger.debug("bracket only.")
            if fa <= fc:
                self.environmental_genetic_ratio = xa
                cost = fa
            elif fb <= fc:
                self.environmental_genetic_ratio = xb
                cost = fb
            else:
                self.environmental_genetic_ratio = xc
                cost = fc

        else:
            self._bracket_delta_x = (xa, xb, xc)
            self._bracket_delta_f = (fa, fb, fc)

            r = minimize_scalar(self._environmental_genetic_delta_cost,
                                bracket=[xa, xb, xc],
                                method='Brent',
                                tol=HYPERPARAM_EPS*10)

            self.environmental_genetic_ratio = r.x
            cost = -self.lml()

        self._real_variances.append(self.real_variance)
        self._environmental_genetic_ratios.append(self.environmental_genetic_ratio)

        self._calls_real_variance_cost += 1
        return cost

    def optimize(self):
        from time import time

        start = time()

        self._logger.debug("Start of optimization.")
        self._logger.debug(self.__str__())

        self._real_variances = []
        self._environmental_genetic_ratios = []

        rv = self.real_variance

        r = minimize_scalar(self._real_variance_cost,
                            bracket=[log(rv), log(rv+1.0)],
                            method='Brent',
                            tol=HYPERPARAM_EPS)

        rv = exp(r.x)
        self.real_variance = rv
        self.environmental_genetic_ratio = self._get_optimal_delta(rv)

        self._logger.debug("Optimizer info: %s.", str(r))

        if not r.success:
            self._logger.warn("Optimizer has failed: %s.", str(r))

        nfev = r.nfev

        self._optimize_beta()

        self._logger.debug(self.__str__())
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
        return ttau / (self.environmental_variance * ttau + 1)

    @cached
    def _A2(self):
        """:math:`\\tilde{\\mathrm T}^{-1} \\mathrm A_1`"""
        ttau = self._sites.tau
        return 1 / (self.environmental_variance * ttau + 1)

    @cached
    def K(self):
        """:math:`K = (v Q S Q.T + e I)`"""
        return sum2diag(self.genetic_variance * self._Q0S0Q0t(), self.environmental_variance)

    @cached
    def diagK(self):
        return self.genetic_variance * self._diagQ0S0Q0t() + self.environmental_variance

    @cached
    def _Q0B1Q0t(self):
        if self.genetic_variance < 1e-6:
            nsamples = self._Q0.shape[0]
            return zeros((nsamples, nsamples))
        return super(OverdispersionEP, self)._Q0B1Q0t()

    def __str__(self):
        v = self.genetic_variance
        e = self.environmental_variance
        beta = self.beta
        M = self.M
        Q0 = self._Q0
        Q1 = self._Q1
        S0 = self._S0
        tvar = self.total_variance
        set_printoptions(precision=3, threshold=10)
        def indent(s):
            final = []
            for si in s.split('\n'):
                final.append('      ' + si)
            return '\n'.join(final)
        cvar = self.covariates_variance
        ivar = self.instrumental_variance
        h2 = self.heritability
        s = """
Prior:
  Normal(M {b}.T, {v} * Kinship + {e} * I)

Definitions:
  Kinship = Q0 S0 Q0.T
  I       = environment
  M       = covariates effect

Input data:
  M:
{M}
  Q0:
{Q0}
  Q1:
{Q1}
  S0: {S0}

Statistics (latent space):
  Total variance:        {tvar}
  Instrumental variance: {ivar}
  Covariates variance:   {cvar}
  Heritability:          {h2}""".format(v="%.4f" % v, e="%.4f" % e, b=beta,
                          Q0=indent(bytes(Q0)), Q1=indent(bytes(Q1)),
                          S0=bytes(S0), M=indent(bytes(M)),
                          tvar="%.4f" % tvar, cvar="%.4f" % cvar,
                          h2="%.4f" % h2, ivar="%.4f" % ivar)
        set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                         precision=8, suppress=False, threshold=1000,
                         formatter=None)
        return s
