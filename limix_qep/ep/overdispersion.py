from __future__ import absolute_import
from __future__ import division

import logging

from scipy.optimize import minimize_scalar

from hcache import cached

from limix_math.linalg import sum2diag

from .config import HYPERPARAM_EPS
from .dists import Joint
from .util import normal_bracket
from .base import EP

from numpy import get_printoptions
from numpy import set_printoptions


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

    ############################################################################
    ############################################################################
    #################### Optimization related methods ##########################
    ############################################################################
    ############################################################################
    def _noise_ratio_cost(self, r):
        print("  - Evaluating for ratio: %.5f." % r)
        self.environmental_variance = r / (1 - r)
        h2 = self.heritability
        self.genetic_variance = h2 * self.environmental_variance / (1 - h2)
        self._optimize_beta()
        c = -self.lml()
        print("  - Cost: %.5f" % c)
        return c

    def _h2_cost(self, h2):
        # self._logger.debug("Evaluating for h2: %e.", h2)
        print("- Evaluating for h2: %.5f." % h2)
        var = self.h2tovar(h2)
        self.genetic_variance = var

        r = self.environmental_variance / (self.environmental_variance + 1)
        bracket = normal_bracket(r, 1.)
        print("    Ratio bracket: %s." % str(bracket))
        try:
            res = minimize_scalar(self._noise_ratio_cost,
                                bracket=bracket,
                                tol=HYPERPARAM_EPS, method='Brent')
        except ValueError:
            fa = self._noise_ratio_cost(bracket[0])
            fc = self._noise_ratio_cost(bracket[2])
            if fa < fc:
                r = bracket[0]
            else:
                r = bracket[2]
        else:
            r = res.x

        self.environmental_variance = r / (1 - r)
        return -self.lml()

    def optimize(self):
        super(OverdispersionEP, self).optimize()

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

    def __str__(self):
        v = self.genetic_variance
        e = self.environmental_variance
        beta = self.beta
        M = self.M
        Q0 = self._Q0
        Q1 = self._Q1
        S0 = self._S0
        tvar = self.total_variance
        printopts = get_printoptions()
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
        set_printoptions(printopts)
        return s
