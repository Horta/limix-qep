from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from time import time

import logging

from hcache import cached

from limix_math.linalg import sum2diag

from numpy import set_printoptions
from numpy import argmin
from numpy import zeros
from numpy import array
from .util import greek_letter

from .dists import Joint
from .base import EP
from ._minimize_scalar import find_minimum


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

    # Getters and setters
    @property
    def noise_ratio(self):
        e = self.environmental_variance
        i = self.instrumental_variance
        return e / (e + i)

    @noise_ratio.setter
    def noise_ratio(self, value):
        gr = self.genetic_ratio
        i = self.instrumental_variance
        self.environmental_variance = i * value / (1 - value)
        self.genetic_ratio = gr

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
        self._e = value

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

    @property
    def genetic_ratio(self):
        e = self.environmental_variance
        i = self.instrumental_variance
        v = self.genetic_variance
        return v / (e + i + v)

    @genetic_ratio.setter
    def genetic_ratio(self, value):
        e = self.environmental_variance
        i = self.instrumental_variance
        c = e + i
        self.genetic_variance = value * c / (1 - value)

    # Optimization related methods
    def _get_optimal_delta(self, real_variance):
        rvs = array(self._real_variances)
        i = argmin(abs(rvs - real_variance))
        return self._environmental_genetic_ratios[i]

    def _eg_delta_cost(self, delta):
        self.environmental_genetic_ratio = delta
        self._optimize_beta()
        c = -self.lml()
        self._logger.debug("_environmental_genetic_cost:cost: %.15f", c)
        return c

    def _environmental_genetic_delta_cost(self, delta):
        self._logger.debug("_environmental_genetic_cost:delta: %.15f.", delta)
        return self._eg_delta_cost(delta)

    def _noise_ratio_cost(self, nr):
        self.noise_ratio = nr
        self._optimize_beta()
        c = -self.lml()
        self._logger.debug("_noise_ratio_cost:nr_cost: %.10f %.15f", nr, c)
        return c

    def _find_best_noise_ratio(self):
        start = time()

        a, b = 1e-3, 1 - 1e-3

        nr = self.noise_ratio
        nr, nfev = find_minimum(self._noise_ratio_cost, nr, a=a, b=b,
                                rtol=0, atol=1e-4)

        self.noise_ratio = nr

        self._logger.debug("End of noise_ratio optimization (%.3f seconds" +
                           ", %d function calls).", time() - start, nfev)

    def _genetic_ratio_cost(self, gr):
        self._logger.debug("_genetic_ratio_cost:gr: %.10f", gr)
        self.genetic_ratio = gr
        self._find_best_noise_ratio()
        return -self.lml()

    def optimize(self, progress=None):
        start = time()

        self._logger.debug("Start of optimization.")
        # self._logger.debug(self.__str__())

        a, b = 1e-3, 1 - 1e-3

        gr = self.genetic_ratio

        def func(gr):
            if progress:
                progress.update(func.i)
            func.i += 1
            return self._genetic_ratio_cost(gr)
        func.i = 0

        gr, nfev = find_minimum(func, gr, a=a, b=b, rtol=0, atol=1e-4)

        self.genetic_ratio = gr
        self._find_best_noise_ratio()
        self._optimize_beta()

        # self._logger.debug(self.__str__())
        self._logger.debug("End of optimization (%.3f seconds" +
                           ", %d function calls).", time() - start, nfev)

    # Key but Intermediary Matrix Definitions
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
        return sum2diag(self.genetic_variance * self._Q0S0Q0t(),
                        self.environmental_variance)

    @cached
    def diagK(self):
        return (self.genetic_variance * self._diagQ0S0Q0t() +
                self.environmental_variance)

    @cached
    def _Q0B1Q0t(self):
        if self.genetic_variance < 1e-6:
            nsamples = self._Q0.shape[0]
            return zeros((nsamples, nsamples))
        return super(OverdispersionEP, self)._Q0B1Q0t()

    def __repr__(self):
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
        gr = self.genetic_ratio
        nr = self.noise_ratio
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
  Heritability:          {h2}
  Genetic ratio:         {gr}
  Noise ratio:           {nr}
  """.format(v="%.4f" % v, e="%.4f" % e, b=beta, Q0=indent(bytes(Q0)),
             Q1=indent(bytes(Q1)), S0=bytes(S0), M=indent(bytes(M)),
             tvar="%.4f" % tvar, cvar="%.4f" % cvar, h2="%.4f" % h2,
             ivar="%.4f" % ivar, gr="%.4f" % gr, nr="%.4f" % nr)
        set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                         precision=8, suppress=False, threshold=1000,
                         formatter=None)
        return s

    def __str__(self):
        return self.__unicode__().encode('utf-8')

    def __unicode__(self):
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
        gr = self.genetic_ratio
        nr = self.noise_ratio
        var_sym = unichr(0x3bd).encode('utf-8')
        s = """
Latent phenotype:
  f_i = o_i + u_i + e_i

Definitions:
  o: fixed-effects signal
     M {b}.T
  u: background signal
     Normal(0, {v} * Kinship)
  e: environmental signal
     Normal(0, {e} * I)

Log marginal likelihood: {lml}

Statistics (latent space):
  Total variance:        {tvar}     {vs}_o + {vs}_u + {vs}_e + {vs}_{eps}
  Instrumental variance: {ivar}     {vs}_{eps}
  Covariates variance:   {cvar}     {vs}_o
  Heritability:          {h2}     {vs}_u / ({vs}_o + {vs}_u + {vs}_e)
  Genetic ratio:         {gr}     {vs}_u / ({vs}_u + {vs}_e + {vs}_{eps})
  Noise ratio:           {nr}     {vs}_e / ({vs}_e + {vs}_{eps})
  """.format(v="%7.4f" % v, e="%7.4f" % e, b=beta, Q0=indent(bytes(Q0)),
             Q1=indent(bytes(Q1)), S0=bytes(S0), M=indent(bytes(M)),
             tvar="%7.4f" % tvar, cvar="%7.4f" % cvar, h2="%7.4f" % h2,
             ivar="%7.4f" % ivar, gr="%7.4f" % gr, nr="%7.4f" % nr,
             vs=var_sym, eps=greek_letter('epsilon'),
             lml="%9.6f" % self.lml())
        set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                         precision=8, suppress=False, threshold=1000,
                         formatter=None)
        return s
