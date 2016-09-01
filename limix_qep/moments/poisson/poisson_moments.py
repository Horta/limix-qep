from __future__ import division

from numpy import arange
from numpy import sqrt
from numpy import log
from numpy import exp


from limix_math.special import normal_pdf
from limix_math.special import normal_logpdf


class PoissonMoments(object):

    def __init__(self, nintervals):
        super(PoissonMoments, self).__init__()
        self._nintervals = nintervals

    def compute(self, y, eta, tau, lmom0, mu, var):
        from scipy.integrate import quad

        normal_mu = eta / tau
        normal_var = 1 / tau

        for i in range(len(y)):
            def int0(x):
                mui = normal_mu[i]
                sai = sqrt(normal_var[i])
                n0 = normal_logpdf((x - mui) / sai) - log(sai)
                n1 = y[i] * x - exp(x) - log(arange(1, y[i] + 1)).sum()
                # print((n0, n1))
                return exp(n0 + n1)
            r = quad(int0, -30, 30)
            assert r[1] < 1e-6
            lmom0[i] = log(r[0])

        for i in range(len(y)):
            def int0(x):
                mui = normal_mu[i]
                sai = sqrt(normal_var[i])
                n0 = normal_logpdf((x - mui) / sai) - log(sai)
                n1 = y[i] * x - exp(x) - log(arange(1, y[i] + 1)).sum()
                # print((n0, n1))
                return x * exp(n0 + n1)
            r = quad(int0, -30, 30)
            assert r[1] < 1e-6
            mu[i] = r[0]

        for i in range(len(y)):
            def int0(x):
                mui = normal_mu[i]
                sai = sqrt(normal_var[i])
                n0 = normal_logpdf((x - mui) / sai) - log(sai)
                n1 = y[i] * x - exp(x) - log(arange(1, y[i] + 1)).sum()
                # print((n0, n1))
                return x * x * exp(n0 + n1)
            r = quad(int0, -30, 30)
            assert r[1] < 1e-6
            var[i] = r[0]

        mu[:] = mu / exp(lmom0)
        var[:] = var / exp(lmom0) - mu * mu
