from __future__ import division

from numpy import ones
from numpy import sqrt
from numpy import array
from numpy import asarray
from numpy import empty
from numpy import hstack
from numpy.testing import assert_almost_equal
from numpy.random import RandomState

from limix_math.linalg import qs_decomposition

from limix_qep.ep import PoissonEP

import lim


def test_poisson_optimize():

    random = RandomState(10)
    nsamples = 200
    nfeatures = 1200

    G = random.randn(nsamples, nfeatures)
    G = (G - G.mean(0)) / G.std(0)
    G /= sqrt(200)

    mean = lim.mean.OffsetMean()
    mean.offset = 0.0
    mean.set_data(nsamples, 'sample')

    cov1 = lim.cov.LinearCov()
    cov1.set_data((G, G), 'sample')

    cov2 = lim.cov.EyeCov()
    a = lim.util.fruits.Apples(nsamples)
    cov2.set_data((a, a), 'sample')

    cov1.scale = 1.0
    cov2.scale = 1.0

    cov = lim.cov.SumCov([cov1, cov2])

    link = lim.link.LogLink()
    lik = lim.lik.Poisson(0, link)
    y = lim.random.GLMMSampler(lik, mean, cov).sample(random)

    (Q, S) = qs_decomposition(G)

    ep = PoissonEP(y, ones((nsamples, 1)), Q[0], Q[1], S[0])
    print(ep.lml())
    ep.optimize()
    print(ep.lml())
    # assert_almost_equal(ep.lml(), -827.22936074000131, decimal=4)

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])


# from numpy import ones
# from numpy import array
# from numpy import empty
# from numpy import hstack
# from numpy import random
# from numpy.testing import assert_almost_equal
#
# from limix_math.linalg import qs_decomposition
#
# from limix_qep.ep import PoissonEP
# from limix_qep.ep import BinomialEP
# # from limix_qep.ep import PoissonEP2
#
# from limix_qep.tool.util import create_binomial
#
#
# def test_poisson_lml():
#
#     random.seed(6)
#     n = 500
#     nfeatures = 10
#
#     M = ones((n, 1)) * 1.
#     G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
#
#     (Q, S) = qs_decomposition(G)
#
#     ntrials = random.randint(1, 20, n)
#     (nsuc, G) = create_binomial(n, nfeatures, ntrials, seed=10, offset=-1.)
#
#     (Q, S) = qs_decomposition(G)
#
#     print(nsuc)
#     ep1 = BinomialEP(nsuc, ntrials, M, Q[0], Q[1], S[0])
#     ep2 = PoissonEP(nsuc, M, Q[0], Q[1], S[0])
#     # ep3 = PoissonEP2(nsuc, M, Q[0], Q[1], S[0])
#
#     print(ep2.lml())
#     # print(ep2.lml())
#
#     ep2.optimize()
#     print(ep2.lml())
# #     # ep1.beta = array([1.])
# #     # ep1.genetic_variance = 1.
# #     # ep1.environmental_variance = 1e-7
# #     # lml1 = ep1.lml()
# #     # print(lml1)
# #     #
# #     # ep2 = BernoulliEP(y, M, hstack(Q), empty((n, 0)), hstack(S) + 1.0)
# #     # ep2.beta = array([1.])
# #     # ep2.genetic_variance = 1.
# #     # lml2 = ep2.lml()
# #     #
# #     # assert_almost_equal(lml1 - lml2, 0., decimal=5)
