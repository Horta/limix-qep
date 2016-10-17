# from __future__ import division
#
# from numpy import ones
# from numpy import sqrt
# from numpy import array
# from numpy import asarray
# from numpy import empty
# from numpy import hstack
# from numpy.testing import assert_almost_equal
# from numpy.random import RandomState
#
# from limix_math.linalg import qs_decomposition
#
# from limix_qep.ep import BinomialEP
# from limix_qep.ep import BernoulliEP
#
# from limix_qep.tool.util import create_binomial
#
# import lim

from __future__ import division

from numpy import array, dot, empty, hstack, ones, sqrt
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from limix_math.linalg import qs_decomposition
from limix_qep import BinomialEP


def test_binomial_lml():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    (Q, S) = qs_decomposition(G)
    nsuccesses = array([1., 0., 1.])
    ntrials = array([1., 1., 1.])
    ep = BinomialEP(nsuccesses, ntrials, M, hstack(Q),
                    empty((n, 0)), hstack(S) + 1.0)
    ep.beta = array([1.])
    assert_almost_equal(ep.beta, array([1.]))
    ep.sigma2_b = 1.
    ep.sigma2_epsilon = 1e-6
    assert_almost_equal(ep.lml(), -2.34493650833)


# def test_binomial_optimize():
#
#     random = RandomState(10)
#     nsamples = 200
#     nfeatures = 1200
#     ntrials = random.randint(1, nsamples, nsamples)
#
#     G = random.randn(nsamples, nfeatures)
#     G = (G - G.mean(0)) / G.std(0)
#     G /= sqrt(200)
#
#     mean = lim.mean.OffsetMean()
#     mean.offset = 0.0
#     mean.set_data(nsamples, 'sample')
#
#     cov1 = lim.cov.LinearCov()
#     cov1.set_data((G, G), 'sample')
#
#     cov2 = lim.cov.EyeCov()
#     a = lim.util.fruits.Apples(nsamples)
#     cov2.set_data((a, a), 'sample')
#
#     cov1.scale = 1.0
#     cov2.scale = 1.0
#
#     cov = lim.cov.SumCov([cov1, cov2])
#
#     link = lim.link.LogitLink()
#     lik = lim.lik.Binomial(0, ntrials, link)
#     nsuc = lim.random.GLMMSampler(lik, mean, cov).sample(random)
#
#     (Q, S) = qs_decomposition(G)
#
#     ep = BinomialEP(nsuc, ntrials, ones((nsamples, 1)), Q[0], Q[1], S[0])
#     ep.optimize()
#     assert_almost_equal(ep.lml(), -827.22936074000131, decimal=4)
#
# if __name__ == '__main__':
#     __import__('pytest').main([__file__, '-s'])
