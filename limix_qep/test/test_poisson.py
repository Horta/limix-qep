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
# from limix_qep.ep import PoissonEP
#
# import lim
#
#
# def test_poisson_optimize():
#
#     random = RandomState(10)
#     nsamples = 200
#     nfeatures = 1200
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
#     link = lim.link.LogLink()
#     lik = lim.lik.Poisson(0, link)
#     y = lim.random.GLMMSampler(lik, mean, cov).sample(random)
#
#     (Q, S) = qs_decomposition(G)
#
#     ep = PoissonEP(y, ones((nsamples, 1)), Q[0], Q[1], S[0])
#     ep.optimize()
#     assert_almost_equal(ep.lml(), -530.8988815711748, decimal=4)
#
#
# def test_poisson_optimize2():
#
#     random = RandomState(10)
#     nsamples = 200
#     nfeatures = 1200
#
#     G = random.randn(nsamples, nfeatures)
#     G = (G - G.mean(0)) / G.std(0)
#     G /= sqrt(1200)
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
#     cov1.scale = 10.0
#     cov2.scale = 1.0
#
#     cov = lim.cov.SumCov([cov1, cov2])
#
#     link = lim.link.LogLink()
#     lik = lim.lik.Poisson(0, link)
#     y = lim.random.GLMMSampler(lik, mean, cov).sample(random)
#
#     (Q, S) = qs_decomposition(G)
#
#     ep = PoissonEP(y, ones((nsamples, 1)), Q[0], Q[1], S[0])
#     ep.optimize()
#     print(ep.genetic_variance)
#     print(ep.environmental_variance)
#     print(ep.beta)
#
# if __name__ == '__main__':
#     test_poisson_optimize2()
#     # __import__('pytest').main([__file__, '-s'])
