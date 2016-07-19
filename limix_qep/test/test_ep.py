#
# from numpy import dot
# from numpy import sqrt
# from numpy import full
# from numpy import ones
# from numpy import eye
# from numpy import array
# from numpy import asarray
# from numpy import isscalar
# from numpy import empty
# from numpy import random
# from numpy import newaxis
# from numpy.testing import assert_almost_equal
#
# from limix_qep import Binomial
# from limix_qep.ep import BernoulliEP
#
# from limix_math.linalg import economic_QS
# from limix_math.linalg import _QS_from_K
#
# from scipy.misc import logsumexp
#
#
# # K = \sigma_g^2 Q (S + \delta I) Q.T
# def create_binomial(nsamples, nfeatures, ntrials, sigg2=0.8, delta=0.2,
#                     sige2=1., seed=None):
#     if seed is not None:
#         random.seed(seed)
#
#     if isscalar(ntrials):
#         ntrials = full(nsamples, ntrials, dtype=int)
#     else:
#         ntrials = asarray(ntrials, int)
#
#     X = random.randn(nsamples, nfeatures)
#     X -= X.mean(0)
#     X /= X.std(0)
#     X /= sqrt(nfeatures)
#
#     u = random.randn(nfeatures) * sqrt(sigg2)
#
#     u -= u.mean()
#     u /= u.std()
#     u *= sqrt(sigg2)
#
#     g1 = dot(X, u)
#     g1 -= g1.mean()
#     g1 /= g1.std()
#     g1 *= sqrt(sigg2)
#     g2 = random.randn(nsamples)
#     g2 -= g2.mean()
#     g2 /= g2.std()
#     g2 *= sqrt(sigg2 * delta)
#
#     g = g1 + g2
#
#     E = random.randn(nsamples, max(ntrials))
#     E *= sqrt(sige2)
#
#     Z = g[:, newaxis] + E
#
#     Z[Z >  0.] = 1.
#     Z[Z <= 0.] = 0.
#
#     y = empty(nsamples)
#     for i in range(y.shape[0]):
#         y[i] = sum(Z[i,:ntrials[i]])
#
#     return (y, X)
#
# # class TestEP(unittest.TestCase):
# #     def setUp(self):
# #         np.random.seed(5)
# #         n = 5
# #         self._n = n
# #         p = n+4
# #
# #         M = np.ones((n, 1)) * 0.4
# #         G = np.random.randint(3, size=(n, p))
# #         G = np.asarray(G, dtype=float)
# #         G -= G.mean(axis=0)
# #         G /= G.std(axis=0)
# #         G /= np.sqrt(p)
# #
# #         K = dot(G, G.T) + np.eye(n)*0.1
# #         (S, Q) = np.linalg.eigh(K)
# #         self._S = S
# #         self._Q = Q
# #         self._M = M
#
#     # def test_binomial_lml(self):
#     #
#     #     np.random.seed(6)
#     #     n = 3
#     #     M = np.ones((n, 1)) * 1.
#     #     G = np.array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
#     #     K = dot(G, G.T) + np.eye(n)*1.0
#     #     (Q, S) = _QS_from_K(K)
#     #     y = np.array([1., 0., 1.])
#     #     ep = EP(y, M, Q, S)
#     #     ep.beta = np.array([1.])
#     #     ep.sigg2 = 1.
#     #     lml1 = ep.lml()
#     #
#     #     np.random.seed(6)
#     #     n = 3
#     #     M = np.ones((n, 1)) * 1.
#     #     G = np.array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
#     #     K = dot(G, G.T) + np.eye(n)*1.0
#     #     (Q, S) = _QS_from_K(K)
#     #     y = np.array([1., 0., 1.])
#     #     ep = EP(y, M, Q, S, outcome_type=Binomial(1, n))
#     #     ep.beta = np.array([1.])
#     #     ep.sigg2 = 1.
#     #     ep.delta = 1e-7
#     #     self.assertAlmostEqual(lml1 - ep.lml(), 0., places=5)
#     #
#     # def test_binomial_optimize(self):
#     #
#     #     seed = 10
#     #     nsamples = 30
#     #     nfeatures = 600
#     #     ntrials = 1
#     #
#     #     M = np.ones((nsamples, 1))
#     #
#     #     (y, G) = create_binomial(nsamples, nfeatures, ntrials, sigg2=1.,
#     #                              delta=0.1, sige2=0.1, seed=seed)
#     #
#     #     (Q, S) = economic_QS(G, 'G')
#     #
#     #     ep = EP(y, M, Q, S, outcome_type=Binomial(1, nsamples))
#     #     ep.optimize(disp=True)
#     #
#     #     self.assertAlmostEqual(ep.lml(), -19.649207220129359, places=5)
#
#     # def test_bernoulli_optimize(self):
#     #     seed = 15
#     #     nsamples = 500
#     #     nfeatures = 600
#     #     ntrials = 1
#     #
#     #     M = np.ones((nsamples, 1))
#     #
#     #     (y, G) = create_binomial(nsamples, nfeatures, ntrials, sigg2=1.0,
#     #                              delta=1e-6, seed=seed)
#     #
#     #     (Q, S) = economic_QS(G, 'G')
#     #
#     #     ep = EP(y, M, Q, S)
#     #     ep.optimize(opt_delta=False)
#     #     # import pylab as plt
#     #     # sigg2s = np.linspace(1.679, 1.680, 1000)
#     #     # lmls = []
#     #     # for sigg2 in sigg2s:
#     #     #     ep.sigg2 = sigg2
#     #     #     lmls.append(ep.lml())
#     #     #
#     #     # print("Best sigg2: %.10f" % sigg2s[np.argmax(lmls)])
#     #     # plt.plot(sigg2s, lmls)
#     #     # plt.show()
#     #
#     #     self.assertAlmostEqual(ep.sigg2, 1.6795435940073431, places=5)
#     #     # # np.testing.assert_allclose(ep.beta, [0.13111], rtol=1e-5)
#     #     # # self.assertEqual(ep.delta, 0.)
#     #
#     # def test_bernoulli_prediction(self):
#     #     seed = 15
#     #     nsamples = 500
#     #     nfeatures = 600
#     #     ntrials = 1
#     #
#     #     M = np.ones((nsamples, 1))
#     #
#     #     (y, G) = create_binomial(nsamples, nfeatures, ntrials, sigg2=1.0,
#     #                              delta=1e-6, seed=seed)
#     #
#     #     (Q, S) = economic_QS(G, 'G')
#     #
#     #     ep = EP(y, M, Q, S)
#     #     ep.optimize(opt_delta=False)
#     #
#     #     prob_y = []
#     #     for i in range(4):
#     #         g = G[i,:]
#     #         var = dot(g, g)
#     #         covar = dot(g, G.T)
#     #         p = ep.predict(M[i,:], var, covar)
#     #         prob_y.append(p.pdf(y[i])[0])
#     #
#     #     prob_yi = [0.48705911290518589, 0.40605290158743768,
#     #                0.84365032664655915, 0.83794141874476269]
#     #
#     #     np.testing.assert_almost_equal(prob_y[:4], prob_yi, decimal=6)
#     # #
#     # # def test_binomial_prediction(self):
#     # #     seed = 1
#     # #     nsamples = 20
#     # #     nfeatures = 50
#     # #     ntrials = 100
#     # #
#     # #     M = np.ones((nsamples, 1))
#     # #
#     # #     (y, G) = create_binomial(nsamples, nfeatures, ntrials, sigg2=1.0,
#     # #                              delta=0.5, seed=seed)
#     # #
#     # #     (Q, S) = economic_QS(G, 'G')
#     # #
#     # #     ep = EP(y, M, Q, S, Binomial(ntrials, nsamples))
#     # #     ep.optimize()
#     # #
#     # #     logpdfs = []
#     # #     for i in range(4):
#     # #         g = G[i,:]
#     # #         var = dot(g, g)
#     # #         covar = dot(g, G.T)
#     # #
#     # #         p = ep.predict(ep.beta * M[i,:], ep.sigg2 * var + ep.sigg2 * ep.delta, ep.sigg2 * covar)
#     # #         logpdfs.append(p.logpdf(y[i], ntrials)[0])
#     # #
#     # #     assert_almost_equal(logpdfs, [-2.6465067277259564,
#     # #                                   -2.1481110362156208,
#     # #                                   -1.2882192451708478,
#     # #                                   -2.9608451283086055])
#     #
#     # def test_bernoulli_optimize_degenerated_covariate(self):
#     #     seed = 15
#     #     nsamples = 500
#     #     nfeatures = 600
#     #     ntrials = 1
#     #
#     #     M = np.ones((nsamples, 4))
#     #     M[:,2] = 0.
#     #
#     #     (y, G) = create_binomial(nsamples, nfeatures, ntrials, sigg2=1.0,
#     #                              delta=1e-6, seed=seed)
#     #
#     #     (Q, S) = economic_QS(G, 'G')
#     #
#     #     ep = EP(y, M, Q, S)
#     #     ep.optimize(opt_delta=False)
#     #     self.assertAlmostEqual(ep.sigg2, 1.6795458112344945, places=4)
#     #     np.testing.assert_allclose(ep.beta, [0.13111]+[0.]*3, rtol=1e-5)
#     #     self.assertEqual(ep.delta, 0.)
