import numpy as np
from numpy import newaxis
from numpy import dot
import unittest

from limix_qep import Binomial
from limix_qep.ep.base import EP
from limix_qep.ep.base2 import EP2
from limix_math.linalg import economic_QS

# K = \sigma_g^2 Q (S + \delta I) Q.T
def create_binomial(nsamples, nfeatures, ntrials, sigg2=0.8, delta=0.2,
                    sige2=1., seed=None):
    if seed is not None:
        np.random.seed(seed)

    if np.isscalar(ntrials):
        ntrials = np.full(nsamples, ntrials, dtype=int)
    else:
        ntrials = np.asarray(ntrials, int)

    X = np.random.randn(nsamples, nfeatures)
    X -= np.mean(X, 0)
    X /= np.std(X, 0)
    X /= np.sqrt(nfeatures)

    u = np.random.randn(nfeatures) * np.sqrt(sigg2)

    u -= np.mean(u)
    u /= np.std(u)
    u *= np.sqrt(sigg2)

    g1 = dot(X, u)
    g2 = np.random.randn(nsamples)
    g2 -= np.mean(g2)
    g2 /= np.std(g2)
    g2 *= np.sqrt(sigg2 * delta)

    g = g1 + g2

    E = np.random.randn(nsamples, np.max(ntrials))
    E *= np.sqrt(sige2)

    Z = g[:, newaxis] + E

    Z[Z >  0.] = 1.
    Z[Z <= 0.] = 0.

    y = np.empty(nsamples)
    for i in xrange(y.shape[0]):
        y[i] = np.sum(Z[i,:ntrials[i]])

    return (y, X)

class TestEP(unittest.TestCase):
    def setUp(self):
        pass
        # np.random.seed(5)
        # n = 5
        # self._n = n
        # p = n+4
        #
        # M = np.ones((n, 1)) * 0.4
        # G = np.random.randint(3, size=(n, p))
        # G = np.asarray(G, dtype=float)
        # G -= G.mean(axis=0)
        # G /= G.std(axis=0)
        # G /= np.sqrt(p)
        #
        # K = dot(G, G.T) + np.eye(n)*0.1
        # self._K = K

    def test_bernoulli_lml(self):
        n = 3
        M = np.ones((n, 1)) * 1.
        G = np.array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
        K = dot(G, G.T) + np.eye(n)*1.0
        (Q, S) = economic_QS(K, 'K')
        y = np.array([1., 0., 1.])
        ep2 = EP2(y, M, K)
        ep = EP(y, M, Q, S)
        ep2.beta = np.array([1.])
        ep2.sigg2 = 1.
        ep.beta = np.array([1.])
        ep.sigg2 = 1.
        self.assertAlmostEqual(ep.lml(), ep2.lml())

    # def test_bernoulli_optimize(self):
    #     np.random.seed(5)
    #     nsamples = 500
    #     nfeatures = 600
    #     ntrials = 1
    #
    #     M = np.ones((nsamples, 1))
    #     G = np.random.randint(3, size=(nsamples, nfeatures))
    #     G = np.asarray(G, dtype=float)
    #     G -= G.mean(axis=0)
    #     G /= G.std(axis=0)
    #     G /= np.sqrt(nfeatures)
    #
    #     K = dot(G, G.T) + np.eye(nsamples)*0.1
    #
    #     (y, G) = create_binomial(nsamples, nfeatures, ntrials, sigg2=1.0,
    #                              delta=1e-6, seed=5)
    #
    #
    #     ep = EP2(y, M, K)
    #     ep.optimize(opt_delta=False)
    #     # self.assertAlmostEqual(ep.sigg2, 1.6795458112344945)
    #     # np.testing.assert_allclose(ep.beta, [0.13111], rtol=1e-5)
    #     # self.assertEqual(ep.delta, 0.)

    # def test_bernoulli_optimize_degenerated_covariate(self):
    #     seed = 15
    #     nsamples = 500
    #     nfeatures = 600
    #     ntrials = 1
    #
    #     M = np.ones((nsamples, 4))
    #     M[:,2] = 0.
    #
    #     (y, G) = create_binomial(nsamples, nfeatures, ntrials, sigg2=1.0,
    #                              delta=1e-6, seed=seed)
    #
    #     (Q, S) = economic_QS(G, 'G')
    #
    #     ep = EP(y, M, Q, S)
    #     ep.optimize(opt_delta=False)
    #     self.assertAlmostEqual(ep.sigg2, 1.6795458112344945)
    #     np.testing.assert_allclose(ep.beta, [0.13111]+[0.]*3, rtol=1e-5)
    #     self.assertEqual(ep.delta, 0.)

if __name__ == '__main__':
    unittest.main()
