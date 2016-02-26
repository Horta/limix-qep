import numpy as np
from numpy import newaxis
from numpy import dot
import unittest

from limix_qep import Binomial
from limix_qep.ep import EP
from limix_math.linalg import economic_QS
from limix_math.linalg import _QS_from_K

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
        np.random.seed(5)
        n = 5
        self._n = n
        p = n+4

        M = np.ones((n, 1)) * 0.4
        G = np.random.randint(3, size=(n, p))
        G = np.asarray(G, dtype=float)
        G -= G.mean(axis=0)
        G /= G.std(axis=0)
        G /= np.sqrt(p)

        K = dot(G, G.T) + np.eye(n)*0.1
        (S, Q) = np.linalg.eigh(K)
        self._S = S
        self._Q = Q
        self._M = M

    def test_binomial_lml(self):

        np.random.seed(6)
        n = 3
        M = np.ones((n, 1)) * 1.
        G = np.array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
        K = dot(G, G.T) + np.eye(n)*1.0
        (Q, S) = _QS_from_K(K)
        y = np.array([1., 0., 1.])
        ep = EP(y, M, Q, S)
        ep.beta = np.array([1.])
        ep.sigg2 = 1.
        lml1 = ep.lml()

        np.random.seed(6)
        n = 3
        M = np.ones((n, 1)) * 1.
        G = np.array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
        K = dot(G, G.T) + np.eye(n)*1.0
        (Q, S) = _QS_from_K(K)
        y = np.array([1., 0., 1.])
        ep = EP(y, M, Q, S, outcome_type=Binomial(1, n))
        ep.beta = np.array([1.])
        ep.sigg2 = 1.
        ep.delta = 1e-7
        self.assertAlmostEqual(lml1 - ep.lml(), 0., places=5)

    def test_binomial_optimize(self):

        seed = 10
        nsamples = 30
        nfeatures = 600
        ntrials = 1

        M = np.ones((nsamples, 1))

        (y, G) = create_binomial(nsamples, nfeatures, ntrials, sigg2=1.,
                                 delta=0.1, sige2=0.1, seed=seed)

        (Q, S) = economic_QS(G, 'G')

        ep = EP(y, M, Q, S, outcome_type=Binomial(1, nsamples))
        ep.optimize(disp=True)

        self.assertAlmostEqual(ep.lml(), -19.2257156813, places=5)

    def test_bernoulli_lml(self):
        n = 3
        M = np.ones((n, 1)) * 1.
        G = np.array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
        K = dot(G, G.T) + np.eye(n)*1.0
        (Q, S) = _QS_from_K(K)
        y = np.array([1., 0., 1.])
        ep = EP(y, M, Q, S)
        ep.beta = np.array([1.])
        ep.sigg2 = 1.
        self.assertAlmostEqual(ep.lml(), -2.59563598457)


    def test_bernoulli_optimize(self):
        seed = 15
        nsamples = 500
        nfeatures = 600
        ntrials = 1

        M = np.ones((nsamples, 1))

        (y, G) = create_binomial(nsamples, nfeatures, ntrials, sigg2=1.0,
                                 delta=1e-6, seed=seed)

        (Q, S) = economic_QS(G, 'G')

        ep = EP(y, M, Q, S)
        ep.optimize(opt_delta=False)
        self.assertAlmostEqual(ep.sigg2, 1.6795458112344945)
        np.testing.assert_allclose(ep.beta, [0.13111], rtol=1e-5)
        self.assertEqual(ep.delta, 0.)

    def test_bernoulli_optimize_degenerated_covariate(self):
        seed = 15
        nsamples = 500
        nfeatures = 600
        ntrials = 1

        M = np.ones((nsamples, 4))
        M[:,2] = 0.

        (y, G) = create_binomial(nsamples, nfeatures, ntrials, sigg2=1.0,
                                 delta=1e-6, seed=seed)

        (Q, S) = economic_QS(G, 'G')

        ep = EP(y, M, Q, S)
        ep.optimize(opt_delta=False)
        self.assertAlmostEqual(ep.sigg2, 1.6795458112344945)
        np.testing.assert_allclose(ep.beta, [0.13111]+[0.]*3, rtol=1e-5)
        self.assertEqual(ep.delta, 0.)

if __name__ == '__main__':
    unittest.main()
