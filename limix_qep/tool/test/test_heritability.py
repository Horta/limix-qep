from numpy.testing import assert_almost_equal
from numpy.random import RandomState
from numpy import asarray
from numpy import ones
from numpy import sqrt
from numpy import eye
from numpy import zeros_like
from numpy import dot
import lim

from limix_qep.tool.heritability import estimate
from limix_qep.lik import Binomial
from limix_qep.lik import Poisson


def test_h2_bernoulli():
    random = RandomState(981)
    n = 500
    p = n + 4

    M = ones((n, 1)) * 0.4
    G = random.randint(3, size=(n, p))
    G = asarray(G, dtype=float)
    G -= G.mean(axis=0)
    G /= G.std(axis=0)
    G /= sqrt(p)

    K = dot(G, G.T)
    Kg = K / K.diagonal().mean()
    K = 0.5 * Kg + 0.5 * eye(n)
    K = K / K.diagonal().mean()

    z = random.multivariate_normal(M.ravel(), K)
    y = zeros_like(z)
    y[z > 0] = 1.

    h2 = estimate(y, K=Kg, covariate=M)[0]
    assert_almost_equal(h2, 0.403163261934, decimal=5)


def test_h2_binomial():
    random = RandomState(981)
    ntrials = 1
    n = 500
    p = n + 4

    M = ones((n, 1)) * 0.4
    G = random.randint(3, size=(n, p))
    G = asarray(G, dtype=float)
    G -= G.mean(axis=0)
    G /= G.std(axis=0)
    G /= sqrt(p)

    K = dot(G, G.T)
    Kg = K / K.diagonal().mean()
    K = 0.5 * Kg + 0.5 * eye(n)
    K = K / K.diagonal().mean()

    z = random.multivariate_normal(M.ravel(), K)
    y = zeros_like(z)
    y[z > 0] = 1.

    outcome = Binomial(ntrials, n)
    import logging
    logging.basicConfig(level=logging.DEBUG)
    h2 = estimate(y, K=Kg, covariate=M, outcome_type=outcome)[0]
    assert_almost_equal(h2, 0.2414058745955906, decimal=5)


def test_h2_poisson():
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

    outcome = Poisson()
    import logging
    logging.basicConfig(level=logging.DEBUG)

    h2 = estimate(y, G=G, outcome_type=outcome)[0]
    assert_almost_equal(h2, 0.99981071847359615, decimal=5)

#
#     # def test_h2_binomial_fast(self):
#     #     random = np.random.RandomState(981)
#     #     ntrials = 1
#     #     n = 50
#     #     p = n+4
#     #
#     #     M = np.ones((n, 1)) * 0.4
#     #     G = random.randint(3, size=(n, p))
#     #     G = np.asarray(G, dtype=float)
#     #     G -= G.mean(axis=0)
#     #     G /= G.std(axis=0)
#     #     G /= np.sqrt(p)
#     #
#     #     K = np.dot(G, G.T)
#     #     Kg = K / K.diagonal().mean()
#     #     K = 0.5*Kg + 0.5*np.eye(n)
#     #     K = K / K.diagonal().mean()
#     #
#     #     z = random.multivariate_normal(M.ravel(), K)
#     #     y = np.zeros_like(z)
#     #     y[z>0] = 1.
#     #
#     #     outcome = Binomial(ntrials, n)
#     #     # import logging
#     #     # logging.basicConfig(level=logging.DEBUG)
#     #     h2 = estimate(y, K=Kg, covariate=M, outcome_type=outcome)[0]
#     #     print(h2)
#     #     self.assertAlmostEqual(h2, 0.403163261934)
#
#
# if __name__ == '__main__':
#     unittest.main()
