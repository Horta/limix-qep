from numpy import ones
from numpy import array
from numpy import empty
from numpy import hstack
from numpy import random
from numpy import linspace
from numpy.testing import assert_almost_equal

from limix_math.linalg import qs_decomposition

from limix_qep.ep import BinomialEP
from limix_qep.ep import BernoulliEP

from .util import create_binomial

def test_binomial_lml():

    random.seed(6)
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])

    (Q, S) = qs_decomposition(G)
    y = array([1., 0., 1.])
    ep1 = BinomialEP(y, 1, M, hstack(Q), empty((n,0)), hstack(S) + 1.0)
    ep1.beta = array([1.])
    ep1.genetic_variance = 1.
    ep1.environmental_variance = 1e-7
    lml1 = ep1.lml()

    ep2 = BernoulliEP(y, M, hstack(Q), empty((n,0)), hstack(S) + 1.0)
    ep2.beta = array([1.])
    ep2.genetic_variance = 1.
    lml2 = ep2.lml()

    assert_almost_equal(lml1 - lml2, 0., decimal=5)

def test_binomial_optimize():

    seed = 10
    nsamples = 400
    nfeatures = 600
    ntrials = 300

    M = ones((nsamples, 1))

    (y, G) = create_binomial(nsamples, nfeatures, ntrials, var=1.0,
                             delta=0.1, sige2=0.1, seed=seed)

    (Q, S) = qs_decomposition(G)

    ep = BinomialEP(y, ntrials, M, Q[0], Q[1], S[0])
    ep.optimize()
    assert_almost_equal(ep.lml(), -1602.7098976758323, decimal=5)

# def test_binomial_optimize():
#
#     seed = 10
#     nsamples = 400
#     nfeatures = 600
#     # ntrials = 2
#     ntrials = 300
#
#     M = ones((nsamples, 1))
#
#     (y, G) = create_binomial(nsamples, nfeatures, ntrials, var=1.0,
#                              delta=0.1, sige2=0.1, seed=seed)
#
#     (Q, S) = qs_decomposition(G)
#
#     ep = BinomialEP(y, ntrials, M, Q[0], Q[1], S[0])
#     # print("Previous lml: %.5f" % ep.lml())
#     ep.beta = array([-0.02100733])
#
#     h2s = linspace(0.01, 1-0.01, 10)
#     rs = linspace(0.01, 1-0.01, 10)
#
#     for h2 in h2s:
#         ep.heritability = h2
#         for r in rs:
#             ep.noise_ratio = r
#     # print("Genetic variance: %.5f" % ep.genetic_variance)
#     # print("Environmental variance: %.5f" % ep.environmental_variance)
#     # print("Instrumental variance: %.5f" % ep.instrumental_variance)
#     # print("Covariates variance: %.5f" % ep.covariates_variance)
#     # print("Heritability: %.5f" % ep.heritability)
#     # print("Beta: %s" % str(ep.beta))
#     # print("After lml: %.10f" % ep.lml())
