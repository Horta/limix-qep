from numpy import ones
from numpy import array
from numpy import empty
from numpy import hstack
from numpy import random
from numpy.testing import assert_almost_equal

from limix_math.linalg import qs_decomposition

from limix_qep.ep import BinomialEP
from limix_qep.ep import BernoulliEP

from .util import create_binomial

# def test_binomial_lml():
#
#     random.seed(6)
#     n = 3
#     M = ones((n, 1)) * 1.
#     G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
#
#     (Q, S) = qs_decomposition(G)
#     y = array([1., 0., 1.])
#     ep1 = BinomialEP(y, 1, M, hstack(Q), empty((n,0)), hstack(S) + 1.0)
#     ep1.beta = array([1.])
#     ep1.var = 1.
#     ep1.e = 1e-7
#     lml1 = ep1.lml()
#
#     ep2 = BernoulliEP(y, M, hstack(Q), empty((n,0)), hstack(S) + 1.0)
#     ep2.beta = array([1.])
#     ep2.var = 1.
#     lml2 = ep2.lml()
#
#     assert_almost_equal(lml1 - lml2, 0., decimal=5)

def test_binomial_optimize():

    seed = 10
    nsamples = 400
    nfeatures = 600
    # ntrials = 2
    ntrials = 300

    M = ones((nsamples, 1))

    (y, G) = create_binomial(nsamples, nfeatures, ntrials, var=1.0,
                             delta=0.1, sige2=0.1, seed=seed)

    (Q, S) = qs_decomposition(G)

    ep = BinomialEP(y, ntrials, M, Q[0], Q[1], S[0])
    print("Previous lml: %.5f" % ep.lml())
    ep.optimize()
    print("Genetic variance: %.5f" % ep.genetic_variance)
    print("Environmental variance: %.5f" % ep.environmental_variance)
    print("Instrumental variance: %.5f" % ep.instrumental_variance)
    print("Covariates variance: %.5f" % ep.covariates_variance)
    print("Heritability: %.5f" % ep.heritability)
    print("Beta: %s" % str(ep.beta))
    print("After lml: %.10f" % ep.lml())

# def test_binomial_optimize2():
#
#     seed = 10
#     nsamples = 200
#     nfeatures = 600
#     ntrials = 2
#
#     M = ones((nsamples, 1))
#
#     (y, G) = create_binomial(nsamples, nfeatures, ntrials, var=1.0,
#                              delta=0.1, sige2=0.1, seed=seed)
#
#     (Q, S) = qs_decomposition(G)
#
#     ep = BinomialEP(y, ntrials, M, Q[0], Q[1], S[0])
#     print("Previous lml: %.5f" % ep.lml())
#     ep.genetic_variance = 1.15389
#     ep.environmental_variance = 0.03939
#     # ep.covariates_variance
#     ep.beta = array([0.0158122])
#     print("After lml: %.10f" % ep.lml())
#
#     print("Genetic variance: %.5f" % ep.genetic_variance)
#     print("Environmental variance: %.5f" % ep.environmental_variance)
#     print("Instrumental variance: %.5f" % ep.instrumental_variance)
#     print("Covariates variance: %.5f" % ep.covariates_variance)
#     print("Beta: %s" % str(ep.beta))

# def test_binomial_optimize():
#
#     seed = 10
#     nsamples = 30
#     nfeatures = 600
#     ntrials = 1
#
#     M = ones((nsamples, 1))
#
#     (y, G) = create_binomial(nsamples, nfeatures, ntrials, var=1.0,
#                              delta=0.1, sige2=0.1, seed=seed)
#
#     (Q, S) = qs_decomposition(G)
#
#     ep = BinomialEP(y, 1, M, Q[0], Q[1], S[0])
#     # print("")
#     # print("Previous lml: %.5f" % ep.lml())
#     import ipdb; ipdb.set_trace()
#     ep.optimize()
#     print("Genetic variance: %.5f" % ep.genetic_variance)
#     print("Environmental variance: %.5f" % ep.environmental_variance)
#     print("Instrumental variance: %.5f" % ep.instrumental_variance)
#     print("Covariates variance: %.5f" % ep.covariates_variance)
#     print("Beta: %s" % str(ep.beta))
#     # print(ep.lml())
#     print("After lml: %.10f" % ep.lml())
#     # assert_almost_equal(ep.lml(), -4.9925026426, decimal=5)
#     #
#     # ep.genetic_variance = 0.04103
#     # ep.environmental_variance = 0.00140
#     # ep.beta = array([0.17166795])
#     # print(ep.lml())

# def test_binomial_optimize2():
#
#     seed = 10
#     nsamples = 30
#     nfeatures = 600
#     ntrials = 1
#
#     M = ones((nsamples, 1))
#
#     (y, G) = create_binomial(nsamples, nfeatures, ntrials, var=1.0,
#                              delta=0.1, sige2=0.1, seed=seed)
#
#     (Q, S) = qs_decomposition(G)
#
#     ep = BinomialEP(y, 1, M, Q[0], Q[1], S[0])
#     ep.genetic_variance = 2791.82709
#     ep.environmental_variance = 95.29156
#     ep.beta = array([9.18539605])
#
#     print("Genetic variance: %.5f" % ep.genetic_variance)
#     print("Environmental variance: %.5f" % ep.environmental_variance)
#     print("Instrumental variance: %.5f" % ep.instrumental_variance)
#     print("Covariates variance: %.5f" % ep.covariates_variance)
#     print("Beta: %s" % str(ep.beta))
#
#     print(ep.lml())




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
