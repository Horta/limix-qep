
from numpy import dot
from numpy import ones
from numpy import array
from numpy.testing import assert_almost_equal
import numpy as np

from limix_qep.ep import BernoulliEP

from limix_math.linalg import qs_decomposition

from .util import create_binomial

def test_bernoulli_lml():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    (Q, S) = qs_decomposition(G)
    y = array([1., 0., 1.])
    ep = BernoulliEP(y, M, np.hstack(Q), np.empty((n,0)), np.hstack(S) + 1.0)
    ep.beta = array([1.])
    ep.var = 1.
    assert_almost_equal(ep.lml(), -2.59563598457)

def test_bernoulli_optimize():
    seed = 15
    nsamples = 500
    nfeatures = 600
    ntrials = 1

    M = ones((nsamples, 1))

    (y, G) = create_binomial(nsamples, nfeatures, ntrials, var=1.0,
                             delta=1e-6, seed=seed)

    (Q, S) = qs_decomposition(G)

    ep = BernoulliEP(y, M, Q[0], Q[1], S[0])
    ep.optimize()
    print(ep.beta)
    # assert_almost_equal(ep.var, 1.6795435940073431, decimal=5)
#
# def test_bernoulli_prediction():
#     seed = 15
#     nsamples = 500
#     nfeatures = 600
#     ntrials = 1
#
#     M = ones((nsamples, 1))
#
#     (y, G) = create_binomial(nsamples, nfeatures, ntrials, var=1.0,
#                              delta=1e-6, seed=seed)
#
#     (Q, S) = qs_decomposition(G)
#
#     ep = BernoulliEP(y, M, Q[0], Q[1], S[0])
#     ep.optimize()
#
#     prob_y = []
#     for i in range(4):
#         g = G[i,:]
#         var = dot(g, g)
#         covar = dot(g, G.T)
#         p = ep.predict(M[i,:], var, covar)
#         prob_y.append(p.pdf(y[i])[0])
#
#     prob_yi = [0.48705911290518589, 0.40605290158743768,
#                0.84365032664655915, 0.83794141874476269]
#
#     assert_almost_equal(prob_y[:4], prob_yi, decimal=6)
