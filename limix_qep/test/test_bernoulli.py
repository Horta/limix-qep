
from numpy import dot
from numpy import ones
from numpy import eye
from numpy import array
from numpy.testing import assert_almost_equal

from limix_qep.ep import BernoulliEP

from limix_math.linalg import _QS_from_K
from limix_math.linalg import economic_QS

from .util import create_binomial

def test_bernoulli_lml():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    K = dot(G, G.T) + eye(n)*1.0
    (Q, S) = _QS_from_K(K)
    y = array([1., 0., 1.])
    ep = BernoulliEP(y, M, Q, S)
    ep.beta = array([1.])
    ep.sigg2 = 1.
    assert_almost_equal(ep.lml(), -2.59563598457)

def test_bernoulli_optimize():
    seed = 15
    nsamples = 500
    nfeatures = 600
    ntrials = 1

    M = ones((nsamples, 1))

    (y, G) = create_binomial(nsamples, nfeatures, ntrials, sigg2=1.0,
                             delta=1e-6, seed=seed)

    (Q, S) = economic_QS(G, 'G')

    ep = BernoulliEP(y, M, Q, S)
    ep.optimize()
    assert_almost_equal(ep.sigg2, 1.6795435940073431, decimal=5)
