from numpy import ones
from numpy import array
from numpy import empty
from numpy import hstack
from numpy import random
from numpy.testing import assert_almost_equal

from limix_math.linalg import qs_decomposition

from limix_qep.ep import BinomialEP
from limix_qep.ep import BernoulliEP

from limix_qep.tool.util import create_binomial


def test_binomial_optimize():

    seed = 10
    nsamples = 200
    nfeatures = 1200
    ntrials = random.RandomState(seed).randint(1, 1000, nsamples)

    M = ones((nsamples, 1))

    (y, G) = create_binomial(nsamples, nfeatures, ntrials, var=1.0,
                             delta=0.1, sige2=0.1, seed=seed)

    (Q, S) = qs_decomposition(G)

    ep = BinomialEP(y, ntrials, M, Q[0], Q[1], S[0])
    ep.optimize()
    assert_almost_equal(ep.lml(), -879.08491430506729, decimal=3)
