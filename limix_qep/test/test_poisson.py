from numpy import ones
from numpy import array
from numpy import empty
from numpy import hstack
from numpy import random
from numpy.testing import assert_almost_equal

from limix_math.linalg import qs_decomposition

from limix_qep.ep import PoissonEP
from limix_qep.ep import BinomialEP
# from limix_qep.ep import PoissonEP2

from limix_qep.tool.util import create_binomial


def test_poisson_lml():

    random.seed(6)
    n = 10
    nfeatures = 10

    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])

    (Q, S) = qs_decomposition(G)

    ntrials = random.randint(1, 20, n)
    (nsuc, G) = create_binomial(n, nfeatures, ntrials, seed=10, offset=-1.)

    (Q, S) = qs_decomposition(G)

    print(nsuc)
    ep1 = BinomialEP(nsuc, ntrials, M, Q[0], Q[1], S[0])
    ep2 = PoissonEP(nsuc, M, Q[0], Q[1], S[0])
    # ep3 = PoissonEP2(nsuc, M, Q[0], Q[1], S[0])

    print(ep1.lml())
    print(ep2.lml())
#     # ep1.beta = array([1.])
#     # ep1.genetic_variance = 1.
#     # ep1.environmental_variance = 1e-7
#     # lml1 = ep1.lml()
#     # print(lml1)
#     #
#     # ep2 = BernoulliEP(y, M, hstack(Q), empty((n, 0)), hstack(S) + 1.0)
#     # ep2.beta = array([1.])
#     # ep2.genetic_variance = 1.
#     # lml2 = ep2.lml()
#     #
#     # assert_almost_equal(lml1 - lml2, 0., decimal=5)
