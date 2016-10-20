from __future__ import division

from numpy import array, dot, empty, hstack, ones, pi, sqrt, zeros, exp
from numpy import mean
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from limix_math.linalg import qs_decomposition
from limix_qep import PoissonEP

random = RandomState(2)
nsamples = 1000
nfeatures = 2000

G = random.randn(nsamples, nfeatures) / sqrt(nfeatures)

u = random.randn(nfeatures)

z = 0.1 + 2 * dot(G, u) + random.randn(nsamples)

y = zeros(nsamples)
for i in range(nsamples):
    y[i] = random.poisson(lam=exp(z[i]))
(Q, S) = qs_decomposition(G)

M = ones((nsamples, 1))
ep = PoissonEP(y, M, Q[0], Q[1], S[0])
ep.optimize()
print(ep.beta[0])
print(ep.heritability)
