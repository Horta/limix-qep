import numpy as np
from numpy import dot
from scipy.linalg.lapack import get_lapack_funcs
from limix_math.linalg import cho_solve, sum2diag
from limix_math.linalg import ddot, dotd

class SiteLik(object):
    def __init__(self, nsamples):
        self.tau = np.zeros(nsamples)
        self.eta = np.zeros(nsamples)

    def initialize(self, ok):
        self.tau[ok] = 0.
        self.eta[ok] = 0.

    def update(self, ctau, ceta, hmu, hsig2):
        zf = 1e-16
        self.tau[:] = np.maximum(1.0 / hsig2 - ctau, zf)
        self.eta[:] = hmu / hsig2 - ceta

    @property
    def mu(self):
        return self.eta/self.tau

    @property
    def sig2(self):
        return 1./self.tau

    def copy(self):
        that = SiteLik(len(self.tau))
        that.tau[:] = self.tau
        that.eta[:] = self.eta
        return that


class Joint(object):
    def __init__(self, Q, S):
        self._Q = Q
        self._S = S
        nsamples = Q.shape[0]
        self.tau = np.empty(nsamples)
        self.eta = np.empty(nsamples)

    def initialize(self, m, sigg2, delta):
        Q = self._Q
        S = self._S
        v = sigg2 * dotd(ddot(Q, S, left=False), Q.T) + sigg2 * delta
        self.tau[:] = 1.0 / v
        self.eta[:] = self.tau * m

    def update(self, m, sigg2, delta, S, Q, L1, ttau, teta, A1, A0T):

        SQt = ddot(S, Q.T, left=True)
        L1_Qt = cho_solve(L1, Q.T)
        L1_QtA1 = ddot(L1_Qt, A1, left=False)

        C = ttau * A0T
        D = 1. - C
        DQ = ddot(D, Q, left=True)
        p1 = sigg2 * dotd(DQ, SQt)
        if delta != 0:
            p1 += D * sigg2 * delta

        # DQ_L1t = ddot(D, L1_Qt.T, left=True)
        Z = dot(ddot(dot(L1_QtA1, Q), S, left=False), Q.T)

        p2 = - sigg2 * dotd(DQ, Z)
        if delta != 0:
            p2 -= sigg2 * delta * dotd(DQ, L1_QtA1)

        r = p1 + p2
        self.tau[:] = 1./r

        mu = m - C * m - D * dot(Q, dot(L1_QtA1, m))
        u = sigg2 * dot(Q, S * dot(Q.T, teta))
        if delta != 0:
            u += sigg2 * delta * teta

        u = u - C * u
        mu += u - sigg2 * dot(DQ, dot(Z, teta))
        if delta != 0:
            mu -= sigg2 * delta * dot(DQ, dot(L1_QtA1, teta))

        self.eta[:] = self.tau * mu

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

class Joint2(object):
    def __init__(self, nsamples):
        self.tau = np.empty(nsamples)
        self.eta = np.empty(nsamples)

    def initialize(self, m, K):
        self.tau[:] = 1.0 / K.diagonal()
        self.eta[:] = self.tau * m

    def update(self, m, K, L, ttau, teta):

        potri = get_lapack_funcs('potri', (L,))
        TK_1 = potri(L.T)[0]
        TK_1 = symmetrize(TK_1)
        r = ttau * dotd(TK_1, K)

        self.tau[:] = 1./r

        mu = ttau * (cho_solve(L, dot(K, m)) + cho_solve(L, dot(K, teta)))

        self.eta[:] = self.tau * mu

class Cavity(object):
    def __init__(self, nsamples):
        self.tau = np.empty(nsamples)
        self.eta = np.empty(nsamples)

    def update(self, jtau, jeta, ttau, teta):
        self.tau[:] = jtau - ttau
        self.eta[:] = jeta - teta

    @property
    def mu(self):
        return self.eta / self.tau

    @property
    def sig2(self):
        return 1. / self.tau
