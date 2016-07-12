import logging
import numpy as np
from numpy import dot
from limix_math.linalg import cho_solve
from limix_math.linalg import ddot, dotd
from time import time

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
        self._logger = logging.getLogger(__name__)
        self._Q = Q
        self._S = S
        nsamples = Q.shape[0]
        self.tau = np.empty(nsamples)
        self.eta = np.empty(nsamples)

        self._time_elapsed = dict(update=0, initialize=0, cho_solve=0, mult=0)
        self._calls = dict(update=0, initialize=0, cho_solve=0, mult=0)

    def initialize(self, m, sigg2, delta):
        before = time()
        Q = self._Q
        S = self._S
        v = sigg2 * dotd(ddot(Q, S, left=False), Q.T) + sigg2 * delta
        self.tau[:] = 1.0 / v
        self.eta[:] = self.tau * m
        self._time_elapsed['initialize'] += time() - before
        self._calls['initialize'] += 1

    def update(self, m, sigg2, delta, S, Q, L1, ttau, teta, A1, A0T):
        self._logger.debug('joint update has started')
        before_ = time()
        SQt = ddot(S, Q.T, left=True)
        before = time()
        L1_Qt = cho_solve(L1, Q.T) # !!!CUBIC!!!
        self._time_elapsed['cho_solve'] += time() - before
        self._calls['cho_solve'] += 1
        L1_QtA1 = ddot(L1_Qt, A1, left=False)

        C = ttau * A0T
        D = 1. - C
        DQ = ddot(D, Q, left=True)
        p1 = sigg2 * dotd(DQ, SQt)
        if delta != 0:
            p1 += D * sigg2 * delta

        # DQ_L1t = ddot(D, L1_Qt.T, left=True)
        before = time()
        Z = dot(ddot(dot(L1_QtA1, Q), S, left=False), Q.T) # !!!CUBIC!!!
        self._time_elapsed['mult'] += time() - before
        self._calls['mult'] += 1

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
        self._time_elapsed['update'] += time() - before_
        self._calls['update'] += 1
        self._logger.debug('joint update has finished')

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
