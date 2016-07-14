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

        self._time_elapsed = dict(update=0, initialize=0, cho_solve=0, mult=0, ddot1=0, ddot2=0, ddot3=0, dotvec=0)
        self._calls = dict(update=0, initialize=0, cho_solve=0, mult=0, ddot1=0, ddot2=0, ddot3=0, dotvec=0)

    def initialize(self, m, sigg2, delta):
        before = time()
        Q = self._Q
        S = self._S
        v = sigg2 * dotd(ddot(Q, S, left=False), Q.T) + sigg2 * delta
        if not np.all(np.isfinite(v)):
            import ipdb; ipdb.set_trace()
            pass
        if np.any(v == 0.):
            import ipdb; ipdb.set_trace()
            pass
        self.tau[:] = 1.0 / v
        if not np.all(np.isfinite(m)):
            import ipdb; ipdb.set_trace()
            pass
        self.eta[:] = self.tau * m
        self._time_elapsed['initialize'] += time() - before
        self._calls['initialize'] += 1

    # K is provided if low-rank stuff does not make sense
    def update(self, m, sigg2, delta, S, Q, L1, ttau, teta, A1, A0T, sigg2dotdQSQt, SQt, K=None):
        self._logger.debug('joint update has started')
        before_ = time()

        before = time()
        L1_Qt = cho_solve(L1, Q.T) # !!!CUBIC!!!
        self._time_elapsed['cho_solve'] += time() - before
        self._calls['cho_solve'] += 1

        before = time()
        L1_QtA1 = ddot(L1_Qt, A1, left=False)
        self._time_elapsed['ddot1'] += time() - before
        self._calls['ddot1'] += 1

        before = time()
        # SQt = ddot(S, Q.T, left=True)
        if delta != 0:
            C = ttau * A0T
            D = 1. - C
            # DQ = ddot(D, Q, left=True)
            p1 = ddot(D, sigg2dotdQSQt, left=True)
        else:
            p1 = sigg2dotdQSQt

        if delta != 0:
            p1 += D * sigg2 * delta
        self._time_elapsed['ddot2'] += time() - before
        self._calls['ddot2'] += 1

        # DQ_L1t = ddot(D, L1_Qt.T, left=True)
        before = time()
        if K is None:
            Z = np.linalg.multi_dot([L1_QtA1, Q, SQt]) # !!!CUBIC!!!
        else:
            Z = L1_QtA1.dot(K) # !!!CUBIC!!!
        self._time_elapsed['mult'] += time() - before
        self._calls['mult'] += 1

        before = time()
        if delta != 0:
            DQ = ddot(D, Q, left=True)
            p2 = - sigg2 * dotd(DQ, Z)
        else:
            p2 = - sigg2 * dotd(Q, Z)
        if delta != 0:
            p2 -= sigg2 * delta * dotd(DQ, L1_QtA1)
        self._time_elapsed['ddot3'] += time() - before
        self._calls['ddot3'] += 1

        r = p1 + p2
        self.tau[:] = 1./r

        before = time()

        if delta != 0:
            mu = m - C * m - D * dot(Q, dot(L1_QtA1, m))
            u = sigg2 * dot(Q, S * dot(Q.T, teta))
            u += sigg2 * delta * teta
            u = u - C * u
        else:
            mu = m - dot(Q, dot(L1_QtA1, m))
            u = sigg2 * dot(Q, S * dot(Q.T, teta))
            u += sigg2 * delta * teta


        if delta != 0:
            mu += u - sigg2 * dot(DQ, dot(Z, teta))
            mu -= sigg2 * delta * dot(DQ, dot(L1_QtA1, teta))
        else:
            mu += u - sigg2 * dot(Q, dot(Z, teta))
            mu -= sigg2 * delta * dot(Q, dot(L1_QtA1, teta))

        self._time_elapsed['dotvec'] += time() - before
        self._calls['dotvec'] += 1

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
