from __future__ import division

from numpy import zeros
from numpy import empty
from numpy import maximum

from limix_math.linalg import ddot
from limix_math.linalg import dotd

class SiteLik(object):
    def __init__(self, nsamples):
        self.tau = zeros(nsamples)
        self.eta = zeros(nsamples)

    def initialize(self):
        self.tau[:] = 0.
        self.eta[:] = 0.

    def update(self, ctau, ceta, hmu, hsig2):
        self.tau[:] = maximum(1.0 / hsig2 - ctau, 1e-16)
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
    def __init__(self, nsamples):
        self.tau = empty(nsamples)
        self.eta = empty(nsamples)

    def initialize(self, m, diagK):
        self.tau[:] = 1 / diagK
        self.eta[:] = m
        self.eta[:] *= self.tau

    def update(self, m, teta, A1, A2, QB1Qt, K):

        diagK = K.diagonal()
        QB1QtA1 = ddot(QB1Qt, A1, left=False)
        self.tau[:] = 1 / (A2 * diagK - A2 * dotd(QB1QtA1, K))

        Kteta = K.dot(teta)
        self.eta[:] = A2 * (m - QB1QtA1.dot(m) + Kteta - QB1QtA1.dot(Kteta))
        self.eta *= self.tau

class Cavity(object):
    def __init__(self, nsamples):
        self.tau = empty(nsamples)
        self.eta = empty(nsamples)

    def update(self, jtau, jeta, ttau, teta):
        self.tau[:] = jtau - ttau
        self.eta[:] = jeta - teta

    @property
    def mu(self):
        return self.eta / self.tau

    @property
    def sig2(self):
        return 1. / self.tau
