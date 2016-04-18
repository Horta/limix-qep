from __future__ import division
import numpy as np
from numpy import asarray
import logging
from limix_qep.ep import EP
from limix_qep import Bernoulli
from limix_qep import Binomial
from limix_math.linalg import economic_QS
# from limix_tool.h2 import nh2
from limix_tool.heritability import h2_correct
from .util import gower_kinship_normalization

class Model(object):
    def __init__(self, ep, G, sub, mult, ok):
        self._ep = ep
        self._G = G
        self._sub = sub
        self._mult = mult
        self._ok = ok

    def predict(self, X, G):
        p = []
        M = np.dot(X, self._ep.beta[:,np.newaxis])
        for (i, g) in enumerate(G):
            g = g - self._sub
            g[self._ok] *= self._mult[self._ok]
            var = np.dot(g, g)
            covar = np.dot(g, self._G.T)
            p.append(self._ep.predict(M[i,:], var, covar))
        return p

def learn(y, G, covariate,
             outcome_type=None, prevalence=None):

    assert G is not None
    Gbak = G.copy()
    sub = np.mean(G, 0)
    G = G - sub
    mult = np.std(G, 0)
    ok = mult > 0.
    mult *= np.sqrt(G.shape[1])
    mult = 1./mult[ok]
    G[:,ok] *= mult[ok]
    # G[:,ok] /= std[ok]
    # G /= np.sqrt(G.shape[1])

    if outcome_type is None:
        outcome_type = Bernoulli()

    logger = logging.getLogger(__name__)
    logger.info('Model learning has started.')
    y = asarray(y, dtype=float)

    outcome_type.assert_outcome(y)

    logger.debug('Computing the economic eigen decomposition.')
    (Q, S) = economic_QS(G, 'G')

    logger.debug('Constructing EP.')
    ep = EP(y, covariate, Q, S, outcome_type=outcome_type)
    logger.debug('EP optimization.')
    ep.optimize()

    logger.info('Model learning has finished.')

    return Model(ep, Gbak, sub, mult, ok)

# if __name__ == '__main__':
#     np.random.seed(987)
#     ntrials = 1
#     n = 5
#     p = n+4
#
#     M = np.ones((n, 1)) * 0.4
#     G = np.random.randint(3, size=(n, p))
#     G = np.asarray(G, dtype=float)
#     G -= G.mean(axis=0)
#     G /= G.std(axis=0)
#     G /= np.sqrt(p)
#
#     K = np.dot(G, G.T) + np.eye(n)*0.1
#
#     y = np.random.randint(ntrials + 1, size=n)
#     y = np.asarray(y, dtype=float)
#
#     print(estimate(y, K=K, covariate=M))
