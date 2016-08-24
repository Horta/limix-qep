# from __future__ import division
# import numpy as np
# from numpy import asarray
# import logging
# # from limix_qep.ep import EP
# from limix_qep import Bernoulli
# from limix_qep import Binomial
# from limix_math.linalg import economic_QS
# # from limix_tool.h2 import nh2
# from limix_tool.heritability import h2_correct
# from .util import gower_kinship_normalization
#
# class Model(object):
#     def __init__(self, ep, G, sub, div, ok):
#         self._ep = ep
#         self._G = G
#         self._sub = sub
#         self._div = div
#         if min(abs(div)) < 1e-14:
#             raise ValueError("min(abs(div)) < 1e-14")
#         self._ok = ok
#
#     def logpdf(self, y, ntrials, X, G):
#         ep = self._ep
#         ok = self._ok
#         sub = self._sub
#         div = self._div
#         G = (G[:, ok] - sub) / div
#         _G = (self._G - sub) / div
#
#         M = np.dot(X, self._ep.beta)
#
#         p = []
#         for (i, g) in enumerate(G):
#             # g = (g - sub) / div
#             var = ep.sigg2 * np.dot(g, g) + ep.sigg2 * ep.delta
#             covar = ep.sigg2 * np.dot(g, _G.T)
#             pred = ep.predict(M[i], var, covar)
#             p.append(pred.logpdf(y[i], ntrials[i])[0])
#
#         return p
#
#     def logpdf_outcome(self, y, ntrials, X, G):
#         ep = self._ep
#         ok = self._ok
#         sub = self._sub
#         div = self._div
#         G = (G[:, ok] - sub) / div
#         _G = (self._G - sub) / div
#
#         M = np.dot(X, self._ep.beta)
#
#         p = []
#         o = []
#         for (i, g) in enumerate(G):
#             var = ep.sigg2 * np.dot(g, g) + ep.sigg2 * ep.delta
#             covar = ep.sigg2 * np.dot(g, _G.T)
#             pred = ep.predict(M[i], var, covar)
#
#             logpdfs = []
#             for nt in range(ntrials[i] + 1):
#                 logpdfs.append(pred.logpdf(nt, ntrials[i])[0])
#
#             p.append(np.max(logpdfs))
#             o.append(np.argmax(logpdfs))
#
#         return (p, o)
#
#     def logpdf_outcome_mean(self, y, ntrials, X, G):
#         ep = self._ep
#         ok = self._ok
#         sub = self._sub
#         div = self._div
#         G = (G[:, ok] - sub) / div
#         _G = (self._G - sub) / div
#
#         M = np.dot(X, self._ep.beta)
#
#         p = []
#         o = []
#         m = []
#         for (i, g) in enumerate(G):
#             var = ep.sigg2 * np.dot(g, g) + ep.sigg2 * ep.delta
#             covar = ep.sigg2 * np.dot(g, _G.T)
#             pred = ep.predict(M[i], var, covar)
#
#             logpdfs = []
#             for nt in range(ntrials[i] + 1):
#                 logpdfs.append(pred.logpdf(nt, ntrials[i])[0])
#
#             p.append(np.max(logpdfs))
#             o.append(np.argmax(logpdfs))
#             m.append(sum(np.exp(logpdfs) * np.arange(ntrials[i]+1)))
#
#         return (p, o, m)
#
# def learn(y, ntrials, G, covariate):
#     assert G is not None
#
#     logger = logging.getLogger(__name__)
#     logger.info('Model learning has started.')
#
#     ok = np.std(G, 0) > 0.
#     G = G[:, ok]
#
#     sub = np.mean(G, 0)
#     div = np.std(G, 0) * np.sqrt(G.shape[1])
#
#     Graw = G
#     G = (G - sub) / div
#
#     y = asarray(y, dtype=float)
#     ntrials = asarray(ntrials, dtype=float)
#
#     logger.debug('Computing the economic eigen decomposition.')
#     (Q, S) = economic_QS(G, 'G')
#
#     logger.debug('Constructing EP.')
#     ep = EP(y, covariate, Q, S, outcome_type=Binomial(ntrials))
#     logger.debug('EP optimization.')
#     ep.optimize()
#
#     logger.info('Model learning has finished.')
#
#     return Model(ep, Graw, sub, div, ok)
#
# # if __name__ == '__main__':
# #     np.random.seed(987)
# #     ntrials = 1
# #     n = 5
# #     p = n+4
# #
# #     M = np.ones((n, 1)) * 0.4
# #     G = np.random.randint(3, size=(n, p))
# #     G = np.asarray(G, dtype=float)
# #     G -= G.mean(axis=0)
# #     G /= G.std(axis=0)
# #     G /= np.sqrt(p)
# #
# #     K = np.dot(G, G.T) + np.eye(n)*0.1
# #
# #     y = np.random.randint(ntrials + 1, size=n)
# #     y = np.asarray(y, dtype=float)
# #
# #     print(estimate(y, K=K, covariate=M))
