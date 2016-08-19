# from numpy import dot
# from numpy import sqrt
# from numpy import full
# from numpy import asarray
# from numpy import isscalar
# from numpy import empty
# from numpy import random
# from numpy import newaxis
#
#
# def gower_kinship_normalization(K):
#     """
#     Perform Gower normalizion on covariance matrix K
#     the rescaled covariance matrix has sample variance of 1
#     """
#     c = (K.shape[0]-1) / (K.trace() - K.mean(0).sum())
#     return c * K
#
#
# # K = \sigma_g^2 Q (S + \delta I) Q.T
# def create_binomial(nsamples, nfeatures, ntrials, var=0.8, delta=0.2,
#                     sige2=1., seed=None):
#     if seed is not None:
#         random.seed(seed)
#
#     if isscalar(ntrials):
#         ntrials = full(nsamples, ntrials, dtype=int)
#     else:
#         ntrials = asarray(ntrials, int)
#
#     X = random.randn(nsamples, nfeatures)
#     X -= X.mean(0)
#     X /= X.std(0)
#     X /= sqrt(nfeatures)
#
#     u = random.randn(nfeatures) * sqrt(var)
#
#     u -= u.mean()
#     u /= u.std()
#     u *= sqrt(var)
#
#     g1 = dot(X, u)
#     g1 -= g1.mean()
#     g1 /= g1.std()
#     g1 *= sqrt(var)
#     g2 = random.randn(nsamples)
#     g2 -= g2.mean()
#     g2 /= g2.std()
#     g2 *= sqrt(var * delta)
#
#     g = g1 + g2
#
#     E = random.randn(nsamples, max(ntrials))
#     E *= sqrt(sige2)
#
#     Z = g[:, newaxis] + E
#
#     Z[Z >  0.] = 1.
#     Z[Z <= 0.] = 0.
#
#     y = empty(nsamples)
#     for i in range(y.shape[0]):
#         y[i] = sum(Z[i,:ntrials[i]])
#
#     return (y, X)
#
# # K = \sigma_g^2 Q (S + \delta I) Q.T
# def create_bernoulli(nsamples, nfeatures, h2=0.5, seed=None):
#
#     import numpy as np
#     from numpy import dot
#     from numpy import newaxis
#
#     # K = \sigma_g^2 Q (S + \delta I) Q.T
#     def _create_binomial(nsamples, nfeatures, ntrials, sigg2=0.8, delta=0.2,
#                         sige2=1., seed=None):
#         if seed is not None:
#             np.random.seed(seed)
#
#         if np.isscalar(ntrials):
#             ntrials = np.full(nsamples, ntrials, dtype=int)
#         else:
#             ntrials = np.asarray(ntrials, int)
#
#         X = np.random.randn(nsamples, nfeatures)
#         X -= np.mean(X, 0)
#         X /= np.std(X, 0)
#         X /= np.sqrt(nfeatures)
#
#         u = np.random.randn(nfeatures) * np.sqrt(sigg2)
#
#         u -= np.mean(u)
#         u /= np.std(u)
#         u *= np.sqrt(sigg2)
#
#         g1 = dot(X, u)
#         g2 = np.random.randn(nsamples)
#         g2 -= np.mean(g2)
#         g2 /= np.std(g2)
#         g2 *= np.sqrt(sigg2 * delta)
#
#         g = g1 + g2
#
#         E = np.random.randn(nsamples, np.max(ntrials))
#         E *= np.sqrt(sige2)
#
#         Z = g[:, newaxis] + E
#
#         Z[Z >  0.] = 1.
#         Z[Z <= 0.] = 0.
#
#         y = np.empty(nsamples)
#         for i in range(y.shape[0]):
#             y[i] = np.sum(Z[i,:ntrials[i]])
#
#         return (y, X)
#
#
#     sige2 = 1.
#     sigg2 = h2 * sige2 / (1. - h2)
#     ntrials = 1
#
#     (y, X) = _create_binomial(nsamples, nfeatures, ntrials, sigg2=sigg2,
#                              delta=0., seed=seed)
#
#     return (y, X)
