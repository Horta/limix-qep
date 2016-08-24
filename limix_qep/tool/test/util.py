# import numpy as np
# from numpy import dot
# from numpy import newaxis
#
# # K = \sigma_g^2 Q (S + \delta I) Q.T
# def create_binomial(nsamples, nfeatures, ntrials, sigg2=0.8, delta=0.2,
#                     sige2=1., seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#
#     if np.isscalar(ntrials):
#         ntrials = np.full(nsamples, ntrials, dtype=int)
#     else:
#         ntrials = np.asarray(ntrials, int)
#
#     X = np.random.randn(nsamples, nfeatures)
#     X -= np.mean(X, 0)
#     X /= np.std(X, 0)
#     X /= np.sqrt(nfeatures)
#
#     u = np.random.randn(nfeatures) * np.sqrt(sigg2)
#
#     u -= np.mean(u)
#     u /= np.std(u)
#     u *= np.sqrt(sigg2)
#
#     g1 = dot(X, u)
#     g2 = np.random.randn(nsamples)
#     g2 -= np.mean(g2)
#     g2 /= np.std(g2)
#     g2 *= np.sqrt(sigg2 * delta)
#
#     g = g1 + g2
#
#     E = np.random.randn(nsamples, np.max(ntrials))
#     E *= np.sqrt(sige2)
#
#     Z = g[:, newaxis] + E
#
#     Z[Z >  0.] = 1.
#     Z[Z <= 0.] = 0.
#
#     y = np.empty(nsamples)
#     for i in range(y.shape[0]):
#         y[i] = np.sum(Z[i,:ntrials[i]])
#
#     return (y, X)
#
# # K = \sigma_g^2 Q (S + \delta I) Q.T
# def create_bernoulli(nsamples, nfeatures, h2=0.5, seed=None):
#
#     sige2 = 1.
#     sigg2 = h2 * sige2 / (1. - h2)
#     ntrials = 1
#
#     (y, X) = create_binomial(nsamples, nfeatures, ntrials, sigg2=sigg2,
#                              delta=0., seed=seed)
#
#     return (y, X)
