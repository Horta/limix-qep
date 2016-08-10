from numpy import dot
from numpy import sqrt
from numpy import full
from numpy import asarray
from numpy import isscalar
from numpy import empty
from numpy import random
from numpy import newaxis


def gower_kinship_normalization(K):
    """
    Perform Gower normalizion on covariance matrix K
    the rescaled covariance matrix has sample variance of 1
    """
    c = (K.shape[0]-1) / (K.trace() - K.mean(0).sum())
    return c * K


# K = \sigma_g^2 Q (S + \delta I) Q.T
def create_binomial(nsamples, nfeatures, ntrials, var=0.8, delta=0.2,
                    sige2=1., seed=None):
    if seed is not None:
        random.seed(seed)

    if isscalar(ntrials):
        ntrials = full(nsamples, ntrials, dtype=int)
    else:
        ntrials = asarray(ntrials, int)

    X = random.randn(nsamples, nfeatures)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(nfeatures)

    u = random.randn(nfeatures) * sqrt(var)

    u -= u.mean()
    u /= u.std()
    u *= sqrt(var)

    g1 = dot(X, u)
    g1 -= g1.mean()
    g1 /= g1.std()
    g1 *= sqrt(var)
    g2 = random.randn(nsamples)
    g2 -= g2.mean()
    g2 /= g2.std()
    g2 *= sqrt(var * delta)

    g = g1 + g2

    E = random.randn(nsamples, max(ntrials))
    E *= sqrt(sige2)

    Z = g[:, newaxis] + E

    Z[Z >  0.] = 1.
    Z[Z <= 0.] = 0.

    y = empty(nsamples)
    for i in range(y.shape[0]):
        y[i] = sum(Z[i,:ntrials[i]])

    return (y, X)
