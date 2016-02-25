import numpy as np

def gower_kinship_normalization(K):
    """
    Perform Gower normalizion on covariance matrix K
    the rescaled covariance matrix has sample variance of 1
    """
    n = K.shape[0]
    P = np.eye(n) - np.ones((n,n))/float(n)
    trPCP = np.trace(np.dot(P,np.dot(K,P)))
    r = (n-1) / trPCP
    return r * K
