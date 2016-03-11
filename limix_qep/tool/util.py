def gower_normalization(K):
    """
    Perform Gower normalizion on covariance matrix K
    the rescaled covariance matrix has sample variance of 1
    """
    trPCP = K.trace() - K.mean(0).sum()
    r = (K.shape[0]-1) / trPCP
    return r * K
