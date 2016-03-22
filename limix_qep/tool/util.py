def gower_kinship_normalization(K):
    """
    Perform Gower normalizion on covariance matrix K
    the rescaled covariance matrix has sample variance of 1
    """
    c = (K.shape[0]-1) / (K.trace() - K.mean(0).sum())
    return c * K
