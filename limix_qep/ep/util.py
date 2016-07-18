import logging

def make_sure_reasonable_conditioning(S):
    max_cond = 1e6
    cond = S.max() / S.min()
    if cond > max_cond:
        eps = (max_cond * S.min() - S.max()) / (1 - max_cond)
        logger = logging.getLogger(__name__)
        logger.warn("The covariance matrix's conditioning number" +
                    " is too high: %e. Summing %e to its eigenvalues and " +
                    "renormalizing for a better conditioning number.",
                    cond, eps)
        m = S.mean()
        S += eps
        S *= m / S.mean()

def shall_use_low_rank_tricks(nsamples, nfeatures):
    return nsamples - nfeatures > 1
