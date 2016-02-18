import numpy as np
from numpy import asarray
import logging
from limix_qep import EP
from limix_qep import Bernoulli
from limix_util.linalg import economic_QS
from limix_util.data_ import gower_kinship_normalization

def estimate(y, G=None, K=None, QS=None, covariate=None,
             outcome_type=Bernoulli(), prevalence=None):
    """Estimate the so-called narrow-sense heritability.

    It supports Bernoulli and Binomial phenotypes (see outcome_type).
    The user must specifiy only one of the parameters G, K, and QS for
    defining the genetic background.

    :param numpy.array y: Phenotype. The domain has be the non-negative
                          integers.
    :param numpy.array G: Genetic markers matrix used internally for kinship
                    estimation.
    :param numpy.array K: Kinship matrix.
    :param tuple QS: Economic eigen decomposition of the Kinship matrix.
    :return: a tuple containing the estimated heritability and additional
             information, respectively.
    """

    logger = logging.getLogger(__name__)
    logger.info('Heritability estimation has started.')
    y = asarray(y, dtype=float)

    info = dict()

    if K is not None:
        logger.debug('Covariace matrix normalization.')
        K = gower_kinship_normalization(K)
        info['K'] = K

    if G is not None:
        logger.debug('Genetic markers normalization.')
        G = G - np.mean(G, 0)
        s = np.std(G, 0)
        ok = s > 0.
        G[:,ok] /= s[ok]
        G /= np.sqrt(G.shape[1])
        info['G'] = G

    outcome_type.assert_outcome(y)

    if G is None and K is None and QS is None:
        raise Exception('G, K, and QS cannot be all None.')

    if QS is None:
        logger.debug('Computing the economic eigen decomposition.')
        (Q, S) = economic_QS((G, K), 'GK')
    else:
        Q = QS[0]
        S = QS[1]
        S /= np.mean(S)

    info['Q'] = Q
    info['S'] = S

    if covariate is None:
        logger.debug('Inserting offset covariate.')
        covariate = np.ones((y.shape[0], 1))

    info['covariate'] = covariate

    logger.debug('Constructing EP.')
    ep = EP(y, covariate, Q, S, outcome_type=outcome_type)
    logger.debug('EP optimization.')
    ep.optimize()

    if prevalence is None:
        h2 = ep.h2
    elif isinstance(outcome_type, Bernoulli):
        h2 = ep.h2
        # ascertainment = _calc_ascertainment(y)
        # h2 = _correct_h2(ep.h2, prevalence, ascertainment)

    info['ep'] = ep

    return (h2, info)

if __name__ == '__main__':
    np.random.seed(987)
    ntrials = 1
    n = 5
    p = n+4

    M = np.ones((n, 1)) * 0.4
    G = np.random.randint(3, size=(n, p))
    G = np.asarray(G, dtype=float)
    G -= G.mean(axis=0)
    G /= G.std(axis=0)
    G /= np.sqrt(p)

    K = np.dot(G, G.T) + np.eye(n)*0.1

    y = np.random.randint(ntrials + 1, size=n)
    y = np.asarray(y, dtype=float)

    print estimate(y, K=K, covariate=M)
