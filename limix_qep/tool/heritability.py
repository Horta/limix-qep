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

def _ascertainment(y):
    u = np.unique(y)
    assert len(u) == 2
    ma = np.max(u)
    ascertainment = np.sum(y == ma) / len(y)
    return ascertainment

def estimate(y, G=None, K=None, QS=None, covariate=None,
             outcome_type=None, prevalence=None):
    """Estimate the so-called narrow-sense heritability.

    It supports Bernoulli and Binomial phenotypes (see `outcome_type`).
    The user must specifiy only one of the parameters G, K, and QS for
    defining the genetic background.

    Let :math:`N` be the sample size, :math:`S` the number of covariates, and
    :math:`P_b` the number of genetic markers used for Kinship estimation.

    :param numpy.ndarray y: Phenotype. The domain has be the non-negative
                          integers. Dimension (:math:`N\\times 0`).
    :param numpy.ndarray G: Genetic markers matrix used internally for kinship
                    estimation. Dimension (:math:`N\\times P_b`).
    :param numpy.ndarray K: Kinship matrix. Dimension (:math:`N\\times N`).
    :param tuple QS: Economic eigen decomposition of the Kinship matrix.
    :param numpy.ndarray covariate: Covariates. Default is an offset.
                                  Dimension (:math:`N\\times S`).
    :param object oucome_type: Either :class:`limix_qep.Bernoulli` (default)
                               or a :class:`limix_qep.Binomial` instance.
    :param float prevalence: Population rate of cases for dichotomous
                             phenotypes. Typically useful for case-control
                             studies.
    :return: a tuple containing the estimated heritability and additional
             information, respectively.
    """

    if outcome_type is None:
        outcome_type = Bernoulli()

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
        logger.info('Found heritability before correction: %.5f.', h2)
    elif isinstance(outcome_type, Bernoulli):
        h2 = ep.h2
        logger.info('Found heritability before correction: %.5f.', h2)
        ascertainment = _ascertainment(y)
        h2 = h2_correct(ep.h2, prevalence, ascertainment)
        # h2 = nh2(ep.h2, ascertainment, prevalence)
    elif isinstance(outcome_type, Binomial):
        h2 = ep.h2

    logger.info('Found heritability after correction: %.5f.', h2)

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

    print(estimate(y, K=K, covariate=M))
