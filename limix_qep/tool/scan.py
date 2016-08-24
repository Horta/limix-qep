from __future__ import absolute_import
from tabulate import tabulate
import logging
from numpy import asarray
import numpy as np
from numpy import dot
from limix_qep.lik import Bernoulli
from limix_qep.lik import Binomial
from limix_math.linalg import qs_decomposition
from limix_math.linalg import _QS_from_K_split
from .util import gower_kinship_normalization
import scipy.stats as st
from limix_qep.ep import BernoulliEP
from limix_qep.ep import BinomialEP


def _get_offset_covariate(covariate, n):
    if covariate is None:
        covariate = np.ones((n, 1))
    return covariate


class LRT(object):

    def __init__(self, y, Q0, Q1, S0, outcome_type=Bernoulli(), full=False,
                 covariate=None, null_model_only=False):

        self._logger = logging.getLogger(__name__)

        if not (isinstance(outcome_type, Bernoulli) or
                isinstance(outcome_type, Binomial)):
            raise Exception("Unrecognized outcome type.")

        outcome_type.assert_outcome(y)

        self._full = full
        self._y = y
        self._Q0 = Q0
        self._Q1 = Q1
        self._S0 = S0
        self._covariate = _get_offset_covariate(covariate, y.shape[0])
        self._outcome_type = outcome_type
        self._null_model_ready = False
        self._alt_model_ready = False

        self._genetic_variance = None
        self._instrumental_variance = None
        self._environmental_variance = None
        self._pvals = None
        self._lrs = None
        self._ep = None
        self._betas = None
        self._lml_null = np.nan
        self._X = None
        self._null_model_only = null_model_only
        self._lml_alt = None

    @property
    def genetic_variance(self):
        return self._genetic_variance

    @property
    def instrumental_variance(self):
        return self._instrumental_variance

    @property
    def environmental_variance(self):
        return self._environmental_variance

    @property
    def candidate_markers(self):
        return self._X

    @candidate_markers.setter
    def candidate_markers(self, X):
        if self._X is None:
            self._X = X
            self._alt_model_ready = False
        elif np.any(self._X != X):
            self._X = X
            self._alt_model_ready = False

    def _compute_statistics(self):
        self._logger.info('Statistics computation has started.')

        self._compute_null_model()
        if self._null_model_only:
            return
        if self._full:
            self._compute_alt_models_full()
        else:
            self._compute_alt_models()

    def _compute_alt_models(self):
        if self._alt_model_ready:
            return

        self._logger.info('Alternative model computation has started.')

        X = self._X
        covariate = self._covariate
        lml_null = self._lml_null

        ep = self._ep
        ep.pause = True

        fp_lml_alt = np.full(X.shape[1], -np.inf)
        fep = ep.fixed_ep()

        t = self._lml_alts(fep, X, covariate)
        fp_lml_alt[:] = np.maximum(fp_lml_alt, t)
        self._lml_alt = fp_lml_alt

        fp_lrs = -2 * lml_null + 2 * fp_lml_alt
        chi2 = st.chi2(df=1)
        fp_pvals = chi2.sf(fp_lrs)

        self._pvals = fp_pvals
        self._lrs = fp_lrs
        self._alt_model_ready = True

    def _compute_null_model(self):
        if self._null_model_ready:
            return

        self._logger.info('Null model computation has started.')

        y = self._y
        Q0, Q1 = self._Q0, self._Q1
        S0 = self._S0
        covariate = self._covariate
        outcome_type = self._outcome_type

        if isinstance(outcome_type, Binomial):
            ep = BinomialEP(y, outcome_type.ntrials, covariate, Q0, Q1, S0)
        elif isinstance(outcome_type, Bernoulli):
            ep = BernoulliEP(y, covariate, Q0, Q1, S0)

        ep.optimize()

        self._lml_null = ep.lml()
        self._ep = ep

        self._genetic_variance = ep.genetic_variance
        self._instrumental_variance = ep.instrumental_variance
        self._environmental_variance = ep.environmental_variance
        self._null_model_ready = True

    def _compute_alt_models_full(self):
        if self._alt_model_ready:
            return

        X = self._X
        covariate = self._covariate
        ep = self._ep

        lml_alts = []
        for i in range(X.shape[1]):
            ep.M = np.hstack((covariate, X[:, i][:, np.newaxis]))
            assert False, 'fix me'
            # ep.optimize(only_step2=True)
            lml_alts.append(ep.lml())

        lml_alts = np.asarray(lml_alts, float)

        lrs = -2 * self._lml_null + 2 * lml_alts
        chi2 = st.chi2(df=1)
        pvals = chi2.sf(lrs)

        self._pvals = pvals
        self._lrs = lrs
        self._alt_model_ready = True

    def _lml_alts(self, fep, X, covariate=None):
        if covariate is None:
            covariate = np.ones((X.shape[0], 1))
        lml_alt = []

        p = covariate.shape[1]
        acov = np.hstack((covariate, X))
        self._logger.debug('Finding optimal betas.')
        if p == 1:
            betas = fep.optimal_betas(acov, 1)
        else:
            betas = fep.optimal_betas_general(acov, p)

        ms = dot(covariate, betas[:p, :]) + X * betas[p, :]
        lml_alt = fep.lmls(ms)
        self._betas = betas[p, :]
        return lml_alt

    def lml_alt(self):
        return self._lml_alt

    @property
    def effsizes(self):
        return self._betas

    def lrs(self):
        self._compute_statistics()
        if self._null_model_only:
            return []
        return self._lrs

    def pvals(self):
        self._compute_statistics()
        if self._null_model_only:
            return []
        return self._pvals

    def ep(self):
        self._compute_statistics()
        return self._ep

    def is_valid(self, y, QS, covariate, outcome_type):
        Q = QS[0]
        S = QS[1]
        if np.any(y != self._y):
            return False
        if Q[0, 0] != self._Q[0, 0] or S[0] != self._S[0]:
            return False
        if np.any(S != self._S):
            return False
        if np.any(Q != self._Q):
            return False
        n = y.shape[0]
        if np.any(_get_offset_covariate(covariate, n) != self._covariate):
            return False
        if outcome_type != self._outcome_type:
            return False
        return True


# def _create_LRT(y, QS, covariate, outcome_type, null_model_only):
#     do_create = False
#
#     if _create_LRT.cache is None:
#         do_create = True
#     else:
#         do_create = not _create_LRT.cache.is_valid(y, QS, covariate,
#                                                    outcome_type)
#
#     if do_create:
#         _create_LRT.cache = LRT(y, QS, covariate=covariate,
#                                 outcome_type=outcome_type,
#                                 null_model_only=null_model_only)
#
#     return _create_LRT.cache
# _create_LRT.cache = None

def _create_LRT(y, Q0, Q1, S0, covariate, outcome_type, null_model_only):
    return LRT(y, Q0, Q1, S0, covariate=covariate, outcome_type=outcome_type,
               null_model_only=null_model_only)


def scan(y, X, G=None, K=None, QS=None, covariate=None,
         outcome_type=None, null_model_only=False):
    """Perform association scan between genetic markers and phenotype.

    Matrix `X` shall contain the genetic markers (e.g., number of minor alleles)
    with rows and columsn representing samples and genetic markers,
    respectively.

    It supports Bernoulli and Binomial phenotypes (see `outcome_type`).
    The user must specifiy only one of the parameters `G`, `K`, and `QS` for
    defining the genetic background.

    Let :math:`N` be the sample size, :math:`S` the number of covariates,
    :math:`P_c` the number of genetic markers to be tested, and :math:`P_b`
    the number of genetic markers used for Kinship estimation.

    :param numpy.ndarray y: Phenotype. The domain has be the non-negative
                          integers. Dimension (:math:`N\\times 0`).
    :param numpy.ndarray X: Candidate genetic markers whose association with the
                          phenotype will be tested. Dimension
                          (:math:`N\\times P_c`).
    :param numpy.ndarray G: Genetic markers matrix used internally for kinship
                          estimation. Dimension (:math:`N\\times P_b`).
    :param numpy.ndarray K: Kinship matrix. Dimension (:math:`N\\times N`).
    :param tuple QS:      Economic eigen decomposition of the Kinship matrix.
    :param numpy.array covariate: Covariates. Default is an offset.
                                  Dimension (:math:`N\\times S`).
    :param object oucome_type: Either :class:`limix_qep.Bernoulli` (default)
                               or a :class:`limix_qep.Binomial` instance.
    :return:              a tuple containing the estimated p-values and
                          additional information, respectively.
    """

    if outcome_type is None:
        outcome_type = Bernoulli()

    logger = logging.getLogger(__name__)
    logger.info('Association scan has started.')
    y = asarray(y, dtype=float)

    info = dict()

    if K is not None:
        logger.info('Covariace matrix normalization.')
        K = gower_kinship_normalization(K)
        info['K'] = K

    if G is not None:
        logger.info('Genetic markers normalization.')
        G = G - np.mean(G, 0)
        s = np.std(G, 0)
        ok = s > 0.
        G[:, ok] /= s[ok]
        G /= np.sqrt(G.shape[1])
        info['G'] = G

    outcome_type.assert_outcome(y)
    if G is None and K is None and QS is None:
        raise Exception('G, K, and QS cannot be all None.')

    if QS is None:
        logger.info('Computing the economic eigen decomposition.')
        if K is None:
            QS = qs_decomposition(G)
        else:
            QS = _QS_from_K_split(K)

        Q0, Q1 = QS[0]
        S0 = QS[1][0]
    else:
        Q0 = QS[0]
        S0 = QS[1]
        S0 /= S0.mean()

    logger.info('Genetic marker candidates normalization.')
    X = X - np.mean(X, 0)
    s = np.std(X, 0)
    ok = s > 0.
    X[:, ok] /= s[ok]
    X /= np.sqrt(X.shape[1])
    info['X'] = X

    lrt = _create_LRT(y, Q0, Q1, S0, covariate, outcome_type,
                      null_model_only=null_model_only)
    lrt.candidate_markers = X
    info['lrs'] = lrt.lrs()
    info['effsizes'] = lrt.effsizes
    return_ = (lrt.pvals(), info)

    return return_


def scan_binomial(nsuccesses, ntrials, X, G=None, K=None, covariate=None):
    """Perform association scan between genetic markers and phenotype.

    Matrix `X` shall contain the genetic markers (e.g., number of minor alleles)
    with rows and columsn representing samples and genetic markers,
    respectively.

    The user must specify only one of the parameters `G` and `K` for defining
    the genetic background.

    Let :math:`N` be the sample size, :math:`S` the number of covariates,
    :math:`P_c` the number of genetic markers to be tested, and :math:`P_b`
    the number of genetic markers used for Kinship estimation.

    Args:
        nsuccesses (numpy.ndarray): Phenotype described by the number of
                                    successes, as non-negative integers.
                                    Dimension (:math:`N\\times 0`).
        ntrials    (numpy.ndarray): Phenotype described by the number of
                                    trials, as positive integers. Dimension
                                    (:math:`N\\times 0`).
        X          (numpy.ndarray): Candidate genetic markers (or any other
                                    type of explanatory variable) whose
                                    association with the phenotype will be tested. Dimension
                                    (:math:`N\\times P_c`).
        G          (numpy.ndarray): Genetic markers matrix used internally for kinship
                                    estimation. Dimension (:math:`N\\times P_b`).
        K          (numpy.ndarray): Kinship matrix. Dimension (:math:`N\\times N`).
        covariate  (numpy.ndarray): Covariates. Default is an offset.
                                    Dimension (:math:`N\\times S`).
    Returns:
        tuple: The estimated p-values and additional information, respectively.
    """

    logger = logging.getLogger(__name__)
    logger.info('Association scan has started.')
    nsuccesses = asarray(nsuccesses, dtype=float)

    print("Number of candidate markers to scan: %d" % X.shape[1])

    info = dict()

    if K is not None:
        logger.info('Covariace matrix normalization.')
        K = gower_kinship_normalization(K)
        info['K'] = K

    if G is not None:
        logger.info('Genetic markers normalization.')
        G = G - np.mean(G, 0)
        s = np.std(G, 0)
        ok = s > 0.
        G[:, ok] /= s[ok]
        G /= np.sqrt(G.shape[1])
        info['G'] = G

    if G is None and K is None:
        raise Exception('G, K, and QS cannot be all None.')

    logger.info('Computing the economic eigen decomposition.')
    if K is None:
        QS = qs_decomposition(G)
    else:
        QS = _QS_from_K_split(K)

    logger.info('Genetic marker candidates normalization.')
    X = X - np.mean(X, 0)
    s = np.std(X, 0)
    ok = s > 0.
    X[:, ok] /= s[ok]
    info['X'] = X

    Q0, Q1 = QS[0]
    S0 = QS[1][0]

    print("Scan has began...")
    lrt = _create_LRT(nsuccesses, Q0, Q1, S0, covariate, Binomial(ntrials),
                      null_model_only=False)
    lrt.candidate_markers = X
    info['lrs'] = lrt.lrs()
    info['effsizes'] = lrt.effsizes
    info['ep_null_model'] = lrt._ep
    info['lml_alt'] = lrt.lml_alt()
    return_ = (lrt.pvals(), info)
    print("Scan has finished.")

    print("-------------------------- NULL MODEL --------------------------")
    print(lrt._ep)
    print("----------------------------------------------------------------")
    print("")

    table = [info['effsizes'], info['lml_alt'], info['lrs'], lrt.pvals()]
    table = [list(i) for i in table]
    table = map(list, zip(*table))
    print("---------------------- ALTERNATIVE MODELs ----------------------")
    print(tabulate(table, headers=('EffSiz', 'LML', 'LR', 'Pval')))
    print("----------------------------------------------------------------")

    return return_
