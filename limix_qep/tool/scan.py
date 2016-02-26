import logging
from numpy import asarray
from limix_qep.ep import EP
import numpy as np
from numpy import dot
import scipy as sp
from limix_qep import Bernoulli, Binomial
from limix_math.linalg import economic_QS
from util import gower_kinship_normalization
import scipy.stats as st

class LRT(object):
    def __init__(self, X, y, QS, outcome_type=Bernoulli(), full=False,
                    covariate=None):

        self._logger = logging.getLogger(__name__)

        if covariate is None:
            covariate = np.ones((y.shape[0], 1))

        if not (isinstance(outcome_type, Bernoulli) or
                isinstance(outcome_type, Binomial)):
            raise Exception("Unrecognized outcome type.")

        outcome_type.assert_outcome(y)

        self._full = full
        self._y = y
        self._Q = QS[0]
        self._S = QS[1]
        self._covariate = covariate
        self._X = X
        self._outcome_type = outcome_type
        self._computed = False

        self._varg = np.nan
        self._varo = np.nan
        self._vare = np.nan
        self._pvals = None
        self._lrs = None
        self._ep = None
        self._betas  = None
        self._lml_null = np.nan

    @property
    def varg(self):
        return self._varg

    @property
    def varo(self):
        return self._varo

    @property
    def vare(self):
        return self._vare

    def _compute_statistics(self):

        if self._computed:
            return

        self._logger.info('Statistics computation has started.')

        self._compute_null_model()
        if self._full:
            self._compute_alt_models_full()
        else:
            self._compute_alt_models()

        self._computed = True

    def _compute_alt_models(self):
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

        fp_lrs = -2 * lml_null + 2 * fp_lml_alt
        chi2 = st.chi2(df=1)
        fp_pvals = chi2.sf(fp_lrs)

        self._pvals = fp_pvals
        self._lrs = fp_lrs

    def _compute_null_model(self):
        self._logger.info('Null model computation has started.')

        y = self._y
        Q = self._Q
        S = self._S
        covariate = self._covariate
        outcome_type = self._outcome_type

        ep = EP(y, covariate, Q, S, outcome_type)
        ep.optimize()

        lml_null = ep.lml()

        self._lml_null = lml_null
        self._ep = ep
        # self._ep_sites = ep.get_site_likelihoods()

        varg = ep.sigg2
        varo = ep.sigg2 * ep.delta
        vare = 1.
        vart = varg + varo + vare

        self._varg = varg / vart
        self._varo = varo / vart
        self._vare = vare / vart

    def _compute_alt_models_full(self):
        X = self._X
        covariate = self._covariate
        ep = self._ep

        lml_alts = []
        for i in xrange(X.shape[1]):
            ep.M = np.hstack( (covariate, X[:,i][:,np.newaxis]) )
            assert False, 'fix me'
            # ep.optimize(only_step2=True)
            lml_alts.append(ep.lml())

        lml_alts = np.asarray(lml_alts, float)

        lrs = -2 * self._lml_null + 2 * lml_alts
        chi2 = st.chi2(df=1)
        pvals = chi2.sf(lrs)

        self._pvals = pvals
        self._lrs = lrs

    def _lml_alts(self, fep, X, covariate=None):
        if covariate is None:
            covariate = np.ones((X.shape[0], 1))
        lml_alt = []

        p = covariate.shape[1]
        acov = np.hstack( (covariate, X) )
        self._logger.debug('Finding optimal betas.')
        if p == 1:
            betas = fep.optimal_betas(acov, 1)
        else:
            betas = fep.optimal_betas_general(acov, p)

        ms = dot(covariate, betas[:p,:]) + X * betas[p,:]
        lml_alt = fep.lmls(ms)
        self._betas = betas[p,:]
        return lml_alt

    @property
    def effsizes(self):
        return self._betas

    def lrs(self):
        self._compute_statistics()
        return self._lrs

    def pvals(self):
        self._compute_statistics()
        return self._pvals

    def ep(self):
        self._compute_statistics()
        return self._ep

def scan(y, X, G=None, K=None, QS=None, covariate=None,
         outcome_type=None):
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
        QS = economic_QS((G, K), 'GK')
    else:
        QS[1] /= QS[1].mean()

    logger.debug('Genetic marker candidates normalization.')
    X = X - np.mean(X, 0)
    s = np.std(X, 0)
    ok = s > 0.
    X[:,ok] /= s[ok]
    X /= np.sqrt(X.shape[1])
    info['X'] = X

    lrt = LRT(X, y, QS, covariate=covariate, outcome_type=outcome_type)
    info['lrs'] = lrt.lrs()
    info['effsizes'] = lrt.effsizes
    return (lrt.pvals(), info)
