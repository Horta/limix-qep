import logging
from numpy import asarray
from gwarped.gp.ep import EP
import numpy as np
from numpy import dot
import scipy as sp
from limix_qep.lik import Bernoulli, Binomial
from limix_util.data_ import gower_kinship_normalization
from gwarped.util.linalg import economic_QS

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

        self._logger.info('Statistics computation.')

        self._compute_null_model()
        if self._full:
            self._compute_alt_models_full()
        else:
            self._compute_alt_models()

        self._computed = True

    def _compute_alt_models(self):
        self._logger.info('Alternative model computation.')

        y = self._y
        Q = self._Q
        S = self._S
        X = self._X
        covariate = self._covariate
        outcome_type = self._outcome_type
        lml_null = self._lml_null

        ep = self._ep
        ep._freeze_this_thing = True

        fp_lml_alt = np.full(X.shape[1], -np.inf)
        fep = ep.fixed_ep()

        t = self._lml_alts(fep, X, covariate)
        fp_lml_alt[:] = np.maximum(fp_lml_alt, t)

        fp_lrs = -2 * lml_null + 2 * fp_lml_alt
        chi2 = sp.stats.chi2(df=1)
        fp_pvals = chi2.sf(fp_lrs)

        self._pvals = fp_pvals
        self._lrs = fp_lrs
        self._ep = ep

    def _compute_null_model(self):
        self._logger.info('Null model computation.')

        y = self._y
        Q = self._Q
        S = self._S
        covariate = self._covariate
        outcome_type = self._outcome_type

        ep = EP(y, covariate, Q, S, outcome_type)
        ep.optimize()

        lml_null = ep.lml()
        sigg2_null = ep.sigg2
        delta_null = ep.delta
        beta_null = ep.beta

        self._lml_null = lml_null
        self._sigg2_null = sigg2_null
        self._delta_null = delta_null
        self._beta_null = beta_null
        self._ep = ep
        self._ep_sites = ep.get_site_likelihoods()

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
            ep.optimize(only_step2=True)
            lml_alts.append(ep.lml())

        lml_alts = np.asarray(lml_alts, float)

        lrs = -2 * self._lml_null + 2 * lml_alts
        chi2 = sp.stats.chi2(df=1)
        pvals = chi2.sf(lrs)

    def _lml_alts(self, fep, X, covariate=None):
        if covariate is None:
            covariate = np.ones((X.shape[0], 1))
        lml_alt = []
        time_lml = 0.

        try:
            p = covariate.shape[1]
            acov = np.hstack( (covariate, X) )
            print 'Finding optimal betas...'
            if p == 1:
                betas = fep.optimal_betas(acov, 1)
            else:
                betas = fep.optimal_betas_general(acov, p)
            print 'Done.'
        except Exception as e:
            pass

        # betas = fep.optimal_betas(np.hstack( (covariate, X) ),
        #                            covariate.shape[1])

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
         outcome_type=Bernoulli(), prevalence=None):

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
        (Q, S) = economic_QS((G, K), 'GK')
    else:
        Q = QS[0]
        S = QS[1]
        S /= np.mean(S)

    logger.debug('Genetic marker candidates normalization.')
    X = X - np.mean(X, 0)
    s = np.std(X, 0)
    ok = s > 0.
    X[:,ok] /= s[ok]
    X /= np.sqrt(X.shape[1])
    info['X'] = X

    lrt = LRT(X, y, K, covariate=covariate)
    info['lrs'] = lrt.lrs()
    info['effsizes'] = lrt.effsizes
    return lrt.pvals()

if __name__ == '__main__':
    pass
    # # np.random.seed(978)
    # # np.seterr(all='raise')
    # # # n = 10
    # # # p = 5
    # # # X = np.random.rand(n, p)
    # # # v = dot(X, X.T).diagonal().mean()
    # # # X = X / v
    # # # G = X.copy()
    # # # K = dot(G, G.T)
    # # # y = np.asarray(np.random.randint(0, 2, n), dtype=float)
    # #
    # # # pvals = test_lmm(X, y, G)
    # # # ipvals = np.array([0.13553864, 0.57124235, 0.45712805, 0.02639387, 0.14379783])
    # # # np.testing.assert_allclose(pvals, ipvals, rtol=1e-6)
    # #
    # data = np.load('/Users/horta/workspace/limix-gwarped-exp/gwarped_exp/genetics/pb/data_1k_50k.npz')
    # G = data['X'][0:300,0:800]
    # X = G
    # y = data['y'][0:300]
    # n = y.shape[0]
    # covariate = np.ones((n, 1))
    # #
    # # (pvals, lrs, ep) = lrt(X, y, G, covariate=covariate)
    # # print pvals
    # #
    # # # test_glmm_adjust(ep, pvals, X, ntop=5)
    #
    # np.random.seed(398348)
    # lrt = LRT2(X, y, G=G, outcome_type=Bernoulli())
    # print lrt.pvals()
