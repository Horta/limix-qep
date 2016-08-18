from __future__ import absolute_import

import numpy as np
from numpy import dot
from numpy import full
from numpy.testing import assert_array_less

from limix_qep.tool.util import create_bernoulli
from limix_qep.tool.dataset import create_binomial

from limix_qep.tool.scan import scan
from limix_qep.tool import scan_binomial


def test_estimate_bernoulli_real_trait():
    np.random.seed(987)
    nsamples = 300
    nfeatures = 30
    h2 = 0.9

    (y, G) = create_bernoulli(nsamples, nfeatures, h2=h2, seed=894)
    K = dot(G[:, 0:10], G[:, 0:10].T)

    (pvals, _) = scan(y, G, K=K)

    assert_array_less(np.sum(pvals[10:]) / 20., np.sum(pvals[:10]) / 10.)
    assert_array_less(pvals[10:].min(), pvals[:10].min())
    assert_array_less(pvals.min(), 1e-5)
    assert_array_less(0.9, pvals.max())


def test_estimate_binomial_real_trait():
    seed = 3197
    nsamples = 100
    nfeatures = 30
    ntrials = 25

    (y, G) = create_binomial(nsamples, nfeatures, ntrials, sigg2=1.,
                             delta=0.01, sige2=0.01, seed=seed)

    K = dot(G[:, 0:10], G[:, 0:10].T)
    zG = np.hstack((np.zeros((nsamples, 1)), G))
    (pvals, info) = scan_binomial(y, full(nsamples, ntrials, float), zG, K=K)
    print(info)
    print(pvals)
    # print(pvals)

    assert_array_less(np.sum(pvals[11:]) / 20., np.sum(pvals[1:11]) / 10.)
    assert_array_less(pvals[11:].min(), pvals[1:11].min())
    assert_array_less(pvals[1:].min(), 1e-3)
