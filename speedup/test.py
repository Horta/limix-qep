from __future__ import absolute_import
import unittest
import numpy as np
from numpy import dot

from limix_qep.tool.test.util import create_bernoulli
from limix_qep.tool.test.util import create_binomial

from limix_qep.lik import Binomial
from limix_qep.tool.scan import scan
from limix_util.time import Timer
from time import time


np.random.seed(987)
nsamples = 1000
nfeatures = 1500
h2 = 0.5

(y, G) = create_bernoulli(nsamples, nfeatures, h2=h2, seed=395)
K = dot(G, G.T)
(pvals, _) = scan(y, G, K=K)
pvals_ = np.load('pvals.npy')
print np.linalg.norm(pvals - pvals_)
# np.save('pvals.npy', pvals)


# ntimes = 10
# elapsed_times = []
# for i in range(ntimes):
#     (y, G) = create_bernoulli(nsamples, nfeatures, h2=h2, seed=i)
#     K = dot(G, G.T)
#     before = time()
#     (pvals, _) = scan(y, G, K=K)
#     elapsed_times.append(time() - before)
# print(np.median(elapsed_times))


# print(pvals[0])
# print(pvals[-1])
# assert abs(pvals[0]-0.603338066704) < 1e-6
# assert abs(pvals[-1]-0.672755849461) < 1e-6
# assert (np.sum(pvals[:10])/10. > np.sum(pvals[10:])/20.)
# assert (pvals[:10].min() > pvals[10:].min())
# assert (pvals.min() < 1e-6)
# assert (pvals.max() > 0.9)
