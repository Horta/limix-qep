import numpy as np
from numpy import dot

from limix_util.report import BeginEnd
from limix_qep.tool.scan import scan as scan
from limix_qep.tool.scan2 import scan as scan2

from limix_qep.tool.test.util import create_bernoulli

np.random.seed(987)
nsamples = 5000
nfeatures = 10000
h2 = 0.9

(y, G) = create_bernoulli(nsamples, nfeatures, h2=h2, seed=894)
K = dot(G, G.T)
# K = dot(G[:, 0:10], G[:, 0:10].T)

#with BeginEnd('scan1'):
#     (pvals, _) = scan(y, G, K=K)
with BeginEnd('scan2'):
    (pvals2, _) = scan2(y, G, K=K)
