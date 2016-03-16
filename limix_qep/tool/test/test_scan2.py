import unittest
import numpy as np
from numpy import dot

from util import create_bernoulli

from limix_qep.tool.scan import scan as scan
from limix_qep.tool.scan2 import scan as scan2


class TestScan2(unittest.TestCase):
    def setUp(self):
        pass

    def test_estimate_bernoulli_real_trait(self):
        np.random.seed(987)
        nsamples = 300
        nfeatures = 30
        h2 = 0.9

        (y, G) = create_bernoulli(nsamples, nfeatures, h2=h2, seed=894)
        K = dot(G[:, 0:10], G[:, 0:10].T)

        np.seterr(all='raise')
        (pvals, _) = scan(y, G, K=K)
        (pvals2, _) = scan2(y, G, K=K)

        np.testing.assert_almost_equal(pvals, pvals2)

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
