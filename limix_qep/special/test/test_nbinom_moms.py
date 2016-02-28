import unittest
import numpy as np
from limix_qep.special.nbinom_moms import moments_array3, init

class TestNBinomMoms(unittest.TestCase):
    def setUp(self):
        init(100)


    def _test_moments(self, N, K, mu, var, lmom0, mu_res, var_res):
        mu = np.asarray([mu])
        var = np.asarray([var])
        lmom0n, mu_resn, var_resn = np.empty_like(mu), np.empty_like(mu), \
                                    np.empty_like(mu)
        NN = np.empty_like(mu)
        NN[:] = N

        KK = np.empty_like(mu)
        KK[:] = K

        moments_array3(NN, KK, mu/var, 1./var, lmom0n, mu_resn, var_resn)
        # moments_array3(double[:] N, double[:] K,
        #              double[:] eta, double[:] tau,
        #              double[:] lmom0, double[:] mu_res, double[:] var_res)
        # (lmom0n, mu_resn, var_resn) = self._bm.moments(N, K, mu, var)
        self.assertAlmostEqual(lmom0n, lmom0, places=3)
        self.assertAlmostEqual(mu_resn, mu_res, places=3)
        self.assertAlmostEqual(var_resn, var_res, places=4)

    def test_moments(self):

        N = 5
        K = 2
        mu = 0.1
        var = 1.2
        lmom0 = -1.87923983187
        mu_res = -0.189211912705
        var_res = 0.256669390778
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = +10.
        lmom0 = -36.4957661804
        mu_res = 1.91509505632
        var_res = 0.266046261313
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = -10.
        lmom0 = -31.8411020448
        mu_res = -2.74640905228
        var_res = 0.356869389023
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = 0.2
        var = 1000.
        lmom0 = -5.07624795865
        mu_res = -0.27227922926
        var_res = 0.331437599227
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = 0.2
        var = 1e-5
        lmom0 = -1.38664791822
        mu_res = 0.199985619445
        var_res = 9.99967848389e-06
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = -0.3
        var = 1.3
        N = 100
        K = 1
        lmom0 = -3.71075916369
        mu_res = -2.2289279302
        var_res = 0.109233130032
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = -0.3
        var = 1.3
        N = 100
        K = 99
        lmom0 = -4.72814094543
        mu_res = 2.18025191107
        var_res = 0.101868652147
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = -100.3
        var = 1.3
        N = 100
        K = 99
        lmom0 = -3936.1768724
        mu_res = 0.018693308254
        var_res = 0.0156595155474
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = +50.3
        var = 1.3
        N = 100
        K = 99
        lmom0 = -549.835791078
        mu_res = 21.8437670635
        var_res = 0.565881735833
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = -100.3
        var = 100.3
        N = 100
        K = 1
        lmom0 = -51.8867510056
        mu_res = -2.64440600688
        var_res = 0.229818107683
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = 1.
        var = 1.
        N = 100
        K = 0
        lmom0 = -7.54193804467
        mu_res = -2.34928359673
        var_res = 0.142934913771
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = 1.
        var = 1.
        N = 20
        K = 0
        lmom0 = -5.30844526916
        mu_res = -1.64812044741
        var_res = 0.214799220845
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = -1.
        var = 1.
        N = 20
        K = 0
        lmom0 = -1.50587920825
        mu_res = -2.20355987497
        var_res = 0.364685148511
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = -1.
        var = 1.
        N = 20
        K = 10
        lmom0 = -3.50788671501
        mu_res = -0.0732703933624
        var_res = 0.073291719207
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = -1.
        var = 5.
        N = 500
        K = 10
        lmom0 = -5.02986563106
        mu_res = -2.06203456134
        var_res = 0.0169231050849
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = +1.
        var = 5.
        N = 500
        K = 10
        lmom0 = -5.85332990314
        mu_res = -2.05529748262
        var_res = 0.0167628500774
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = +1.
        var = 5.
        N = 500
        K = 0
        lmom0 = -3.28718157263
        mu_res = -3.82154166515
        var_res = 0.733531530368
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

        mu = +1.
        var = 5.
        N = 500
        K = 500
        lmom0 = -1.69042945587
        mu_res = 4.18977054616
        var_res = 1.14794246481
        self._test_moments(N, K, mu, var, lmom0, mu_res, var_res)

if __name__ == '__main__':
    import logging
    logging.basicConfig()
    unittest.main()
