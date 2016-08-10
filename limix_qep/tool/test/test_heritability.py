# import numpy as np
# import unittest
# from limix_qep.tool.heritability import estimate
# from limix_qep.lik import Binomial
#
# class TestHeritability(unittest.TestCase):
#     def setUp(self):
#         pass
#
#     def test_h2_bernoulli(self):
#         random = np.random.RandomState(981)
#         n = 500
#         p = n+4
#
#         M = np.ones((n, 1)) * 0.4
#         G = random.randint(3, size=(n, p))
#         G = np.asarray(G, dtype=float)
#         G -= G.mean(axis=0)
#         G /= G.std(axis=0)
#         G /= np.sqrt(p)
#
#         K = np.dot(G, G.T)
#         Kg = K / K.diagonal().mean()
#         K = 0.5*Kg + 0.5*np.eye(n)
#         K = K / K.diagonal().mean()
#
#         z = random.multivariate_normal(M.ravel(), K)
#         y = np.zeros_like(z)
#         y[z>0] = 1.
#
#         h2 = estimate(y, K=Kg, covariate=M)[0]
#         self.assertAlmostEqual(h2, 0.403163261934, places=5)
#
#     # def test_h2_binomial(self):
#     #     random = np.random.RandomState(981)
#     #     ntrials = 1
#     #     n = 500
#     #     p = n+4
#     #
#     #     M = np.ones((n, 1)) * 0.4
#     #     G = random.randint(3, size=(n, p))
#     #     G = np.asarray(G, dtype=float)
#     #     G -= G.mean(axis=0)
#     #     G /= G.std(axis=0)
#     #     G /= np.sqrt(p)
#     #
#     #     K = np.dot(G, G.T)
#     #     Kg = K / K.diagonal().mean()
#     #     K = 0.5*Kg + 0.5*np.eye(n)
#     #     K = K / K.diagonal().mean()
#     #
#     #     z = random.multivariate_normal(M.ravel(), K)
#     #     y = np.zeros_like(z)
#     #     y[z>0] = 1.
#     #
#     #     outcome = Binomial(ntrials, n)
#     #     import logging
#     #     logging.basicConfig(level=logging.DEBUG)
#     #     h2 = estimate(y, K=Kg, covariate=M, outcome_type=outcome)[0]
#     #     self.assertAlmostEqual(h2, 0.403163261934)
#
#     # def test_h2_binomial_fast(self):
#     #     random = np.random.RandomState(981)
#     #     ntrials = 1
#     #     n = 50
#     #     p = n+4
#     #
#     #     M = np.ones((n, 1)) * 0.4
#     #     G = random.randint(3, size=(n, p))
#     #     G = np.asarray(G, dtype=float)
#     #     G -= G.mean(axis=0)
#     #     G /= G.std(axis=0)
#     #     G /= np.sqrt(p)
#     #
#     #     K = np.dot(G, G.T)
#     #     Kg = K / K.diagonal().mean()
#     #     K = 0.5*Kg + 0.5*np.eye(n)
#     #     K = K / K.diagonal().mean()
#     #
#     #     z = random.multivariate_normal(M.ravel(), K)
#     #     y = np.zeros_like(z)
#     #     y[z>0] = 1.
#     #
#     #     outcome = Binomial(ntrials, n)
#     #     # import logging
#     #     # logging.basicConfig(level=logging.DEBUG)
#     #     h2 = estimate(y, K=Kg, covariate=M, outcome_type=outcome)[0]
#     #     print(h2)
#     #     self.assertAlmostEqual(h2, 0.403163261934)
#
#
# if __name__ == '__main__':
#     unittest.main()
