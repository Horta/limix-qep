# import numpy as np
# import unittest
# from limix_qep.tool.predict import learn
#
# class TestPredict(unittest.TestCase):
#     def setUp(self):
#         pass
#
#     def test_bernoulli_learn(self):
#         random = np.random.RandomState(981)
#         n = 500
#         p = n+4
#
#         X = np.ones((n, 1))
#         G = random.randint(3, size=(n, p))
#         G = np.asarray(G, dtype=float)
#         G -= G.mean(axis=0)
#         G /= G.std(axis=0)
#         G /= np.sqrt(p)
#
#         K = np.dot(G, G.T)
#         Kg = K / K.diagonal().mean()
#         K = 0.9*Kg + 0.1*np.eye(n)
#         K = K / K.diagonal().mean()
#
#         z = random.multivariate_normal(X.ravel() * 0.4, K)
#         y = np.zeros_like(z)
#         y[z>0] = 1.
#
#         model = learn(y, G=G, covariate=X)
#         p = model.predict(X, G)
#         self.assertAlmostEqual(0.857912746548,
#                                np.mean([p[i].pdf(y[i])[0] for i in range(n)]),
#                                places=4)
#         self.assertAlmostEqual(0.142087253452,
#                                np.mean([1-p[i].pdf(y[i])[0] for i in range(n)]),
#                                places=4)
#
#     def test_bernoulli_predict(self):
#         random = np.random.RandomState(981)
#         n = 500
#         p = n+4
#
#         X = np.ones((n, 1))
#         G = random.randint(3, size=(n, p))
#         G = np.asarray(G, dtype=float)
#         G -= G.mean(axis=0)
#         G /= G.std(axis=0)
#         G /= np.sqrt(p)
#
#         K = np.dot(G, G.T)
#         Kg = K / K.diagonal().mean()
#         K = 0.9*Kg + 0.1*np.eye(n)
#         K = K / K.diagonal().mean()
#
#         z = random.multivariate_normal(X.ravel() * 0.4, K)
#         y = np.zeros_like(z)
#         y[z>0] = 1.
#
#         model = learn(y[:400], G=G[:400,:], covariate=X[:400,:])
#         p = model.predict(X[400:,:], G[400:,:])
#         # print(p)
#
# if __name__ == '__main__':
#     unittest.main()
