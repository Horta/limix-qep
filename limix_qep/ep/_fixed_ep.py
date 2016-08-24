from numpy import sum as sum_
from numpy import dot
from numpy import newaxis
from numpy import vstack
from numpy import maximum
from numpy import minimum
from numpy import where
from numpy import all as all_
from limix_math.linalg import ddot
from limix_math.linalg import sum2diag

_ZERO_FLOOR = 1e-6


class FixedBaseEP(object):

    def __init__(self, lml_const, A1, A2teta, Q0B1Q0t, beta_nom):
        self._lml_const = lml_const
        self._Q0B1Q0t = Q0B1Q0t
        self._A2teta = A2teta
        self._A1 = A1
        self._u = dot(Q0B1Q0t, A2teta)
        self._beta_nom = beta_nom
        AQA = ddot(A1, ddot(Q0B1Q0t, A1, left=False), left=True)
        self._beta_den = sum2diag(-AQA, A1)

    def lmls(self, ms):
        A1 = self._A1
        A1ms = A1[:, newaxis] * ms
        A2teta = self._A2teta
        Q0B1Q0t = self._Q0B1Q0t

        p5 = A2teta.dot(ms) - dot(self._u, A1ms)
        p6 = (-sum_(A1ms * ms, 0) + sum_(A1ms * dot(Q0B1Q0t, A1ms), 0)) / 2

        return self._lml_const + p5 + p6

    # I assume that Ms represents several n-by-2 M,
    # where for each snp candidate M[:,0] is the offset
    # and M[:,1] is a candidate snp.
    # Ms[:,0] will have the offset and M[:,1:] will have
    # all the candidate snps
    def optimal_betas(self, Ms, ncovariates):
        assert ncovariates == 1

        noms = dot(self._beta_nom, Ms)
        dens = dot(self._beta_den, Ms)

        # dens = dot(A1 * Ms) - dot(Q0B1Q0t, A1M)

        row0 = dot(Ms[:, 0], dens)
        row11 = sum_(Ms[:, 1:] * dens[:, 1:], axis=0)

        obetas_0 = noms[0] * row11[:] - noms[1:] * row0[1:]
        obetas_1 = -noms[0] * row0[1:] + noms[1:] * row0[0]

        obetas = vstack((obetas_0, obetas_1))
        denom = row0[0] * row11[:] - row0[1:]**2
        denom[denom >= 0.] = maximum(denom[denom >= 0.], _ZERO_FLOOR)
        denom[denom < 0.] = minimum(denom[denom < 0.], _ZERO_FLOOR)
        obetas /= denom

        allzero = where(all_(Ms == 0., 0))[0]
        if len(allzero) > 0:
            obetas[0, allzero - 1] = noms[0] / row0[0]
            assert all(obetas[1, allzero - 1] == 0.)

        return obetas
#
#     # p = ncovariates
#     # I assume that Ms represents several n-by-p M,
#     # where for each snp candidate M[:,0:p] are the covariates (without the
#     # candidate snp)
#     # and M[:,p] is candidate snp, M[:,p+1] is another one, etc..
#     def optimal_betas_general(self, Ms, ncovariates):
#         p = ncovariates
#
#         noms = dot(self._opt_bnom, Ms)
#         dens = dot(self._opt_bden, Ms)
#
#         row0 = dot(Ms[:, 0:p].T, dens)
#
#         obetas = []
#         for i in range(p, Ms.shape[1]):
#             nom = np.append(noms[0:p], noms[i])
#             nom = nom[:, np.newaxis]
#
#             row11 = sum_(Ms[:, i] * dens[:, i])
#
#             top = np.hstack((row0[0:p, 0:p], row0[:, i][:, np.newaxis]))
#             bottom = np.append(row0[:, i], row11)
#             bottom = bottom[np.newaxis, :]
#             den = np.vstack((top, bottom))
#
#             try:
#                 s = solve(den, nom)
#             except np.linalg.LinAlgError as e:
#                 print('Warning: %s. Returning zeros for beta.' % str(e))
#                 s = np.zeros(den.shape[1])
#             obetas.append(s)
#
#         return np.hstack(obetas)
