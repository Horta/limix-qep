import numpy as np
from numpy import dot
from limix_math.linalg import ddot, sum2diag
from limix_math.linalg import solve, cho_solve

_ZERO_FLOOR = 1e-6
def _is_zero(x):
    return abs(x) < 1e-9

class FixedEP(object):

    def __init__(self, lml_fixed_part, A0A0pT_teta, f0, A1, L1, Q, opt_bnom):

        self._lml_fixed_part = lml_fixed_part

        self._A0A0pT_teta = A0A0pT_teta
        self._f0 = f0
        self._f1 = 0.5 * dot(ddot(A1, Q, left=True), cho_solve(L1, ddot(Q.T, A1, left=False)))

        self._A1 = A1
        self._L1 = L1
        self._Q = Q
        self._opt_bnom = opt_bnom

        QtA1 = ddot(Q.T, A1, left=False)

        self._opt_bden = sum2diag(-ddot(A1, dot(Q, cho_solve(L1, QtA1)), left=True), A1)

    def lmls(self, ms):
        A1 = self._A1

        A0A0pT_teta = self._A0A0pT_teta
        f0 = self._f0
        p5 = dot(ms.T, A0A0pT_teta) - dot(ms.T, f0)


        A1m = ms * A1[:, np.newaxis]

        f1 = self._f1
        p6 = - 0.5 * np.sum(ms * A1m, 0) +\
            np.sum(ms * dot(f1, ms), 0)

        return self._lml_fixed_part + p5 + p6


    # I assume that Ms represents several n-by-2 M,
    # where for each snp candidate M[:,0] is the offset
    # and M[:,1] is candidate snp.
    # Ms[:,0] will have the offset and M[:,1:] will have
    # all the candidate snps
    def optimal_betas(self, Ms, ncovariates):
        assert ncovariates == 1
        noms = dot(self._opt_bnom, Ms)
        dens = dot(self._opt_bden, Ms)

        row0 = dot(Ms[:,0], dens)
        row11 = np.sum(Ms[:,1:] * dens[:,1:], axis=0)

        obetas_0 = noms[0] * row11[:] - noms[1:] * row0[1:]
        obetas_1 = -noms[0] * row0[1:] + noms[1:] * row0[0]

        obetas = np.vstack((obetas_0, obetas_1))
        denom = row0[0] * row11[:] - row0[1:]**2
        denom[denom >= 0.] = np.maximum(denom[denom >= 0.], _ZERO_FLOOR)
        denom[denom < 0.] = np.minimum(denom[denom < 0.], _ZERO_FLOOR)
        obetas /= denom

        allzero = np.where(np.all(Ms==0., 0))[0]
        if len(allzero) > 0:
            obetas[0, allzero-1] = noms[0]/row0[0]
            assert np.all(obetas[1, allzero-1] == 0.)

        return obetas

    # p = ncovariates
    # I assume that Ms represents several n-by-p M,
    # where for each snp candidate M[:,0:p] are the covariates (without the
    # candidate snp)
    # and M[:,p] is candidate snp, M[:,p+1] is another one, etc..
    def optimal_betas_general(self, Ms, ncovariates):
        p = ncovariates

        noms = dot(self._opt_bnom, Ms)
        dens = dot(self._opt_bden, Ms)

        row0 = dot(Ms[:,0:p].T, dens)

        obetas = []
        for i in xrange(p, Ms.shape[1]):
            nom = np.append(noms[0:p], noms[i]); nom = nom[:, np.newaxis]


            row11 = np.sum(Ms[:,i] * dens[:,i])

            top = np.hstack((row0[0:p, 0:p], row0[:, i][:, np.newaxis]))
            bottom = np.append(row0[:, i], row11)
            bottom = bottom[np.newaxis, :]
            den = np.vstack((top, bottom))

            try:
                s = solve(den, nom)
            except np.linalg.LinAlgError as e:
                print('Warning: %s. Returning zeros for beta.' % str(e))
                s = np.zeros(den.shape[1])
            obetas.append(s)

        return np.hstack(obetas)
