import numpy as np
from numpy import dot
from scipy.linalg.lapack import get_lapack_funcs
from limix_math.linalg import ddot
from limix_math.linalg import solve, lu_solve

_ZERO_FLOOR = 1e-6
def _is_zero(x):
    return abs(x) < 1e-9

class FixedEP2(object):

    def __init__(self, lml_fixed_part, opt_bnom, ttau, teta, K, LUf):
        self._lml_fixed_part = lml_fixed_part

        self._teta = teta
        self._f0 = ttau * lu_solve(LUf, dot(K, teta))
        self._opt_bnom = opt_bnom

        getri = get_lapack_funcs('getri', LUf)
        LU_1 = getri(LUf[0], LUf[1])[0]
        ddot(ttau, LU_1, left=True)
        self._opt_bden = ddot(ttau, LU_1, left=True)

    def lmls(self, ms):
        p5 = dot(ms.T, self._teta)
        p5 -= dot(ms.T, self._f0)
        p6 = -0.5 * np.sum(ms * dot(self._opt_bden, ms), 0)
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
                print 'Warning: %s. Returning zeros for beta.' % str(e)
                s = np.zeros(den.shape[1])
            obetas.append(s)

        return np.hstack(obetas)
