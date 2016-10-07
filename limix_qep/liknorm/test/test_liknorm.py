from __future__ import division
from limix_qep.liknorm import LikNormMoments

from numpy import array
from numpy import empty
from numpy import set_printoptions
from numpy.testing import assert_allclose


def test_liknorm():
    set_printoptions(precision=16)

    ln = LikNormMoments(350)

    N = array([3, 7, 1, 98], float)
    K = array([2, 0, 1, 66], float)

    tau = array([0.1, 0.9, 1.3, 1.1])
    eta = array([-1.2, 0.0, +0.9, -0.1]) * tau

    log_zeroth = empty(4)
    mean = empty(4)
    variance = empty(4)

    ln.bimomial(K, N, eta, tau, log_zeroth, mean, variance)

    # assert_allclose([-2.3408345658185441, -2.1095165980375743,
    #                  1.6817388883718341, -62.8505790200703558], log_zeroth)
    # assert_allclose([0.0354041395808048, -1.3853743380699952,
    #                  1.7654307169642343, 0.696967590209962], mean)
    # assert_allclose([0.0931700379328726, 0.454999155845663,
    #                  1.113326179275818, 0.0445446693427618], variance)
    #
    # ln.destroy()
    #
    # ln = PoissonMoments(350)
    #
    # k = array([2, 0, 1, 66], float)
    #
    # tau = array([0.1, 0.9, 1.3, 1.1])
    # eta = array([-1.2, 0.0, +0.9, -0.1]) * tau
    #
    # ln.compute(k, eta, tau, log_zeroth, mean, variance)
    #
    # assert_allclose([-1.2495777244111932e+00, -1.0082075657296130e-01,
    #                  -1.9302711085886148e-01, 2.0104292304239095e+02],
    #                 log_zeroth)
    # assert_allclose([0.0752434403858402, -0.6283056227745004,
    #                  0.4213176599565073, 4.1214915483196641], mean)
    # assert_allclose([0.0899087536848139, 0.5727406406064515,
    #                  0.4162894959504529, 0.0159825976367891], variance)

    ln.destroy()
