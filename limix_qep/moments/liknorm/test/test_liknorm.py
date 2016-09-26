from __future__ import division
from limix_qep.moments.liknorm import LikNormMoments

from numpy import array
from numpy import empty
from numpy import set_printoptions
from numpy.testing import assert_allclose


def test_liknorm():
    set_printoptions(precision=16)

    ln = LikNormMoments(350, "binomial")

    N = [3, 7, 1, 98]
    K = array([2, 0, 1, 66], float)
    y = K / N
    aphi = 1 / array(N, float)

    tau = array([0.1, 0.9, 1.3, 1.1])
    eta = array([-1.2, 0.0, +0.9, -0.1]) * tau

    mean = empty(4)
    variance = empty(4)

    ln.compute(y, aphi, eta, tau, mean, variance)

    assert_allclose([0.0354041395808048, -1.3853743380699952,
                     1.7654307169642343, 0.696967590209962], mean)
    assert_allclose([0.0931700379328726, 0.454999155845663,
                     1.113326179275818, 0.0445446693427618], variance)

    ln.destroy()

    ln = LikNormMoments(350, "poisson")

    y = array([2, 0, 1, 66], float)
    aphi = array([1, 1, 1, 1], float)

    tau = array([0.1, 0.9, 1.3, 1.1])
    eta = array([-1.2, 0.0, +0.9, -0.1]) * tau

    mean = empty(4)
    variance = empty(4)

    ln.compute(y, aphi, eta, tau, mean, variance)

    assert_allclose([0.0752434403858402, -0.6283056227745004,
                     0.4213176599565073, 4.1214915483196641], mean)
    assert_allclose([0.0899087536848139, 0.5727406406064515,
                     0.4162894959504529, 0.0159825976367891], variance)

    ln.destroy()
