import numpy as np
from numpy import sqrt
import scipy as sp
import scipy.integrate
import scipy.stats


def logistic(x):
    return 1 / (1 + np.exp(-x))


def joint0(x):
    return sp.stats.binom.pmf(1, 3, p=logistic(x)) * sp.stats.norm.pdf(x)


def joint(x):
    return x * sp.stats.binom.pmf(1, 3, p=logistic(x)) * sp.stats.norm.pdf(x)


def joint2(x):
    return x * x * sp.stats.binom.pmf(1, 3, p=logistic(x)) * sp.stats.norm.pdf(x)


Z = sp.integrate.quad(joint0, -10, +10)[0]
mu = sp.integrate.quad(joint, -10, +10)[0] / Z
xx = sp.integrate.quad(joint2, -10, +10)[0] / Z
print("mu: %.30f\n" % mu)
print("var: %.30f\n" % (xx - mu * mu))


def logistic(x):
    return 1 / (1 + np.exp(-x))


def joint0(x):
    return sp.stats.binom.pmf(2, 9, p=logistic(x)) * sp.stats.norm.pdf(x)


def joint(x):
    return x * sp.stats.binom.pmf(2, 9, p=logistic(x)) * sp.stats.norm.pdf(x)


def joint2(x):
    return x * x * sp.stats.binom.pmf(2, 9, p=logistic(x)) * sp.stats.norm.pdf(x)


Z = sp.integrate.quad(joint0, -10, +10)[0]
mu = sp.integrate.quad(joint, -10, +10)[0] / Z
xx = sp.integrate.quad(joint2, -10, +10)[0] / Z
print("mu: %.30f\n" % mu)
print("var: %.30f\n" % (xx - mu * mu))


def logistic(x):
    return 1 / (1 + np.exp(-x))


def joint0(x):
    return sp.stats.binom.pmf(8, 9, p=logistic(x)) * sp.stats.norm.pdf(x)


def joint(x):
    return x * sp.stats.binom.pmf(8, 9, p=logistic(x)) * sp.stats.norm.pdf(x)


def joint2(x):
    return x * x * sp.stats.binom.pmf(8, 9, p=logistic(x)) * sp.stats.norm.pdf(x)

Z = sp.integrate.quad(joint0, -10, +10)[0]
mu = sp.integrate.quad(joint, -10, +10)[0] / Z
xx = sp.integrate.quad(joint2, -10, +10)[0] / Z
print("mu: %.30f\n" % mu)
print("var: %.30f\n" % (xx - mu * mu))


def logistic(x):
    return 1 / (1 + np.exp(-x))


def joint0(x):
    return sp.stats.binom.pmf(8, 9, p=logistic(x)) * sp.stats.norm.pdf(x, loc=+1.2, scale=sqrt(0.3))


def joint(x):
    return x * sp.stats.binom.pmf(8, 9, p=logistic(x)) * sp.stats.norm.pdf(x, loc=+1.2, scale=sqrt(0.3))


def joint2(x):
    return x * x * sp.stats.binom.pmf(8, 9, p=logistic(x)) * sp.stats.norm.pdf(x, loc=+1.2, scale=sqrt(0.3))

Z = sp.integrate.quad(joint0, -10, +10)[0]
mu = sp.integrate.quad(joint, -10, +10)[0] / Z
xx = sp.integrate.quad(joint2, -10, +10)[0] / Z
print("mu: %.30f\n" % mu)
print("var: %.30f\n" % (xx - mu * mu))


def logistic(x):
    return 1 / (1 + np.exp(-x))


def joint0(x):
    return sp.stats.binom.pmf(2, 29, p=logistic(x)) * sp.stats.norm.pdf(x, loc=+1.2, scale=sqrt(2.1))


def joint(x):
    return x * sp.stats.binom.pmf(2, 29, p=logistic(x)) * sp.stats.norm.pdf(x, loc=+1.2, scale=sqrt(2.1))


def joint2(x):
    return x * x * sp.stats.binom.pmf(2, 29, p=logistic(x)) * sp.stats.norm.pdf(x, loc=+1.2, scale=sqrt(2.1))

Z = sp.integrate.quad(joint0, -20, +20)[0]
mu = sp.integrate.quad(joint, -20, +20)[0] / Z
xx = sp.integrate.quad(joint2, -20, +20)[0] / Z
print("mu: %.30f\n" % mu)
print("var: %.30f\n" % (xx - mu * mu))


def logistic(x):
    return 1 / (1 + np.exp(-x))


def joint0(x):
    return sp.stats.binom.pmf(2, 29, p=logistic(x)) * sp.stats.norm.pdf(x, loc=-9.2, scale=sqrt(2.1))


def joint(x):
    return x * sp.stats.binom.pmf(2, 29, p=logistic(x)) * sp.stats.norm.pdf(x, loc=-9.2, scale=sqrt(2.1))


def joint2(x):
    return x * x * sp.stats.binom.pmf(2, 29, p=logistic(x)) * sp.stats.norm.pdf(x, loc=-9.2, scale=sqrt(2.1))

Z = sp.integrate.quad(joint0, -20, +20)[0]
mu = sp.integrate.quad(joint, -20, +20)[0] / Z
xx = sp.integrate.quad(joint2, -20, +20)[0] / Z
print("mu: %.30f\n" % mu)
print("var: %.30f\n" % (xx - mu * mu))


def logistic(x):
    return 1 / (1 + np.exp(-x))


def joint0(x):
    return sp.stats.binom.pmf(0, 5, p=logistic(x)) * sp.stats.norm.pdf(x, loc=-9.2, scale=sqrt(2.1))


def joint(x):
    return x * sp.stats.binom.pmf(0, 5, p=logistic(x)) * sp.stats.norm.pdf(x, loc=-9.2, scale=sqrt(2.1))


def joint2(x):
    return x * x * sp.stats.binom.pmf(0, 5, p=logistic(x)) * sp.stats.norm.pdf(x, loc=-9.2, scale=sqrt(2.1))

Z = sp.integrate.quad(joint0, -20, +20)[0]
mu = sp.integrate.quad(joint, -20, +20)[0] / Z
xx = sp.integrate.quad(joint2, -20, +20)[0] / Z
print("mu: %.30f\n" % mu)
print("var: %.30f\n" % (xx - mu * mu))


def logistic(x):
    return 1 / (1 + np.exp(-x))


def joint0(x):
    return sp.stats.binom.pmf(5, 5, p=logistic(x)) * sp.stats.norm.pdf(x, loc=-1.2, scale=sqrt(2.1))


def joint(x):
    return x * sp.stats.binom.pmf(5, 5, p=logistic(x)) * sp.stats.norm.pdf(x, loc=-1.2, scale=sqrt(2.1))


def joint2(x):
    return x * x * sp.stats.binom.pmf(5, 5, p=logistic(x)) * sp.stats.norm.pdf(x, loc=-1.2, scale=sqrt(2.1))

Z = sp.integrate.quad(joint0, -20, +20)[0]
mu = sp.integrate.quad(joint, -20, +20)[0] / Z
xx = sp.integrate.quad(joint2, -20, +20)[0] / Z
print("mu: %.30f\n" % mu)
print("var: %.30f\n" % (xx - mu * mu))
