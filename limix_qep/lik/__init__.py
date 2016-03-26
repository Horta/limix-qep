from __future__ import absolute_import
import numpy as np

class Bernoulli(object):
    """Represent a Bernoulli phenotype.

    There is no parameter to be set for this object type.
    """
    def assert_outcome(self, y):
        y = np.asarray(y)
        if not np.all(np.logical_or(y == 0.0, y == 1.0)):
            raise Exception("Wrong outcome value(s): %s." % str(y))

    def __eq__(self, other):
        return isinstance(other, Bernoulli)

    def __ne__(self, other):
        return not self.__eq__(other)

class Binomial(object):
    """Represent a Binomial phenotype.

    Each sample has its own number of trials.
    """
    # ntrials can be either a scalar, in which case it is assumed
    # that all samples have the same number of trials, and a
    # array, in which case a different number of trials can be specified
    # for each sample.
    def __init__(self, ntrials, nsamples=None):
        """Construct a Binomial phenotype representation by specifying
        the number of trials of each sample.

        :param ntrials: A single scalar means that all samples will have the
                        same number of trials. Use :class:`~numpy:numpy.ndarray`
                        instead if the number of trials vary across samples.
        :type ntrials: int, numpy.ndarray
        """
        if np.isscalar(ntrials):
            assert nsamples is not None, ("You need to set" +\
                                          " the number of samples.")
            ntrials = np.full(nsamples, ntrials, dtype=float)
        self._ntrials = np.asarray(ntrials, float)
        assert len(self._ntrials.shape) == 1
        if nsamples is not None:
            assert self._ntrials.shape[0] == nsamples

    @property
    def ntrials(self):
        return self._ntrials

    def assert_outcome(self, y):
        ntrials = self.ntrials
        y = np.asarray(y)
        for i in range(self.ntrials.shape[0]):
            if y[i] > ntrials[i] or y[i] < 0 or int(y[i]) != y[i]:
                raise Exception("Wrong outcome value: %s." % str(y[i]))

    def __eq__(self, other):
        if not isinstance(other, Binomial):
            return False
        if self._ntrials.ndim != other.ntrials.ndim:
            return False
        return np.all(self._ntrials == other.ntrials)

    def __ne__(self, other):
        return not self.__eq__(other)
