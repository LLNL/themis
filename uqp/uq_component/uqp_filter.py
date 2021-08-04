# Copyright (c) 2004 - 2014, Lawrence Livermore National Security, LLC.
# All rights reserved.
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

UNIMPLEMENTED = 'Unimplemented method'


class Filter(object):
    def __init__(self):
        object.__init__(self)

    @staticmethod
    def get_type():
        raise UNIMPLEMENTED

    @staticmethod
    def get_weights(output_values, observed_value, observed_std, **kwargs):
        raise UNIMPLEMENTED

    @staticmethod
    def _saveWts(npWts, ls01, **kwArgs):
        """
        :param npWts:
        :param ls01:
        :param kwArgs:
        :return:
        """
        if 'no_01' in kwArgs:
            lsWts = ['%g' % w for w in npWts]
        else:
            lsWts = ['%g[%d]' % (w, ls01[i]) for i, w in enumerate(npWts)]
        nMaxWtLen = 0
        for sWt in lsWts:
            if len(sWt) > nMaxWtLen:
                nMaxWtLen = len(sWt)
        fd = kwArgs['save_wts_foreach_op']
        if 'cat_val' in kwArgs:
            fd.write('# Max weight string size for categorical value: %d : %d\n' % (kwArgs['cat_val'], nMaxWtLen))
            fd.write(' '.join(lsWts))
            fd.write('\n')
        else:
            fd.write('# Max weight string size: %d\n' % nMaxWtLen)
            fd.write('\n'.join(lsWts))
            fd.write('\n')


class GaussianFilter(Filter):
    """
    Gaussian Likelihood
    """

    # def __init__(self):
    #    Filter.__init__(self)

    @staticmethod
    def get_type():
        return 'gaussian'

    @staticmethod
    def get_weights(output_values, observed_value, observed_std, **kwargs):
        """
        :param output_values:
        :param observed_value:
        :param observed_std:
        :param kwargs:
        :return:
        """
        if observed_std < 0:
            raise ValueError("Standard deviation cannot be less than zero."
                             "(experiment_stddev: {})".format(observed_std))

        if 'sigma_cut' in kwargs:
            sigma_cut = kwargs['sigma_cut']
            if sigma_cut < 0:
                raise ValueError("Sigma cutoff cannot be less than zero."
                                 "(sigma_cut: {})".format(sigma_cut))
        else:
            sigma_cut = 1.0

        npWeights = np.exp(-0.5 * (((output_values - observed_value) / observed_std) ** 2.0))

        return np.where(np.logical_and(output_values >= (observed_value - (sigma_cut * observed_std)),
                                       output_values <= (observed_value + (sigma_cut * observed_std))),
                        npWeights, 0.0)


class LogGaussianFilter(Filter):

    @staticmethod
    def get_type():
        return 'log-gaussian'

    @staticmethod
    def get_weights(output_values, observed_value, observed_std, **kwargs):
        """
        Log gaussian normal pdf

        optimized for speed

        :param output_values:
        :param observed_value:
        :param observed_std:
        :param kwargs:
        :return:
        """

        return -0.5 * (((output_values - observed_value) / observed_std) ** 2.0)


class StudentTFilter(Filter):
    """
    Student T Filter
    Based on Student's T distribution. Has fatter tails than the Gaussian Filter.
    Approaches Gaussian Filter as df -> inf
    """

    @staticmethod
    def get_type():
        return 'student-t'

    @staticmethod
    def get_weights(output_values, observed_value, observed_std, **kwargs):
        """
        :param output_values:
        :param observed_value:
        :param observed_std:
        :param kwargs:
        :return:
        """
        if observed_std < 0:
            raise ValueError("Standard deviation cannot be less than zero."
                             "(experiment_stddev: {})".format(observed_std))

        if 'df' in kwargs:
            df = float(kwargs['df'])
            if df <= 0:
                raise ValueError("Degrees of freedom must be positive."
                                 "(df: {})".format(df))
        else:
            df = 1.0

        if 'sigma_cut' in kwargs:
            sigma_cut = kwargs['sigma_cut']
            if sigma_cut < 0:
                raise ValueError("Sigma cutoff cannot be less than zero."
                                 "(sigma_cut: {})".format(sigma_cut))
        else:
            sigma_cut = 1.0

        npWeights = np.power(1.0 + (((output_values - observed_value) / observed_std) ** 2.0) / df,
                             -(df + 1.0) / 2.0)

        return np.where(np.logical_and(output_values >= (observed_value - (sigma_cut * observed_std)),
                                       output_values <= (observed_value + (sigma_cut * observed_std))),
                        npWeights, 0.0)


class TophatFilter(Filter):

    @staticmethod
    def get_type():
        return 'top-hat'

    @staticmethod
    def get_weights(output_values, observed_value, observed_std, **kwargs):
        """
        :param output_values:
        :param observed_value:
        :param observed_std:
        :param kwargs:
        :return:
        """

        if 'sigma_cut' in kwargs:
            sigma_cut = kwargs['sigma_cut']
            if sigma_cut < 0:
                raise ValueError("Sigma cutoff cannot be less than zero."
                                 "(sigma_cut: {})".format(sigma_cut))
        else:
            sigma_cut = 1.0

        if observed_std < 0:
            raise ValueError("Standard deviation cannot be less than zero."
                             "(experiment_stddev: {})".format(observed_std))

        return np.where(np.logical_and(output_values >= (observed_value - sigma_cut),
                                       output_values <= (observed_value + sigma_cut)),
                        1.0, 0.0)


class LogTophatFilter(Filter):

    @staticmethod
    def get_type():
        return 'log-top-hat'

    @staticmethod
    def get_weights(output_values, observed_value, observed_std, **kwargs):

        return np.where(np.logical_and(output_values >= (observed_value - observed_std),
                                       output_values <= (observed_value + observed_std)),
                        0.0, float('-inf'))


class GaussianMixtureFilter(Filter):

    @staticmethod
    def get_weights(output_values, observed_value, observed_std, **kwargs):
        """
        :type array-like [[]]
        :param output_values:

        :type array-like [[]]
        :param observed_value:

        :type array-like [[]]
        :param observed_std:

        :param kwargs:

        :type numpy array
        :return:

        The Gaussian Mixture Filter both filters and combines weights all at once.
        Inputs are expected to be multidimensional and in list (i.e. 2d arrays)
        """

        npXData = None

        for obs, std in zip(observed_value, observed_std):

            npCov = np.zeros((len(std), len(std)))

            for i in range(len(std)):
                npCov[i, i] = std[i] ** 2.0

            np.random.seed(20150515)
            dx = np.random.multivariate_normal(obs, npCov, 30).T
            if npXData is None:
                npXData = np.c_[dx]
            else:
                npXData = np.c_[npXData, dx]

        from sklearn.mixture import GaussianMixture

        # Find optimal number of components by minimizing Bayesian Information Criteria
        lsBICScores = []
        for i in range(len(observed_value)):
            gmm = GaussianMixture(n_components=i + 1, covariance_type='full')
            gmm.fit(npXData.T)
            lsBICScores += [gmm.bic(npXData.T)]
        minBic = lsBICScores.index(min(lsBICScores))

        # Fit model
        gmm = GaussianMixture(n_components=minBic + 1, covariance_type='full', n_init=10)
        gmm.fit(npXData.T)

        # Weight is the probability of point being in the model
        npLogProb = []
        for output_value in output_values:
            npLogProb += [gmm.score(np.array(output_value).reshape(1, -1))]

        return np.exp(npLogProb - max(npLogProb))
