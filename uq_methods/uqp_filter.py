# Copyright (c) 2004 - 2014, Lawrence Livermore National Security, LLC.
# All rights reserved.
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from sklearn.mixture import GaussianMixture


class Filter(object):
    def __init__(self):
        object.__init__(self)

    @staticmethod
    def get_type():
        raise NotImplementedError

    @staticmethod
    def get_weights(output_values, observed_value, observed_std, **kwargs):
        raise NotImplementedError


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

        np_weights = np.exp(-0.5 * (((output_values - observed_value) / observed_std) ** 2.0))

        return np.where(np.logical_and(output_values >= (observed_value - (sigma_cut * observed_std)),
                                       output_values <= (observed_value + (sigma_cut * observed_std))),
                        np_weights, 0.0)


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

        np_weights = np.power(1.0 + (((output_values - observed_value) / observed_std) ** 2.0) / df, -(df + 1.0) / 2.0)

        return np.where(np.logical_and(output_values >= (observed_value - (sigma_cut * observed_std)),
                                       output_values <= (observed_value + (sigma_cut * observed_std))),
                        np_weights, 0.0)


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
    def get_type():
        return 'gaussian-mixture'

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

        np_x_data = None

        for obs, std in zip(observed_value, observed_std):

            np_cov = np.diag(std) ** 2

            np.random.seed(20150515)
            dx = np.random.multivariate_normal(obs, np_cov, 30).T
            if np_x_data is None:
                np_x_data = np.c_[dx]
            else:
                np_x_data = np.c_[np_x_data, dx]

        # Find optimal number of components by minimizing Bayesian Information Criteria
        ls_bic_scores = []
        for i in range(len(observed_value)):
            gmm = GaussianMixture(n_components=i + 1, covariance_type='full')
            gmm.fit(np_x_data.T)
            ls_bic_scores += [gmm.bic(np_x_data.T)]
        num_min_bic = ls_bic_scores.index(min(ls_bic_scores))

        # Fit model
        gmm = GaussianMixture(n_components=num_min_bic + 1, covariance_type='full', n_init=10)
        gmm.fit(np_x_data.T)

        # Weight is the probability of point being in the model
        np_log_prob = np.array([gmm.score(val[np.newaxis, :]) for val in output_values])

        return np.exp(np_log_prob - np_log_prob.max())
