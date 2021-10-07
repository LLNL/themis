from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np


class Likelihood(object):

    @staticmethod
    def combine_weights(weights, **kwargs):
        raise NotImplementedError


class Intersection(Likelihood):
    """
    Intersection
    """

    @staticmethod
    def combine_weights(weights, **kwargs):
        """
        :param weights:
        :param kwargs:
        :return:
        """
        if len(weights) == 0:
            return np.ones(1)

        np_combined_weights = np.prod(weights, axis=0)
        # raises ValueError if weight arrays are not the same length

        return np.divide(np_combined_weights, np.sum(np_combined_weights))

    '''
    def getLikelihood(lsFilters):
        #lsFilters is a list of numpy arrays
        if len(lsFilters) == 0:
            return np.ones(1)
        iLength = lsFilters[0].shape[0]
        npCmbndWghts = np.ones(iLength)
        for npFilter in lsFilters:
            if iLength != npFilter.shape[0]:
                print "mismatch in filter lengths"
                return np.ones(1)
            npCmbndWghts *= npFilter
        npCmbndWghts = np.divide(npCmbndWghts, np.sum(npCmbndWghts))
    '''


class StatFiltering(Likelihood):
    """
    Statistical Filtering
    """

    @staticmethod
    def combine_weights(weights, **kwargs):
        """
        :param weights:
        :param kwargs:
        :return:
        """
        if len(weights) == 0:
            return np.ones(1)

        np_combined_weights = np.sum(weights, axis=0)
        # raises ValueError if weight arrays are not the same length

        f_max_weight = np_combined_weights.max()

        if 'chi' in kwargs:
            chi = kwargs['chi']
        else:
            # compute chi
            np_max_weights = np.max(weights, axis=1)
            chi = np.log(1.0 - np.mean(np_max_weights) / f_max_weight)
            chi = max(-1.0 / chi, 1.0)
            # print "chi: {}".format(chi)

        return (np_combined_weights / f_max_weight) ** chi

    '''
    def likelihood(self, lsFltrs, chi = 1.0):
        print type(lsFltrs)
        npMefSmplWts = None
        for fltr in lsFltrs:
            if npMefSmplWts is None:
                npMefSmplWts = fltr
            else:
                npMefSmplWts += fltr
        fMaxWght = npMefSmplWts.max()
        if chi != 1.0:
            chi = np.log(1.0 - np.mean(lsMaxPosteriorWts)/maxWght)
        return (npMefSmplWts/fMaxWght) ** chi
    '''
