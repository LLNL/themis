from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

UNIMPLEMENTED = 'Unimplemented method'


class Likelihood(object):

    @staticmethod
    def combine_weights(lsWeights, **kwargs):
        raise UNIMPLEMENTED


class Intersection(Likelihood):
    """
    Intersection
    """

    @staticmethod
    def combine_weights(lsWeights, **kwargs):
        """
        :param lsWeights:
        :param kwargs:
        :return:
        """
        if len(lsWeights) == 0:
            return np.ones(1)

        npCombinedWeights = np.prod(lsWeights, axis=0)
        # raises ValueError if weight arrays are not the same length

        return np.divide(npCombinedWeights, np.sum(npCombinedWeights))

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
    def combine_weights(lsWeights, **kwargs):
        """
        :param lsWeights:
        :param kwargs:
        :return:
        """
        if len(lsWeights) == 0:
            return np.ones(1)

        npCombinedWeights = np.sum(lsWeights, axis=0)
        # raises ValueError if weight arrays are not the same length

        fMaxWght = npCombinedWeights.max()

        if 'chi' in kwargs:
            chi = kwargs['chi']
        else:
            # compute chi
            npMaxWeights = np.max(lsWeights, axis=1)
            chi = np.log(1.0 - np.mean(npMaxWeights) / fMaxWght)
            chi = max(-1.0 / chi, 1.0)
            # print "chi: {}".format(chi)

        return (npCombinedWeights / fMaxWght) ** chi

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
