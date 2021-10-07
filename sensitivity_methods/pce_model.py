from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import chain, combinations_with_replacement

import numpy as np
from scipy.special import eval_legendre
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def _legendre(x, n):
    return eval_legendre(n, x)


def _var_multiplier(interaction_list):
    return np.prod([1. / (2 * el + 1) for el in interaction_list])


def _make_variance_terms(coefficients, interactions):
    return [_var_multiplier(interaction) * coefficient ** 2 for coefficient, interaction in
            zip(coefficients, interactions)]


class PolynomialBasis(TransformerMixin):
    """Transforms to Legendre Basis"""

    def __init__(self, num_degrees, basis):
        self.num_degrees = num_degrees
        self._basis = basis
        self.interactions = None

    def fit(self, x, y=None):
        dim = x.shape[1]

        # Enumerate all interactions between variables for the given number of degrees
        combinations = chain.from_iterable(combinations_with_replacement(range(dim),
                                                                         i + 1) for i in range(self.num_degrees))
        self.interactions = np.array([[k.count(d) for d in range(dim)] for k in combinations])

        return self

    def transform(self, x):
        # Apply basis function and multiply dimensions based on interaction level
        return np.prod(self._basis(x[..., np.newaxis], self.interactions.T[np.newaxis, ...]), axis=1)


class RangeScaler(TransformerMixin):
    """Scales to a new range from a given range"""

    def __init__(self, feature_range_from=None, feature_range_to=None):
        self.feature_range_from = feature_range_from
        self.feature_range_to = feature_range_to

    @staticmethod
    def _scale(val, from_lo, from_hi, to_lo, to_hi):
        return (((val - from_lo) * (to_hi - to_lo)) / (from_hi - from_lo)) + to_lo

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        vals = self._scale(x,
                           self.feature_range_from[:, 0],
                           self.feature_range_from[:, 1],
                           self.feature_range_to[:, 0],
                           self.feature_range_to[:, 1])

        return vals


class PolynomialChaosExpansionModel(object):
    """Linear regression on orthogonal polynomial basis"""

    def __init__(self, num_degrees=1, ranges=None):

        if ranges is None:
            raise Exception()

        np_ranges_from = np.array(ranges)

        ranges_shape = np_ranges_from.shape

        if len(ranges_shape) != 2:
            raise Exception()

        if ranges_shape[1] != 2:
            raise Exception

        # Pipeline chains together transformation and regression
        self.pipeline = Pipeline([('Range_Scaling', RangeScaler(feature_range_from=np_ranges_from,
                                                                feature_range_to=np.array([[-1, 1]]))),
                                  ('Legendre_Basis', PolynomialBasis(num_degrees=num_degrees, basis=_legendre)),
                                  ('Linear_Model', LinearRegression(fit_intercept=True, copy_X=False))])

        self.num_degrees = num_degrees
        self.coefficients = None
        self.intercept = None
        self.interactions = None
        self.variance_terms = None

    def fit(self, x, y):
        self.pipeline.fit(x, y)

        self.coefficients = self.pipeline.named_steps['Linear_Model'].coef_
        self.intercept = self.pipeline.named_steps['Linear_Model'].intercept_
        self.interactions = self.pipeline.named_steps['Legendre_Basis'].interactions
        self.variance_terms = np.array([_make_variance_terms(coef_list,
                                                             self.interactions) for coef_list in self.coefficients])

        return self

    def predict(self, x):
        return self.pipeline.predict(x)

    def make_contributions(self, degree):
        non_zero = self.interactions > 0
        range_selector = non_zero.sum(axis=1) <= degree
        same_term = np.prod(~(non_zero[..., np.newaxis] ^ non_zero.T[np.newaxis, ...]), axis=1)
        var_term_sum = np.dot(self.variance_terms, same_term)
        first_selection = ~np.sum((self.interactions == 1) ^ non_zero, axis=1, dtype='bool')
        return var_term_sum[:, first_selection & range_selector]


if __name__ == "__main__":
    pass
