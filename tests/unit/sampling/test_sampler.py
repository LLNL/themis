from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import unittest
import os

import numpy as np

import sampling_methods.sampler


class TestSampler(unittest.TestCase):

    def test_LatinHyperCubeSampler_valid(self):
        ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
        ls_test_box_mixed = [[0.0, 25.0], []]
        ls_test_values_mixed = [[], [1, 2, 3]]
        ls_test_values = [['a', 'b', 'c'], [1, 2, 3]]

        np_actual_values_std = sampling_methods.sampler.LatinHyperCubeSampler.sample_points(box=ls_test_box,
                                                                                            num_points=25,
                                                                                            seed=2018)

        ls_expected_values_std = [[4.35852039, -5.08143228, -10.43091164],
                                  [16.49596893, -20.47257130, -21.89988126],
                                  [23.06817236, -6.63575213, -14.80702562],
                                  [3.48292099, -11.07181371, 10.72657464],
                                  [8.71872672, -24.50730858, 13.15911348],
                                  [17.23687915, -7.76359297, 22.14749438],
                                  [7.23967253, -3.46858462, 20.81440090],
                                  [5.69152361, -16.83474467, -24.13821534],
                                  [14.97822306, -23.62861300, -11.40816531],
                                  [0.58635238, -4.25774819, -4.02309835],
                                  [1.29754221, -12.76295943, 1.37451344],
                                  [10.68748533, -1.60803635, 4.07240071],
                                  [13.11060267, -14.94061026, 6.01096442],
                                  [19.49896516, -15.89723107, -15.42845101],
                                  [12.89555267, -18.44274988, 16.13493786],
                                  [22.25150919, -13.18069769, 12.24502754],
                                  [18.40577694, -8.51018962, 7.50005944],
                                  [15.58535840, -2.09769303, 17.10681348],
                                  [11.63920998, -22.50995564, 23.19247850],
                                  [24.29783886, -9.99696411, -5.97836841],
                                  [20.38615112, -19.15633785, -8.05345522],
                                  [21.27186961, -17.45631128, -2.96984493],
                                  [9.94115246, -0.86130853, 0.31071369],
                                  [2.36364119, -10.91427125, -20.96794990],
                                  [6.29954539, -21.59838791, -17.91665014]]

        np.testing.assert_array_almost_equal(np_actual_values_std, ls_expected_values_std)

        np_actual_values_geo = sampling_methods.sampler.LatinHyperCubeSampler.sample_points(box=ls_test_box,
                                                                                            num_points=25,
                                                                                            geo_degree=1.2,
                                                                                            seed=2018)

        ls_expected_values_geo = [[2.50362419, -3.03469651, -15.08930384],
                                  [18.77359544, -22.37383044, -23.46824307],
                                  [24.03243512, -4.34533614, -18.90632539],
                                  [1.90600791, -9.48422308, 15.38388974],
                                  [6.46417686, -24.76583133, 17.57739272],
                                  [19.56781870, -5.43265170, 23.59905594],
                                  [4.91063990, -1.89665036, 22.88385271],
                                  [3.52669501, -19.14878463, -24.59040699],
                                  [16.96351106, -24.32851333, -16.02219007],
                                  [0.27868428, -2.43051254, -7.64991430],
                                  [0.63247481, -14.14349642, 3.77776085],
                                  [8.94738347, -0.79650743, 7.71737515],
                                  [14.27229556, -16.91204514, 10.23112302],
                                  [21.62692165, -18.09780767, -19.37279380],
                                  [14.97220416, -20.72503146, 19.88536001],
                                  [23.55689222, -14.37890465, 16.77233628],
                                  [20.69188967, -6.23322505, 12.00244273],
                                  [17.71388978, -1.06094719, 20.58269349],
                                  [10.34718887, -23.70865679, 24.14091279],
                                  [24.66627359, -8.00295416, -10.19099706],
                                  [22.31113137, -21.35061527, -12.61532047],
                                  [22.93394050, -19.78645144, -6.20413647],
                                  [7.93424954, -0.40936672, 1.94196056],
                                  [1.21711691, -9.25769661, -22.97401955],
                                  [4.04396729, -23.14706354, -21.11128515]]

        np.testing.assert_array_almost_equal(np_actual_values_geo, ls_expected_values_geo)

        np_actual_values_mixed = sampling_methods.sampler.LatinHyperCubeSampler.sample_points(box=ls_test_box_mixed,
                                                                                              values=ls_test_values_mixed,
                                                                                              num_points=5,
                                                                                              seed=2018)

        ls_expected_values_mixed = [[4.53504667, 2],
                                    [6.53199449, 3],
                                    [17.23204436, 2],
                                    [22.94992696, 1],
                                    [14.18555550, 1]]

        np.testing.assert_array_almost_equal(np_actual_values_mixed, ls_expected_values_mixed)

        np_actual_values_discrete = sampling_methods.sampler.LatinHyperCubeSampler.sample_points(values=ls_test_values,
                                                                                                 num_points=5,
                                                                                                 seed=2018)

        np_expected_values_discrete = np.array([['a', 2],
                                                ['a', 3],
                                                ['c', 2],
                                                ['c', 1],
                                                ['b', 1]], dtype='O')

        np.testing.assert_array_equal(np_actual_values_discrete, np_expected_values_discrete)

    def test_LatinHyperCubeSampler_invalid(self):
        # nPts not given
        self.assertRaises(TypeError, sampling_methods.sampler.LatinHyperCubeSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]])

        # neither box nor values given
        self.assertRaises(TypeError, sampling_methods.sampler.LatinHyperCubeSampler.sample_points,
                          num_points=10)

        # nPts nor box not given
        self.assertRaises(TypeError, sampling_methods.sampler.LatinHyperCubeSampler.sample_points)

        # not enough dimensions in box
        self.assertRaises(TypeError, sampling_methods.sampler.LatinHyperCubeSampler.sample_points,
                          box=[0.0, 1.0],
                          num_points=10)

        # too many dimensions in nPts
        self.assertRaises(TypeError, sampling_methods.sampler.LatinHyperCubeSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_points=[1, 2])

        # nPts type str
        self.assertRaises(ValueError, sampling_methods.sampler.LatinHyperCubeSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_points='f')

        # box not list
        self.assertRaises(TypeError, sampling_methods.sampler.LatinHyperCubeSampler.sample_points,
                          box=1.0,
                          num_points=10)

        # too many dimensions in box
        self.assertRaises(TypeError, sampling_methods.sampler.LatinHyperCubeSampler.sample_points,
                          box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                          num_points=10)

        # geo_degree too low
        self.assertRaises(ValueError, sampling_methods.sampler.LatinHyperCubeSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_points=10,
                          geo_degree=-1.0)

        # geo_degree too high
        self.assertRaises(ValueError, sampling_methods.sampler.LatinHyperCubeSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_points=10,
                          geo_degree=3.0)

    def test_MonteCarloSampler_valid(self):
        ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
        np_actual_values = sampling_methods.sampler.MonteCarloSampler.sample_points(box=ls_test_box,
                                                                                    num_points=25,
                                                                                    seed=2018)

        ls_expected_values = [[22.05873279, -20.98472622, 18.71205141],
                              [2.60819345, -22.75118388, -0.93975390],
                              [22.67523334, -0.72944103, -18.13007310],
                              [7.65997247, -4.58556066, 9.51107713],
                              [11.16022181, -10.71585676, 0.10592748],
                              [14.74963479, -16.35367116, -21.27444596],
                              [20.92777749, -14.90639990, 1.17561433],
                              [17.44501518, -21.56542405, 20.92838613],
                              [20.07007094, -2.47663773, 1.37143481],
                              [2.68037697, -1.65159673, -6.78760662],
                              [18.92731315, -23.81557158, 21.40931473],
                              [24.99177532, -8.21232790, -0.36542876],
                              [18.14827494, -24.12920345, -13.17964834],
                              [3.53620599, -18.68271599, 1.57076904],
                              [8.91801491, -11.07187365, -16.73723360],
                              [23.56760275, -11.85441308, -6.43065020],
                              [15.25404717, -16.17580535, 12.11259030],
                              [5.68943682, -22.67542577, -13.14797138],
                              [16.71830928, -17.38727558, -5.40181750],
                              [17.32261387, -3.43925346, -22.03051290],
                              [10.42156266, -7.07658654, -19.86155332],
                              [4.29523899, -0.89821269, 2.86250621],
                              [24.42226265, -11.50745340, 15.96511553],
                              [8.25560358, -1.23650456, -0.50948082],
                              [15.72610376, -8.30046097, 20.11534825]]

        np.testing.assert_array_almost_equal(ls_expected_values, np_actual_values)

    def test_MonteCarloSampler_invalid(self):
        # nPts not given
        self.assertRaises(TypeError, sampling_methods.sampler.MonteCarloSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]])

        # box not given
        self.assertRaises(TypeError, sampling_methods.sampler.MonteCarloSampler.sample_points,
                          num_points=10)

        # nPts nor box not given
        self.assertRaises(TypeError, sampling_methods.sampler.MonteCarloSampler.sample_points)

        # not enough dimensions in box
        self.assertRaises(TypeError, sampling_methods.sampler.MonteCarloSampler.sample_points,
                          box=[0.0, 1.0],
                          num_points=10)

        # too many dimensions in nPts
        self.assertRaises(TypeError, sampling_methods.sampler.MonteCarloSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_points=[1, 2])

        # nPts type str
        self.assertRaises(ValueError, sampling_methods.sampler.MonteCarloSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_points='f')

        # box not list
        self.assertRaises(TypeError, sampling_methods.sampler.MonteCarloSampler.sample_points,
                          box=1.0,
                          num_points=10)

        # too many dimensions in box
        self.assertRaises(ValueError, sampling_methods.sampler.MonteCarloSampler.sample_points,
                          box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                          num_points=10)

    def test_QuasiRandomNumberSampler_valid(self):
        ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
        np_actual_values_sobol = sampling_methods.sampler.QuasiRandomNumberSampler.sample_points(box=ls_test_box,
                                                                                                 num_points=3,
                                                                                                 technique='Sobol')

        ls_expected_values_sobol = [[12.50, -12.50, 0.0],
                                    [18.75, -18.75, 12.5],
                                    [6.25, -6.25, -12.5]]

        np.testing.assert_array_equal(np_actual_values_sobol, ls_expected_values_sobol)

        np_actual_values_sobol = sampling_methods.sampler.QuasiRandomNumberSampler.sample_points(box=ls_test_box,
                                                                                                 num_points=3,
                                                                                                 technique='Halton')

        ls_expected_values_halton = [[6.250, -8.33333333, -5.],
                                     [18.750, -22.22222222, 5.],
                                     [3.125, -13.88888889, 15.]]

        np.testing.assert_array_almost_equal(np_actual_values_sobol, ls_expected_values_halton)

    def test_QuasiRandomNumberSampler_invalid(self):
        # nPts not given
        self.assertRaises(TypeError, sampling_methods.sampler.QuasiRandomNumberSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]])

        # box not given
        self.assertRaises(TypeError, sampling_methods.sampler.QuasiRandomNumberSampler.sample_points,
                          num_points=10)

        # nPts nor box not given
        self.assertRaises(TypeError, sampling_methods.sampler.QuasiRandomNumberSampler.sample_points)

        # not enough dimensions in box
        self.assertRaises(TypeError, sampling_methods.sampler.QuasiRandomNumberSampler.sample_points,
                          box=[0.0, 1.0],
                          num_points=10)

        # too many dimensions in nPts
        self.assertRaises(TypeError, sampling_methods.sampler.QuasiRandomNumberSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_points=[1, 2])

        # nPts type str
        self.assertRaises(ValueError, sampling_methods.sampler.QuasiRandomNumberSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_points='f')

        # box not list
        self.assertRaises(TypeError, sampling_methods.sampler.QuasiRandomNumberSampler.sample_points,
                          box=1.0,
                          num_points=10)

        # too many dimensions in box
        self.assertRaises(TypeError, sampling_methods.sampler.QuasiRandomNumberSampler.sample_points,
                          box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                          num_points=10)

        # invalid sequence type
        self.assertRaises(ValueError, sampling_methods.sampler.QuasiRandomNumberSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_points=10,
                          technique='foobar')

        # at_most type str
        self.assertRaises(ValueError, sampling_methods.sampler.QuasiRandomNumberSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_points=10,
                          technique='halton',
                          at_most='b')

    def test_CenteredSampler_valid(self):
        ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
        np_actual_values = sampling_methods.sampler.CenteredSampler.sample_points(box=ls_test_box,
                                                                                  num_divisions=[3, 5, 6],
                                                                                  default=[1, 2, 3])

        np_expected_values = [[0., 2., 3.],
                              [12.5, 2., 3.],
                              [25., 2., 3.],
                              [1., -25., 3.],
                              [1., -18.75, 3.],
                              [1., -12.5, 3.],
                              [1., -6.25, 3.],
                              [1., 0., 3.],
                              [1., 2., -25.],
                              [1., 2., -15.],
                              [1., 2., -5.],
                              [1., 2., 5.],
                              [1., 2., 15.],
                              [1., 2., 25.]]

        np.testing.assert_array_equal(np_actual_values, np_expected_values)

    def test_CenteredSampler_invalid(self):
        # nDiv not given
        self.assertRaises(TypeError, sampling_methods.sampler.CenteredSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          default=[1, 2])

        # box not given
        self.assertRaises(TypeError, sampling_methods.sampler.CenteredSampler.sample_points,
                          num_divisions=10,
                          default=[1, 2])

        # default not given
        self.assertRaises(ValueError, sampling_methods.sampler.CenteredSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_divisions=10)

        # not enough dimensions in box
        self.assertRaises(TypeError, sampling_methods.sampler.CenteredSampler.sample_points,
                          box=[0.0, 1.0],
                          num_divisions=10,
                          default=[1, 2])

        # nDiv type str
        self.assertRaises(TypeError, sampling_methods.sampler.CenteredSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_divisions='ff',
                          default=[1, 2])

        # box not list
        self.assertRaises(TypeError, sampling_methods.sampler.CenteredSampler.sample_points,
                          box=1.0,
                          num_divisions=10,
                          default=[1, 2])

        # too many dimensions in box
        self.assertRaises(TypeError, sampling_methods.sampler.CenteredSampler.sample_points,
                          box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                          num_divisions=10,
                          dim_indices=[1],
                          default=[1, 2])

        # default length less than number of dimensions in box
        self.assertRaises(ValueError, sampling_methods.sampler.CenteredSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_divisions=10,
                          default=[2])

        # default length more than number of dimensions in box
        self.assertRaises(ValueError, sampling_methods.sampler.CenteredSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_divisions=10,
                          default=[1, 2, 3])

        # default type str
        self.assertRaises(ValueError, sampling_methods.sampler.CenteredSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_divisions=10,
                          default="ab")

        # default type int
        self.assertRaises(TypeError, sampling_methods.sampler.CenteredSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_divisions=10,
                          default=2)

    def test_OneAtATimeSampler_valid(self):
        ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
        np_actual_values = sampling_methods.sampler.OneAtATimeSampler.sample_points(box=ls_test_box,
                                                                                    default=[1, 2, 3],
                                                                                    do_oat=True,
                                                                                    use_low=True,
                                                                                    use_high=True,
                                                                                    use_default=True)

        ls_expected_values = [[0., -25., -25.],
                              [25., 0., 25.],
                              [1., 2., 3.],
                              [0., 2., 3.],
                              [25., 2., 3.],
                              [1., -25., 3.],
                              [1., 0., 3.],
                              [1., 2., -25.],
                              [1., 2., 25.]]

        np.testing.assert_array_equal(np_actual_values, ls_expected_values)

    def test_OneAtATimeSampler_invalid(self):
        # box not given
        self.assertRaises(TypeError, sampling_methods.sampler.OneAtATimeSampler.sample_points,
                          default=[1, 2])

        # not enough dimensions in box
        self.assertRaises(TypeError, sampling_methods.sampler.OneAtATimeSampler.sample_points,
                          box=[0.0, 1.0],
                          default=[1, 2],
                          do_oat=True)

        # box not list
        self.assertRaises(TypeError, sampling_methods.sampler.OneAtATimeSampler.sample_points,
                          box=1.0,
                          default=[1, 2],
                          do_oat=True)

        # too many dimensions in box
        self.assertRaises(TypeError, sampling_methods.sampler.OneAtATimeSampler.sample_points,
                          box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                          default=[1, 2],
                          do_oat=True)

        # default length less than number of dimensions in box
        self.assertRaises(IndexError, sampling_methods.sampler.OneAtATimeSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          default=[2],
                          use_default=True)

        # default type int
        self.assertRaises(TypeError, sampling_methods.sampler.OneAtATimeSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          default=2,
                          use_default=True)

    def test_DefaultValueSampler_valid(self):
        ls_test_default = [-1, 0, 2, np.pi]
        np_actual_values = sampling_methods.sampler.DefaultValueSampler.sample_points(num_points=2,
                                                                                      default=ls_test_default)

        ls_expected_values = [[-1., 0., 2., np.pi],
                              [-1., 0., 2., np.pi]]

        np.testing.assert_array_equal(np_actual_values, ls_expected_values)

    def test_DefaultValueSampler_invalid(self):
        # nPts not given
        self.assertRaises(TypeError, sampling_methods.sampler.DefaultValueSampler.sample_points,
                          default=[1, 2, 3])

        # default not given
        self.assertRaises(TypeError, sampling_methods.sampler.DefaultValueSampler.sample_points,
                          num_points=5)

        # default and box not same dimension
        self.assertRaises(ValueError, sampling_methods.sampler.DefaultValueSampler.sample_points,
                          num_points=5,
                          default=[1, 2, 3],
                          box=[[0, 1], [0, 1]])

    def test_CornerSampler_valid(self):
        ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
        ls_test_box_mixed = [[0.0, 25.0], []]
        ls_test_values_mixed = [[], [1, 2, 3]]
        ls_test_values = [['a', 'b', 'c'], [1, 2, 3]]
        np_actual_values = sampling_methods.sampler.CornerSampler.sample_points(box=ls_test_box,
                                                                                num_points=9)

        ls_expected_values = [[0., -25., -25.],
                              [0., -25., 25.],
                              [0., 0., -25.],
                              [0., 0., 25.],
                              [25., -25., -25.],
                              [25., -25., 25.],
                              [25., 0., -25.],
                              [25., 0., 25.],
                              [0., -25., -25.]]

        np.testing.assert_array_equal(np_actual_values, ls_expected_values)

        np_actual_values_mixed = sampling_methods.sampler.CornerSampler.sample_points(box=ls_test_box_mixed,
                                                                                      values=ls_test_values_mixed)

        ls_expected_values_mixed = [[0., 1],
                                    [0., 3],
                                    [25., 1],
                                    [25., 3]]

        np.testing.assert_array_equal(np_actual_values_mixed, ls_expected_values_mixed)

        np_actual_values_discrete = sampling_methods.sampler.CornerSampler.sample_points(values=ls_test_values,
                                                                                         num_points=9)

        np_expected_values_discrete = np.array([['a', 1],
                                                ['a', 3],
                                                ['c', 1],
                                                ['c', 3],
                                                ['a', 1],
                                                ['a', 3],
                                                ['c', 1],
                                                ['c', 3],
                                                ['a', 1]], dtype='O')

        np.testing.assert_array_equal(np_actual_values_discrete, np_expected_values_discrete)

    def test_CornerSampler_invalid(self):
        # box not given
        self.assertRaises(TypeError, sampling_methods.sampler.CornerSampler.sample_points,
                          num_points=10)

        # values nor box not given
        self.assertRaises(TypeError, sampling_methods.sampler.CornerSampler.sample_points)

        # not enough dimensions in box
        self.assertRaises(TypeError, sampling_methods.sampler.CornerSampler.sample_points,
                          box=[0.0, 1.0],
                          num_points=10)

        # too many dimensions in nPts
        self.assertRaises(TypeError, sampling_methods.sampler.CornerSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_points=[1, 2])

        # nPts type str
        self.assertRaises(ValueError, sampling_methods.sampler.CornerSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_points='f')

        # box not list
        self.assertRaises(TypeError, sampling_methods.sampler.CornerSampler.sample_points,
                          box=1.0,
                          num_points=10)

    def test_UniformSampler_valid(self):
        ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
        np_actual_values = sampling_methods.sampler.UniformSampler.sample_points(box=ls_test_box,
                                                                                 num_points=6)

        ls_expected_values = [[0., -25., -25.],
                              [5., -20., -15.],
                              [10., -15., -5.],
                              [15., -10., 5.],
                              [20., -5., 15.],
                              [25., 0., 25.]]

        np.testing.assert_array_equal(np_actual_values, ls_expected_values)

        np_actual_values_equal_area = sampling_methods.sampler.UniformSampler.sample_points(box=ls_test_box,
                                                                                            num_points=5,
                                                                                            equal_area_divs=True)
        ls_expected_values_equal_area = [[2.5, -22.5, -20.],
                                         [7.5, -17.5, -10.],
                                         [12.5, -12.5, 0.],
                                         [17.5, -7.5, 10.],
                                         [22.5, -2.5, 20.]]

        np.testing.assert_array_equal(np_actual_values_equal_area, ls_expected_values_equal_area)

    def test_UniformSampler_invalid(self):
        # box not given
        self.assertRaises(TypeError, sampling_methods.sampler.UniformSampler.sample_points,
                          num_points=10)

        # nDiv not given
        self.assertRaises(TypeError, sampling_methods.sampler.UniformSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]])

        # nDiv nor box not given
        self.assertRaises(TypeError, sampling_methods.sampler.UniformSampler.sample_points)

        # not enough dimensions in box
        self.assertRaises(ValueError, sampling_methods.sampler.UniformSampler.sample_points,
                          box=[0.0, 1.0],
                          num_points=10)

        # box not list
        self.assertRaises(ValueError, sampling_methods.sampler.UniformSampler.sample_points,
                          box=1.0,
                          num_points=10)

        # too many dimensions in box
        self.assertRaises(ValueError, sampling_methods.sampler.UniformSampler.sample_points,
                          box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                          num_points=10)

        # too many dimensions in nDiv
        self.assertRaises(TypeError, sampling_methods.sampler.UniformSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]],
                          num_points=[1, 2])

    def test_CartesianCrossSampler_valid(self):
        ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
        np_actual_values = sampling_methods.sampler.CartesianCrossSampler.sample_points(box=ls_test_box,
                                                                                        num_divisions=3)

        ls_expected_values = [[0.0, -25.0, -25.],
                              [0.0, -25.0, 0.],
                              [0.0, -25.0, 25.],
                              [0.0, -12.5, -25.],
                              [0.0, -12.5, 0.],
                              [0.0, -12.5, 25.],
                              [0.0, 0.0, -25.],
                              [0.0, 0.0, 0.],
                              [0.0, 0.0, 25.],
                              [12.5, -25.0, -25.],
                              [12.5, -25.0, 0.],
                              [12.5, -25.0, 25.],
                              [12.5, -12.5, -25.],
                              [12.5, -12.5, 0.],
                              [12.5, -12.5, 25.],
                              [12.5, 0.0, -25.],
                              [12.5, 0.0, 0.],
                              [12.5, 0.0, 25.],
                              [25.0, -25.0, -25.],
                              [25.0, -25.0, 0.],
                              [25.0, -25.0, 25.],
                              [25.0, -12.5, -25.],
                              [25.0, -12.5, 0.],
                              [25.0, -12.5, 25.],
                              [25.0, 0.0, -25.],
                              [25.0, 0.0, 0.],
                              [25.0, 0.0, 25.]]

        np.testing.assert_array_equal(np_actual_values, ls_expected_values)

        np_actual_values_equal_area = sampling_methods.sampler.CartesianCrossSampler.sample_points(box=ls_test_box,
                                                                                                   num_divisions=2,
                                                                                                   equal_area_divs=True)

        ls_expected_values_equal_area = [[6.25, -18.75, -12.5],
                                         [6.25, -18.75, 12.5],
                                         [6.25, -6.25, -12.5],
                                         [6.25, -6.25, 12.5],
                                         [18.75, -18.75, -12.5],
                                         [18.75, -18.75, 12.5],
                                         [18.75, -6.25, -12.5],
                                         [18.75, -6.25, 12.5]]

        np.testing.assert_array_equal(np_actual_values_equal_area, ls_expected_values_equal_area)

        ls_test_box_mixed = [[0.0, 25.0], []]
        ls_test_values_mixed = [[], [1, 2, 3]]

        np_actual_values_mixed = sampling_methods.sampler.CartesianCrossSampler.sample_points(box=ls_test_box_mixed,
                                                                                              values=ls_test_values_mixed,
                                                                                              num_divisions=3)

        ls_expected_values_mixed = [[0.0, 1],
                                    [0.0, 2],
                                    [0.0, 3],
                                    [12.5, 1],
                                    [12.5, 2],
                                    [12.5, 3],
                                    [25.0, 1],
                                    [25.0, 2],
                                    [25.0, 3]]

        np.testing.assert_array_equal(np_actual_values_mixed, ls_expected_values_mixed)

        ls_test_values = [['a', 'b', 'c'], [1, 2, 3]]

        np_actual_values_discrete = sampling_methods.sampler.CartesianCrossSampler.sample_points(num_divisions=3,
                                                                                                 values=ls_test_values)

        np_expected_values_discrete = np.array([['a', 1],
                                                ['a', 2],
                                                ['a', 3],
                                                ['b', 1],
                                                ['b', 2],
                                                ['b', 3],
                                                ['c', 1],
                                                ['c', 2],
                                                ['c', 3]], dtype='O')

        np.testing.assert_array_equal(np_actual_values_discrete, np_expected_values_discrete)

    def test_CartesianCrossSampler_invalid(self):
        # box or value not given
        self.assertRaises(TypeError, sampling_methods.sampler.CartesianCrossSampler.sample_points,
                          num_divisions=10)

        # num_divisions not given
        self.assertRaises(TypeError, sampling_methods.sampler.CartesianCrossSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]])

        # num_divisions, box, nor values not given
        self.assertRaises(TypeError, sampling_methods.sampler.CartesianCrossSampler.sample_points)

        # not enough dimensions in box
        self.assertRaises(IndexError, sampling_methods.sampler.CartesianCrossSampler.sample_points,
                          box=[0.0, 1.0],
                          num_divisions=10)

        # box not list
        self.assertRaises(TypeError, sampling_methods.sampler.CartesianCrossSampler.sample_points,
                          box=1.0,
                          num_divisions=10)

        # too many dimensions in box
        self.assertRaises(ValueError, sampling_methods.sampler.CartesianCrossSampler.sample_points,
                          box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                          num_divisions=10)

    def test_SamplePointsSampler_valid(self):
        ls_test_input = [(1, 2.3), (2, 3.4)]
        np_actual_values = sampling_methods.sampler.SamplePointsSampler.sample_points(samples=ls_test_input)

        np.testing.assert_array_equal(np_actual_values, ls_test_input)

    def test_SamplePointsSampler_invalid(self):
        # samples not given
        self.assertRaises(TypeError, sampling_methods.sampler.SamplePointsSampler.sample_points)

        # not enough dimensions in  samples
        self.assertRaises(np.AxisError, sampling_methods.sampler.SamplePointsSampler.sample_points,
                          samples=[0.0, 1.0])

        # too many dimensions in samples
        self.assertRaises(TypeError, sampling_methods.sampler.SamplePointsSampler.sample_points,
                          samples=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]])

        # samples not list
        self.assertRaises(np.AxisError, sampling_methods.sampler.SamplePointsSampler.sample_points,
                          samples=1)

        # samples list of strings
        self.assertRaises(ValueError, sampling_methods.sampler.SamplePointsSampler.sample_points,
                          samples=["foo", "bar"])

        # sample with incorrect size
        self.assertRaises(ValueError, sampling_methods.sampler.SamplePointsSampler.sample_points,
                          samples=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0, 2.0]])

    def test_RejectionSampler_valid(self):
        ls_test_box = [[-5., 5.0], [-5.0, 5.0], [-5.0, 5.0]]

        def func(x):
            return np.exp(np.sum(-np.power(x, 2.) / 2.))

        np_actual_values_rejection = sampling_methods.sampler.RejectionSampler.sample_points(num_points=25,
                                                                                             box=ls_test_box,
                                                                                             func=func,
                                                                                             seed=2018)
        np_expected_values_rejection = [[0.40057563, -1.25351001, -2.61304318],
                                        [-1.07381617, 0.00961220, 1.27224451],
                                        [-0.61153237, -0.05641841, -0.93891683],
                                        [0.00638401, -1.17238788, -0.18477953],
                                        [-2.24043485, -0.54363168, 1.30650383],
                                        [-0.20162045, -1.57054472, 0.00993160],
                                        [2.73830558, 1.09649763, -0.93251958],
                                        [-0.62149077, -1.95475357, 0.92823968],
                                        [-1.76550807, -0.21535468, -0.73526719],
                                        [-0.52720472, -0.58894374, -1.43066312],
                                        [1.53039716, -0.15851950, -0.64281130],
                                        [-0.28274113, 0.16373812, -1.02166101],
                                        [-0.92093671, 1.38762111, -1.74614927],
                                        [-0.02100135, -0.99098210, 0.20080051],
                                        [-1.44409931, 0.38640898, -1.19540595],
                                        [1.07644660, 1.46602592, -0.02166618],
                                        [0.12550015, 1.07923414, -0.13951586],
                                        [1.19383555, -1.37313410, 0.13410598],
                                        [0.53018472, 1.26069710, -2.47937573],
                                        [-0.08699791, 0.95906146, -0.66407291],
                                        [-0.54729379, 0.62944655, -0.97524317],
                                        [1.45894174, -1.08945671, 0.64928777],
                                        [-0.44666867, 1.45799030, 1.45998521],
                                        [-0.59862535, 0.95229270, -1.75587455],
                                        [-0.27850650, -0.30791447, -1.07258371]]

        np.testing.assert_array_almost_equal(np_actual_values_rejection, np_expected_values_rejection)

        np_actual_values_metropolis = sampling_methods.sampler.RejectionSampler.sample_points(num_points=25,
                                                                                              box=ls_test_box,
                                                                                              func=func,
                                                                                              seed=2018,
                                                                                              metropolis=True,
                                                                                              burn_in=1000)

        np_expected_values_metropolis = [[-1.41792483, -2.84079562, -3.58690606],
                                         [-0.59613595, -0.07309829, -0.60564416],
                                         [-1.56889534, -1.26488961, -1.52789929],
                                         [-0.87091969, -2.85938522, -1.19933636],
                                         [-1.50718497, -0.82696786, 0.98850861],
                                         [-2.9158022, -1.22711139, -0.40774782],
                                         [-3.00322143, -0.36339367, 2.61243505],
                                         [-1.12922303, -1.69903612, 1.45579197],
                                         [-1.22556711, 0.29214426, 1.29975049],
                                         [-0.56950507, 2.01672788, 0.96088011],
                                         [0.48219265, -0.15552369, 0.61913659],
                                         [-0.24890541, -1.52077338, 1.31333001],
                                         [0.69354689, -0.99653604, 1.38836756],
                                         [-1.84524373, -0.30884195, 0.81729428],
                                         [3.24987048, -1.22842400, -0.23691424],
                                         [0.23621548, -1.29768201, 1.69258297],
                                         [0.25906199, -1.77343239, 0.81158214],
                                         [0.38043382, -1.78100946, 0.82915394],
                                         [-0.65820766, -2.00797198, 0.93239357],
                                         [0.48609420, -0.75330191, 2.26987343],
                                         [0.85482598, -0.16131444, 1.05467902],
                                         [0.91724278, 1.05577052, 1.47391691],
                                         [1.80611106, 0.08171868, 1.16372164],
                                         [1.01369104, 0.14083339, 2.05100511],
                                         [1.69799361, 1.77466968, 2.34618417]]

        np.testing.assert_array_almost_equal(np_actual_values_metropolis, np_expected_values_metropolis)

    def test_RejectionSampler_invalid(self):
        ls_test_box = [[-5., 5.0], [-5.0, 5.0], [-5.0, 5.0]]

        def func(x):
            return np.exp(np.sum(-np.power(x, 2.) / 2.))

        # no parameters given
        self.assertRaises(TypeError, sampling_methods.sampler.RejectionSampler.sample_points)

        # nPts not given
        self.assertRaises(TypeError, sampling_methods.sampler.RejectionSampler.sample_points,
                          box=ls_test_box,
                          func=func)

        # box not given
        self.assertRaises(TypeError, sampling_methods.sampler.RejectionSampler.sample_points,
                          num_points=25,
                          func=func)

        # func not given
        self.assertRaises(TypeError, sampling_methods.sampler.RejectionSampler.sample_points,
                          num_points=25,
                          box=ls_test_box)

        # too many dimensions in box
        self.assertRaises(TypeError, sampling_methods.sampler.RejectionSampler.sample_points,
                          num_points=25,
                          box=[[[-5., 5.0], [-5.0, 5.0], [-5.0, 5.0]],
                               [[-5., 5.0], [-5.0, 5.0], [-5.0, 5.0]],
                               [[-5., 5.0], [-5.0, 5.0], [-5.0, 5.0]]],
                          func=func)

        # not enough dimensions in box
        self.assertRaises(TypeError, sampling_methods.sampler.RejectionSampler.sample_points,
                          num_points=25,
                          box=[-5., 5.],
                          func=func)

        # box not list
        self.assertRaises(TypeError, sampling_methods.sampler.RejectionSampler.sample_points,
                          num_points=25,
                          box=1,
                          func=func)

        # func not function
        self.assertRaises(TypeError, sampling_methods.sampler.RejectionSampler.sample_points,
                          num_points=25,
                          box=ls_test_box,
                          func="foo")

        # func does not return scalar
        self.assertRaises(ValueError, sampling_methods.sampler.RejectionSampler.sample_points,
                          num_points=25,
                          box=ls_test_box,
                          func=lambda x: np.exp(-np.power(x, 2.) / 2.))

        self.assertRaises(ValueError, sampling_methods.sampler.RejectionSampler.sample_points,
                          num_points=25,
                          box=ls_test_box,
                          func=lambda x: np.exp(-np.power(x, 2.) / 2.),
                          metropolis=True)

        # func returns string
        self.assertRaises(TypeError, sampling_methods.sampler.RejectionSampler.sample_points,
                          num_points=25,
                          box=ls_test_box,
                          func=lambda x: "bar")
        self.assertRaises(TypeError, sampling_methods.sampler.RejectionSampler.sample_points,
                          num_points=25,
                          box=ls_test_box,
                          func=lambda x: "bar",
                          metropolis=True)

    def test_ProbabilityDensityFunctionSampler_valid(self):
        ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
        np_actual_values_normal1 = sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=25,
            box=ls_test_box,
            dist='norm',
            seed=2018,
            loc=[0, 1, 2],
            scale=[1, 2,
                   3])

        np_actual_values_normal2 = sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=25,
            num_dim=3,
            dist='norm',
            seed=2018,
            loc=[0, 1, 2],
            scale=[1, 2,
                   3])

        np_expected_values_normal = [[-2.76767596e-01, 1.12742035e+00, 2.56500242e+00],
                                     [5.81851002e-01, 1.74125677e+00, -2.74069262e+00],
                                     [2.14839926e+00, -2.20908588e+00, 7.59344976e-01],
                                     [-1.27948700e+00, -3.33145875e+00, -1.22908762e+00],
                                     [5.02276889e-01, 1.76074025e+00, -5.55505664e-01],
                                     [8.56029296e-01, 4.46997821e-01, 4.35426380e+00],
                                     [-1.42790075e-01, -1.51363889e-01, 1.82683181e-01],
                                     [1.10078666e-01, 1.84370861e+00, 1.55439922e+00],
                                     [-6.88064791e-01, 2.40956059e+00, 2.40127728e+00],
                                     [4.33564082e-01, 1.09080232e+00, -7.84269975e-01],
                                     [5.10221003e-01, 2.06138014e+00, 1.87191272e+00],
                                     [-1.65130974e-01, 1.27625072e+00, 6.01374931e+00],
                                     [-1.35177905e+00, 1.74092627e+00, 1.93084137e+00],
                                     [5.46630750e-01, 8.92244393e-01, 4.79469766e+00],
                                     [1.23065512e+00, 3.14559200e+00, -1.35700491e+00],
                                     [1.07644610e+00, 1.83050518e+00, 2.40592246e+00],
                                     [-1.21062488e+00, -5.83114027e-01, 1.44430622e+00],
                                     [-3.06676569e-01, -6.06749747e-01, -3.39799248e-01],
                                     [-1.05741884e+00, 1.00800016e+00, 3.35980613e+00],
                                     [4.02056921e-01, 2.67637681e+00, 3.45225970e+00],
                                     [2.89165121e-01, 1.09416744e+00, -5.61257253e-03],
                                     [1.28273322e+00, -3.44237377e+00, -9.44906587e-01],
                                     [-1.06569580e+00, 1.89758089e+00, 4.29167103e-01],
                                     [-1.70663287e+00, 9.73132446e-01, 3.06222913e+00],
                                     [-1.72797393e-01, 2.57408495e+00, 1.08258842e+00]]

        np.testing.assert_array_almost_equal(np_actual_values_normal1, np_actual_values_normal2)
        np.testing.assert_array_almost_equal(np_actual_values_normal1, np_expected_values_normal)
        np.testing.assert_array_almost_equal(np_actual_values_normal2, np_expected_values_normal)

        np_actual_values_t1 = sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points(num_points=25,
                                                                                                       box=ls_test_box,
                                                                                                       dist='t',
                                                                                                       df=[1, 1, 1],
                                                                                                       seed=2018,
                                                                                                       loc=[0, 1, 2],
                                                                                                       scale=[1, 2, 3])

        np_actual_values_t2 = sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points(num_points=25,
                                                                                                       num_dim=3,
                                                                                                       dist='t',
                                                                                                       df=[1, 1, 1],
                                                                                                       seed=2018,
                                                                                                       loc=[0, 1, 2],
                                                                                                       scale=[1, 2, 3])

        np_expected_values_t = [[-4.38396851e-01, 1.80207294e+00, 9.57809380e+00],
                                [3.87860557e-01, -6.88818046e-02, 5.77063075e-01],
                                [-1.17273034e-01, -4.02653982e+00, -8.43897281e-01],
                                [9.72220842e-02, 2.62320953e+00, -1.27796888e+01],
                                [5.77768450e-01, 2.29377858e+00, 3.29600403e+00],
                                [-1.65427161e-01, 2.05853960e+02, -1.17740323e+01],
                                [-1.31848217e+00, 3.59189397e+00, 1.00827335e+01],
                                [-2.41074715e+00, 3.99150012e+00, 1.81229730e+00],
                                [5.91206459e-01, -5.87243841e+00, 2.05007250e+00],
                                [6.60219320e+00, 7.33970763e-02, 8.91551657e-01],
                                [-4.83539318e-01, 6.12506595e-01, 5.41805830e+00],
                                [8.55625426e-02, 3.23003840e+00, 9.81473626e-01],
                                [-1.70720800e-01, 1.62838068e+00, -2.77288159e-01],
                                [-2.24090270e-01, 1.82312419e+00, 8.98879869e+00],
                                [8.20668013e-02, -1.42407680e+01, -1.62087075e+01],
                                [1.90654863e+00, -1.74390999e+00, 1.38502700e+00],
                                [-1.08393467e+00, 3.49285295e+00, 8.79522340e+00],
                                [5.35484894e-03, 1.05911128e+00, 1.14268619e+00],
                                [-4.01811849e-02, 5.21501116e-01, 2.30405553e+03],
                                [3.36765585e+00, 7.78016467e+00, 4.10721500e+00],
                                [-5.86109479e+00, -4.50222965e+00, 7.58487152e+00],
                                [5.50014928e-01, 1.00968879e+00, 3.70347943e+00],
                                [-5.36937830e+00, 2.85703985e+00, 7.03414007e-01],
                                [7.68372982e+00, 2.78730367e+00, 3.79636406e+01],
                                [1.52954164e+00, -2.34433814e+00, 5.96064594e+00]]

        np.testing.assert_array_almost_equal(np_actual_values_t1, np_actual_values_t2)
        np.testing.assert_array_almost_equal(np_actual_values_t1, np_expected_values_t)
        np.testing.assert_array_almost_equal(np_actual_values_t2, np_expected_values_t)

        np_actual_values_lognormal1 = sampling_methods.sampler. \
            ProbabilityDensityFunctionSampler.sample_points(num_points=25,
                                                            box=ls_test_box,
                                                            dist='lognorm',
                                                            s=[3, 4, 5],
                                                            seed=2018,
                                                            loc=[0, 1, 2],
                                                            scale=[1, 2, 3])

        np_actual_values_lognormal2 = sampling_methods.sampler. \
            ProbabilityDensityFunctionSampler.sample_points(num_points=25,
                                                            num_dim=3,
                                                            dist='lognorm',
                                                            s=[3, 4, 5],
                                                            seed=2018,
                                                            loc=[0, 1, 2],
                                                            scale=[1, 2, 3])

        np_expected_values_lognormal = [[4.35917276e-01, 3.58051211e+00, 9.69278584e+00],
                                        [5.72906881e+00, 9.80800285e+00, 2.00111095e+00],
                                        [6.29671208e+02, 1.00326327e+00, 2.37940688e+00],
                                        [2.15267052e-02, 1.00034576e+00, 2.01379843e+00],
                                        [4.51240679e+00, 1.01579988e+01, 2.04240183e+00],
                                        [1.30408650e+01, 1.66175679e+00, 1.53772895e+02],
                                        [6.51570142e-01, 1.19997146e+00, 2.14511205e+00],
                                        [1.39129643e+00, 1.18110029e+01, 3.42752806e+00],
                                        [1.26920500e-01, 3.45242269e+01, 7.85565439e+00],
                                        [3.67183760e+00, 3.39828004e+00, 2.02896006e+00],
                                        [4.62123973e+00, 1.77083312e+01, 4.42330784e+00],
                                        [6.09331440e-01, 4.47518818e+00, 2.41395878e+03],
                                        [1.73296368e-02, 9.80218261e+00, 4.67339155e+00],
                                        [5.15461407e+00, 2.61225846e+00, 3.18221121e+02],
                                        [4.01236270e+01, 1.47105824e+02, 2.01114911e+00],
                                        [2.52629353e+01, 1.15292546e+01, 7.90116459e+00],
                                        [2.64665227e-02, 1.08432466e+00, 3.18821952e+00],
                                        [3.98507202e-01, 1.08043126e+00, 2.06074606e+00],
                                        [4.19089227e-02, 3.03225802e+00, 3.09322195e+01],
                                        [3.34066792e+00, 5.81626549e+01, 3.57521836e+01],
                                        [2.38093999e+00, 3.41447547e+00, 2.10602553e+00],
                                        [4.69085362e+01, 1.00027697e+00, 2.02215781e+00],
                                        [4.08811009e-02, 1.30408974e+01, 2.21883384e+00],
                                        [5.97662883e-03, 2.89536621e+00, 1.96192872e+01],
                                        [5.95477200e-01, 4.75867927e+01, 2.65024438e+00]]

        np.testing.assert_array_almost_equal(np_actual_values_lognormal1, np_actual_values_lognormal2)
        np.testing.assert_array_almost_equal(np_actual_values_lognormal1, np_expected_values_lognormal)
        np.testing.assert_array_almost_equal(np_actual_values_lognormal2, np_expected_values_lognormal)

    # Skip for Python 2
    # For some reason, scipy's distributions are raising a Value Error instead
    #  of a TypeError when 'scale' is the wrong type.
    # This is fixed in Python 3
    @unittest.skipIf(sys.version_info[0] < 3, "Not supported for Python 2")
    def test_ProbabilityDensityFunctionSampler_invalid(self):
        # no parameters given
        self.assertRaises(TypeError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points)

        # nPts not given
        self.assertRaises(TypeError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_dim=3,
                          dist='norm')

        # nDim or box not given
        self.assertRaises(ValueError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          dist='norm')

        # dist not given
        self.assertRaises(TypeError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          num_dim=3)

        # loc too short
        self.assertRaises(ValueError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          num_dim=3,
                          dist='norm',
                          loc=[1, 2])

        # loc too long
        self.assertRaises(ValueError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          num_dim=3,
                          dist='norm',
                          loc=[1, 2, 3, 4])

        # loc wrong type
        self.assertRaises(TypeError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          num_dim=3,
                          dist='norm',
                          loc="foo")

        # scale too short
        self.assertRaises(ValueError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          num_dim=3,
                          dist='norm',
                          scale=[1, 2])

        # scale too long
        self.assertRaises(ValueError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          num_dim=3,
                          dist='norm',
                          scale=[1, 2, 3, 4])

        # scale wrong type
        self.assertRaises(TypeError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          num_dim=3,
                          dist='norm',
                          scale="foo")

        # df too short
        self.assertRaises(ValueError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          num_dim=3,
                          dist='t',
                          df=[1, 2])

        # df too long
        self.assertRaises(ValueError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          num_dim=3,
                          dist='t',
                          df=[1, 2, 3, 4])

        # df wrong type
        self.assertRaises(TypeError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          num_dim=3,
                          dist='t',
                          df="foo")

        # s too short
        self.assertRaises(ValueError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          num_dim=3,
                          dist='lognorm',
                          s=[1, 2])

        # s too long
        self.assertRaises(ValueError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          num_dim=3,
                          dist='lognorm',
                          s=[1, 2, 3, 4])

        # s wrong type
        self.assertRaises(TypeError, sampling_methods.sampler.ProbabilityDensityFunctionSampler.sample_points,
                          num_points=25,
                          num_dim=3,
                          dist='lognorm',
                          s="foo")

    def test_MultiNormalSampler_valid(self):
        np_actual_values = sampling_methods.sampler.MultiNormalSampler.sample_points(num_points=25,
                                                                                     mean=[1.0, 2.0, 3.0],
                                                                                     covariance=[[1.0, 0.5, 0.1],
                                                                                                 [0.5, 1.0, 0.5],
                                                                                                 [0.1, 0.5, 1.0]],
                                                                                     seed=2018)
        np_expected_values = [[2.18507508, 1.33219283, 3.40444004],
                              [2.45580305, 2.79099985, 3.78192789],
                              [0.97809132, 2.42311467, 2.83040529],
                              [0.99830677, 1.67857486, 2.31377346],
                              [2.64062947, 2.69627779, 3.90724736],
                              [-0.63700440, 1.15782608, 2.98721932],
                              [2.07661568, 2.83245411, 3.53719972],
                              [-1.07546486, 1.56953727, 2.35431609],
                              [1.26564231, 1.99785346, 3.18016615],
                              [0.75515235, 3.28813876, 5.66078321],
                              [0.92229847, 2.06972309, 3.69465685],
                              [0.69631836, 1.13603517, 2.63540631],
                              [1.13857746, 1.89814276, 2.64154901],
                              [0.31965689, 1.36832078, 1.76253708],
                              [1.78986522, 2.36816818, 3.78449855],
                              [-0.39596770, 1.76565840, 4.58406723],
                              [1.59020526, 1.93166834, 2.53427698],
                              [1.50373515, 3.88863563, 4.05857294],
                              [1.94032257, 3.02900133, 2.88746379],
                              [0.93033979, 2.53087276, 2.75088314],
                              [1.92030482, 2.04845218, 2.12530156],
                              [-0.35170170, 1.09994845, 3.14959654],
                              [0.73150691, 1.97378189, 3.77789695],
                              [-0.05697331, 1.98179841, 2.83996390],
                              [1.51067421, 2.60407247, 3.03563090]]

        np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)

    def test_MultiNormalSampler_invalid(self):
        # no parameters given
        self.assertRaises(TypeError, sampling_methods.sampler.MultiNormalSampler.sample_points)

        # nPts not given
        self.assertRaises(TypeError, sampling_methods.sampler.MultiNormalSampler.sample_points,
                          mean=[1.0, 2.0, 3.0],
                          covariance=[[1.0, 0.5, 0.1],
                                      [0.5, 1.0, 0.5],
                                      [0.1, 0.5, 1.0]])

        # mean and covariance different lengths
        self.assertRaises(ValueError, sampling_methods.sampler.MultiNormalSampler.sample_points,
                          num_points=25,
                          mean=[1.0, 2.0],
                          covariance=[[1.0, 0.5, 0.1],
                                      [0.5, 1.0, 0.5],
                                      [0.1, 0.5, 1.0]])

        # mean not list
        self.assertRaises(ValueError, sampling_methods.sampler.MultiNormalSampler.sample_points,
                          num_points=25,
                          mean=1.0,
                          covariance=[[1.0, 0.5, 0.1],
                                      [0.5, 1.0, 0.5],
                                      [0.1, 0.5, 1.0]])

        # mean nested list
        self.assertRaises(ValueError, sampling_methods.sampler.MultiNormalSampler.sample_points,
                          num_points=25,
                          mean=[[1.0, 0.5, 0.1],
                                [0.5, 1.0, 0.5],
                                [0.1, 0.5, 1.0]],
                          covariance=[[1.0, 0.5, 0.1],
                                      [0.5, 1.0, 0.5],
                                      [0.1, 0.5, 1.0]])

        # covariance double nested list
        self.assertRaises(ValueError, sampling_methods.sampler.MultiNormalSampler.sample_points,
                          num_points=25,
                          mean=[1.0, 2.0, 3.0],
                          covariance=[[[1.0, 0.5, 0.1],
                                       [0.5, 1.0, 0.5],
                                       [0.1, 0.5, 1.0]],
                                      [[1.0, 0.5, 0.1],
                                       [0.5, 1.0, 0.5],
                                       [0.1, 0.5, 1.0]],
                                      [[1.0, 0.5, 0.1],
                                       [0.5, 1.0, 0.5],
                                       [0.1, 0.5, 1.0]]])

        # covariance not positive semi-definite
        self.assertRaises(ValueError, sampling_methods.sampler.MultiNormalSampler.sample_points,
                          num_points=25,
                          mean=[1.0, 2.0, 3.0],
                          covariance=[[-1.0, 0.5, 0.1],
                                      [0.5, 1.0, 0.5],
                                      [0.1, 0.5, 1.0]])

        if sys.version_info[0] >= 3:
            # covariance not symmetric
            self.assertWarns(RuntimeWarning, sampling_methods.sampler.MultiNormalSampler.sample_points,
                             num_points=25,
                             mean=[1.0, 2.0, 3.0],
                             covariance=[[1.0, 0.5, 0.8],
                                         [0.2, 1.0, 0.6],
                                         [0.4, 0.3, 1.0]])

    @unittest.expectedFailure
    def test_FaceSampler_valid(self):
        ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
        np_actual_values = sampling_methods.sampler.FaceSampler.sample_points(box=ls_test_box,
                                                                              num_divisions=3)

        ls_expected_values = [[0.0, -25.0, -25.0],
                              [0.0, -25.0, 0.0],
                              [0.0, -25.0, 25.0],
                              [0.0, -12.5, -25.0],
                              [0.0, -12.5, 0.0],
                              [0.0, -12.5, 25.0],
                              [0.0, 0.0, -25.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 25.0],
                              [12.5, -25.0, -25.0],
                              [12.5, -25.0, 0.0],
                              [12.5, -25.0, 25.0],
                              [12.5, -12.5, -25.0],
                              [12.5, -12.5, 25.0],
                              [12.5, 0.0, -25.0],
                              [12.5, 0.0, 0.0],
                              [12.5, 0.0, 25.0],
                              [25.0, -25.0, -25.0],
                              [25.0, -25.0, 0.0],
                              [25.0, -25.0, 25.0],
                              [25.0, -12.5, -25.0],
                              [25.0, -12.5, 0.0],
                              [25.0, -12.5, 25.0],
                              [25.0, 0.0, -25.0],
                              [25.0, 0.0, 0.0],
                              [25.0, 0.0, 25.0]]

        np.testing.assert_array_equal(np_actual_values, ls_expected_values)

        np_actual_values_equal_area = sampling_methods.sampler.FaceSampler.sample_points(box=ls_test_box,
                                                                                         num_divisions=2,
                                                                                         equal_area_divs=True)

        ls_expected_values_equal_area = [[0., -25., -25.],
                                         [0., -25., 25.],
                                         [0., 0., -25.],
                                         [0., 0., 25.],
                                         [25., -25., -25.],
                                         [25., -25., 25.],
                                         [25., 0., -25.],
                                         [25., 0., 25.]]

        np.testing.assert_array_equal(np_actual_values_equal_area, ls_expected_values_equal_area)

    def test_FaceSampler_invalid(self):
        # box not given
        self.assertRaises(TypeError, sampling_methods.sampler.FaceSampler.sample_points,
                          num_divisions=10)

        # nDiv not given
        self.assertRaises(TypeError, sampling_methods.sampler.FaceSampler.sample_points,
                          box=[[0.0, 1.0], [0.0, 1.0]])

        # nDiv nor box not given
        self.assertRaises(TypeError, sampling_methods.sampler.FaceSampler.sample_points)

        # not enough dimensions in box
        self.assertRaises(TypeError, sampling_methods.sampler.FaceSampler.sample_points,
                          box=[0.0, 1.0],
                          num_divisions=10)

        # box not list
        self.assertRaises(TypeError, sampling_methods.sampler.FaceSampler.sample_points,
                          box=1.0,
                          num_divisions=10)

        # too many dimensions in box
        self.assertRaises(ValueError, sampling_methods.sampler.FaceSampler.sample_points,
                          box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                          num_divisions=10)

    def test_UserValueSampler_valid(self):
        points1_tab = os.path.join(os.path.dirname(__file__), "points1.tab")
        np_actual_values1 = sampling_methods.sampler.UserValueSampler.sample_points(user_samples_file=points1_tab)

        ls_expected_values1 = [[1.0, 1.0, 1.0],
                               [0.2, 3.0, 4.0],
                               [5.0, 0.6, 7.0],
                               [8.0, 9.0, 0.1],
                               [1.0, 1.0, 1.0]]

        np.testing.assert_array_equal(np_actual_values1, ls_expected_values1)
        self.assertEqual(np_actual_values1.dtype, float)

        points2_tab = os.path.join(os.path.dirname(__file__), "points2.tab")
        np_actual_values2 = sampling_methods.sampler.UserValueSampler.sample_points(user_samples_file=points2_tab)

        ls_expected_values2 = [['1.0', '1.0', '1.0'],
                               ['.2', '3.0', '4.0'],
                               ['5.0', '.6', '7.0'],
                               ['8.0', '9.0', '.1'],
                               ['1.0', '1.0', '1.0'],
                               ['foo', 'bar', 'zyzzx']]

        np.testing.assert_array_equal(np_actual_values2, ls_expected_values2)

    def test_UserValueSampler_invalid(self):
        # user_samples_file not given
        self.assertRaises(TypeError, sampling_methods.sampler.UserValueSampler.sample_points)

        # user_samples_file not string
        self.assertRaises(RuntimeError, sampling_methods.sampler.UserValueSampler.sample_points,
                          user_samples_file=123)

        # user_samples_file points to invalid file
        self.assertRaises(RuntimeError, sampling_methods.sampler.UserValueSampler.sample_points,
                          user_samples_file="points3.tab")


#
# ====================================================================
#                                 main
# ====================================================================
#
#  Run the unit tests.
#
if __name__ == '__main__':
    unittest.main()
