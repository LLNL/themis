from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as gpr

import sampling_methods.sampler
import sampling_methods.adaptive_sampler


def helper_function(np_input):
    np_input = np_input.astype(float)
    out = np.sin(np_input[:, 0]) + np.cos(np_input[:, 1]) + np.tanh(np_input[:, 2])
    return out.reshape(-1, 1)


class TestAdaptiveSamplers(unittest.TestCase):

    def setUp(self):
        ls_test_box = [[0.0, 5.0], [-5.0, 0.0], [-5.0, 5.0]]
        self.np_train_input = sampling_methods.sampler.LatinHyperCubeSampler.sample_points(num_points=200,
                                                                                           box=ls_test_box,
                                                                                           seed=2018)
        self.np_train_output = helper_function(self.np_train_input)
        self.surrogate_model = gpr().fit(self.np_train_input, self.np_train_output)
        self.np_candidate_points = sampling_methods.sampler.LatinHyperCubeSampler.sample_points(num_points=200,
                                                                                                box=ls_test_box,
                                                                                                seed=2019)

    def test_ActiveLearningSampler_valid(self):
        np_actual_values = sampling_methods.adaptive_sampler. \
            ActiveLearningSampler.sample_points(num_points=5,
                                                cand_points=self.np_candidate_points,
                                                model=self.surrogate_model)

        np_expected_values = [[ 3.55452106, -4.11850436,  4.85333288],
                              [ 0.05094415, -3.38346499, -1.49442071],
                              [ 0.16384714, -3.29100777, -4.8358629 ],
                              [ 0.28111892, -4.91649867,  0.87429284],
                              [ 4.26925302, -4.71142410,  4.96952397]]

        np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)

    def test_ActiveLearningSampler_invalid(self):
        # num_points not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.ActiveLearningSampler.sample_points,
                          cand_points=self.np_candidate_points,
                          model=self.surrogate_model)

        # cand_points or box/num_cand_points not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.ActiveLearningSampler.sample_points,
                          num_points=5,
                          model=self.surrogate_model)

        # model not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.ActiveLearningSampler.sample_points,
                          num_points=5,
                          num_cand_points=self.np_candidate_points)


    def test_DeltaSampler_valid(self):
        np_actual_values = sampling_methods.adaptive_sampler. \
            DeltaSampler.sample_points(num_points=5,
                                       cand_points=self.np_candidate_points,
                                       model=self.surrogate_model,
                                       X=self.np_train_input,
                                       Y=self.np_train_output)

        np_expected_values = [[ 1.99899988, -1.81340482,  0.29633238],
                              [ 0.53101766, -2.32897332, -0.87512164],
                              [ 4.73630199, -2.02933796, -0.17188005],
                              [ 3.76551458, -4.65149758,  4.77579306],
                              [ 3.55452106, -4.11850436,  4.85333288]]

        np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)

    def test_DeltaSampler_invalid(self):
        # num_points not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.DeltaSampler.sample_points,
                          cand_points=self.np_candidate_points,
                          model=self.surrogate_model,
                          X=self.np_train_input,
                          Y=self.np_train_output)

        # cand_points or box/num_cand_points not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.DeltaSampler.sample_points,
                          num_points=5,
                          model=self.surrogate_model,
                          X=self.np_train_input,
                          Y=self.np_train_output)

        # model not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.DeltaSampler.sample_points,
                          num_points=5,
                          cand_points=self.np_candidate_points,
                          X=self.np_train_input,
                          Y=self.np_train_output)

        # X not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.DeltaSampler.sample_points,
                          num_points=5,
                          cand_points=self.np_candidate_points,
                          model=self.surrogate_model,
                          Y=self.np_train_output)

        # Y not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.DeltaSampler.sample_points,
                          num_points=5,
                          cand_points=self.np_candidate_points,
                          model=self.surrogate_model,
                          X=self.np_train_input)

    def test_ExpectedImprovementSampler_valid(self):
        np_actual_values = sampling_methods.adaptive_sampler. \
            ExpectedImprovementSampler.sample_points(num_points=5,
                                                     cand_points=self.np_candidate_points,
                                                     model=self.surrogate_model,
                                                     X=self.np_train_input,
                                                     Y=self.np_train_output)

        np_expected_values = [[ 4.73630199, -2.02933796, -0.17188005],
                              [ 1.18354594, -4.79710851, -3.98877172],
                              [ 0.53101766, -2.32897332, -0.87512164],
                              [ 3.76551458, -4.65149758,  4.77579306],
                              [ 3.55452106, -4.11850436,  4.85333288]]

        np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)

    def test_ExpectedImprovementSampler_invalid(self):
        # num_points not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                          cand_points=self.np_candidate_points,
                          model=self.surrogate_model,
                          X=self.np_train_input,
                          Y=self.np_train_output)

        # cand_points or box/num_cand_points not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                          num_points=5,
                          model=self.surrogate_model,
                          X=self.np_train_input,
                          Y=self.np_train_output)

        # model not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                          num_points=5,
                          cand_points=self.np_candidate_points,
                          X=self.np_train_input,
                          Y=self.np_train_output)

        # X not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                          num_points=5,
                          cand_points=self.np_candidate_points,
                          model=self.surrogate_model,
                          Y=self.np_train_output)

        # Y not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                          num_points=5,
                          cand_points=self.np_candidate_points,
                          model=self.surrogate_model,
                          X=self.np_train_input)

    def test_LearningExpectedImprovementSampler_valid(self):
        np_actual_values = sampling_methods.adaptive_sampler. \
            LearningExpectedImprovementSampler.sample_points(num_points=5,
                                                             cand_points=self.np_candidate_points,
                                                             model=self.surrogate_model,
                                                             X=self.np_train_input,
                                                             Y=self.np_train_output)

        np_expected_values = [[ 1.385293500, -1.96129783, -4.13900737],
                              [ 1.60678695, -3.83050635, -1.70850718],
                              [ 1.29085474, -1.38001608,  4.91634429],
                              [ 1.33310570, -0.80990864, -2.60642949],
                              [ 1.80971707, -4.39837040,  2.06877362]]

        np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)

    def test_LearningExpectedImprovementSampler_invalid(self):
        # num_points not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                          cand_points=self.np_candidate_points,
                          model=self.surrogate_model,
                          X=self.np_train_input,
                          Y=self.np_train_output)

        # cand_points or box/num_cand_points not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                          num_points=5,
                          model=self.surrogate_model,
                          X=self.np_train_input,
                          Y=self.np_train_output)

        # model not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                          num_points=5,
                          cand_points=self.np_candidate_points,
                          X=self.np_train_input,
                          Y=self.np_train_output)

        # X not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                          num_points=5,
                          cand_points=self.np_candidate_points,
                          model=self.surrogate_model,
                          Y=self.np_train_output)

        # Y not given
        self.assertRaises(TypeError, sampling_methods.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                          num_points=5,
                          cand_points=self.np_candidate_points,
                          model=self.surrogate_model,
                          X=self.np_train_input)


#
# ====================================================================
#                                 main
# ====================================================================
#
#  Run the unit tests.
#
if __name__ == '__main__':
    unittest.main()
