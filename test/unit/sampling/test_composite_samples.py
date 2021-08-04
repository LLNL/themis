from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import numpy as np

import uqp.sampling.composite_samples
import uqp.sampling.sampler


class TestCompositeSamples(unittest.TestCase):

    def test_genSamples_valid(self):
        test_samples = uqp.sampling.composite_samples.Samples()

        ls_test_variables = ['x', 'y']
        for var in ls_test_variables:
            test_samples.set_continuous_variable(var, 0, 0, 1)

        test_sampler = uqp.sampling.sampler.SamplePointsSampler()

        np_expected_points = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        test_samples.generate_samples(ls_test_variables, test_sampler, samples=np_expected_points)

        for i, var in enumerate(ls_test_variables):
            expected = np_expected_points.T[i]
            actual = test_samples.dt_variables[var].np_points
            np.testing.assert_array_equal(expected, actual)

        test_samples2 = uqp.sampling.composite_samples.Samples()

        test_samples2.set_continuous_variable('Cont', 1, 2, 3)
        test_samples2.set_discrete_variable('Disc', [1, 2, 3], 1)
        test_samples2.set_discrete_ordered_variable('DiscOrdered', [1, 2, 3], 1)

        test_samples2.generate_samples(['Cont', 'Disc', 'DiscOrdered'],
                                       uqp.sampling.sampler.LatinHyperCubeSampler(),
                                       num_points=3)
        test_samples2.generate_samples(['Cont', 'DiscOrdered'],
                                       uqp.sampling.sampler.CornerSampler())

    def test_genSamples_invalid(self):
        test_samples = uqp.sampling.composite_samples.Samples()

        # Variable 'test1' not added yet
        self.assertRaises(KeyError, test_samples.generate_samples,
                          ['test1'],
                          uqp.sampling.sampler.SamplePointsSampler(),
                          samples=np.array([[1], [3]]))

        # Variable 'x' added, but Variable 'y' not added yet
        test_samples.set_continuous_variable('test1', 0, 0, 1)
        self.assertRaises(KeyError, test_samples.generate_samples,
                          ['test1', 'test2'],
                          uqp.sampling.sampler.SamplePointsSampler(),
                          samples=np.array([[1, 2], [3, 4]]))

        # Variable type not list of strings
        self.assertRaises(TypeError, test_samples.generate_samples,
                          [0],
                          uqp.sampling.sampler.SamplePointsSampler(),
                          samples=np.array([[1], [3]]))

        # Variable type not list
        self.assertRaises(KeyError, test_samples.generate_samples,
                          'test1',
                          uqp.sampling.sampler.SamplePointsSampler(),
                          samples=np.array([[1], [3]]))

        # 'sampler' parameter not a sampler.Sampler
        self.assertRaises(ValueError, test_samples.generate_samples,
                          ['test1'],
                          'foo')

        # No kwArgs given
        self.assertRaises(TypeError, test_samples.generate_samples,
                          ['test1'],
                          uqp.sampling.sampler.SamplePointsSampler())

        # Incorrect kwArgs given
        self.assertRaises(TypeError, test_samples.generate_samples,
                          ['test1'],
                          uqp.sampling.sampler.SamplePointsSampler(),
                          incorrectArgument='foo')

        # Check that no points were generated
        np_expected = np.array([])
        np_actual = test_samples.dt_variables['test1'].np_points

        np.testing.assert_array_equal(np_expected, np_actual)

    def test_setContinuousVariable_valid(self):
        # Without scaled parameters
        test_samples = uqp.sampling.composite_samples.Samples()

        str_name1 = 'test1'
        f_low_val1 = -1.1
        f_default_val1 = 0.1
        f_high_val1 = 1.1

        test_samples.set_continuous_variable(str_name1, f_low_val1, f_default_val1, f_high_val1)

        expected_variable1 = uqp.sampling.composite_samples.ContinuousVariable(str_name1,
                                                                           f_low_val1,
                                                                           f_default_val1,
                                                                           f_high_val1,
                                                                           None, None, None, None)
        actual_variable1 = test_samples.dt_variables[str_name1]

        self.assertEqual(expected_variable1, actual_variable1)

        # With scaled parameters
        str_name2 = 'test2'
        f_low_val2 = -2.2
        f_default_val2 = 0.2
        f_high_val2 = 2.2
        f_scaled_low_val2 = -3.3
        f_scaled_default_val2 = .3
        f_scaled_high_val2 = 3.3
        str_scaling_type = 'lin'
        test_samples.set_continuous_variable(str_name2, f_low_val2, f_default_val2, f_high_val2,
                                             str_scaling_type,
                                             f_scaled_low_val2, f_scaled_default_val2, f_scaled_high_val2)

        expected_variable2 = uqp.sampling.composite_samples.ContinuousVariable(str_name2,
                                                                           f_low_val2,
                                                                           f_default_val2,
                                                                           f_high_val2,
                                                                           str_scaling_type,
                                                                           f_scaled_low_val2,
                                                                           f_scaled_default_val2,
                                                                           f_scaled_high_val2)

        actual_variable2 = test_samples.dt_variables[str_name2]
        self.assertEqual(expected_variable2, actual_variable2)

    def test_setContinuousVariable_invalid(self):
        test_samples = uqp.sampling.composite_samples.Samples()

        # No arguments
        self.assertRaises(TypeError, test_samples.set_continuous_variable)

        # Too many arguments
        self.assertRaises(TypeError, test_samples.set_continuous_variable,
                          'test1', -1, 0, 1, 'lin', -2, 0, 2, 'extra_argument')

        # Ensure no variables were added, i.e. Ensure an error does not change the object's state
        expected = {}
        actual = test_samples.dt_variables
        self.assertDictEqual(expected, actual)

    def test_setDiscreteVariable_valid(self):
        test_samples = uqp.sampling.composite_samples.Samples()

        str_name = 'test'
        ls_values = [1, 2, 3, 1.1, 2.2, 3.3, 'foo', 'bar']
        default = 'foo'
        test_samples.set_discrete_variable(str_name, ls_values, default)

        expected_variable = uqp.sampling.composite_samples.DiscreteVariable(str_name, ls_values, default)
        actual_variable = test_samples.dt_variables[str_name]

        self.assertEqual(expected_variable, actual_variable)

        expected_values = ls_values
        actual_values = test_samples.dt_variables[str_name].ls_values

        self.assertEqual(expected_values, actual_values)

    def test_setDiscreteVariable_invalid(self):
        test_samples = uqp.sampling.composite_samples.Samples()

        # No arguments
        self.assertRaises(TypeError, test_samples.set_discrete_variable)

        # Too many arguments
        self.assertRaises(TypeError, test_samples.set_discrete_variable,
                          'test1', ['a', 'b', 1, 2], 'a', 'extra_argument')

        # Ensure no variables were added, i.e. Ensure an error does not change the object's state
        expected = {}
        actual = test_samples.dt_variables
        self.assertDictEqual(expected, actual)

        test_samples.set_discrete_variable('X', [1, 2, 3, 4], 1)

        # Cannot use Continuous Sampler on Discrete Variable
        self.assertRaises(TypeError,
                          test_samples.generate_samples,
                          ['X'], uqp.sampling.sampler.ProbabilityDensityFunctionSampler(), num_points=10, dist='norm')

        # Ensure no points were added
        np.testing.assert_array_equal(test_samples.dt_variables['X'].np_points, np.array([]))

    def test_setDiscreteOrderedVariable_valid(self):
        test_samples = uqp.sampling.composite_samples.Samples()

        str_name = 'test'
        ls_values = [1, 2, 3, 1.1, 2.2, 3.3, 'foo', 'bar']
        default = 'foo'
        test_samples.set_discrete_ordered_variable(str_name, ls_values, default)

        expected_variable = uqp.sampling.composite_samples.DiscreteOrderedVariable(str_name, ls_values, default)
        actual_variable = test_samples.dt_variables[str_name]

        self.assertEqual(expected_variable, actual_variable)

        expected_values = ls_values
        actual_values = test_samples.dt_variables[str_name].ls_values

        self.assertEqual(expected_values, actual_values)

    def test_setDiscreteOrderedVariable_invalid(self):
        test_samples = uqp.sampling.composite_samples.Samples()

        # No arguments
        self.assertRaises(TypeError, test_samples.set_discrete_ordered_variable)

        # Too many arguments
        self.assertRaises(TypeError, test_samples.set_discrete_ordered_variable,
                          'test1', ['a', 'b', 1, 2], 'a', 'extra_argument')

        # Ensure no variables were added, i.e. Ensure an error does not change the object's state
        expected = {}
        actual = test_samples.dt_variables
        self.assertDictEqual(expected, actual)

    def test_getPoints_valid(self):
        test_samples = uqp.sampling.composite_samples.Samples()
        test_sampler = uqp.sampling.sampler.LatinHyperCubeSampler()

        i_num_points = 100
        ls_box = [[-1, 1], [-2, 3], [5.5, 6.7]]
        i_seed = 2016

        np_expected = test_sampler.sample_points(num_points=i_num_points, box=ls_box, seed=i_seed)

        ls_variables = ['var3', 'var1', 'var2']
        for var, rng in zip(ls_variables, ls_box):
            test_samples.set_continuous_variable(var, rng[0], 0, rng[1])
        test_samples.generate_samples(ls_variables, test_sampler, num_points=i_num_points, seed=i_seed)

        # No arguments
        np_actual1 = test_samples.get_points()
        np.testing.assert_array_equal(np_actual1, np_expected)

        # list argument with three
        np_actual2 = test_samples.get_points(ls_variables)
        np.testing.assert_array_equal(np_actual2, np_expected)

        # list argument with one
        np_actual3 = test_samples.get_points([ls_variables[0]])
        np.testing.assert_array_equal(np_actual3, np.array([np_expected.T[0]]).T)

        # list argument with two
        np_actual4 = test_samples.get_points([ls_variables[0], ls_variables[1]])
        np_expected4 = np.array([np_expected.T[0], np_expected.T[1]]).T
        np.testing.assert_array_equal(np_actual4, np_expected4)

        # string argument
        np_actual5 = test_samples.get_points(ls_variables[0])
        np.testing.assert_array_equal(np_actual5, np.array([np_expected.T[0]]).T)

    def test_getPoints_invalid(self):
        test_samples = uqp.sampling.composite_samples.Samples()
        test_sampler = uqp.sampling.sampler.LatinHyperCubeSampler()

        i_num_points = 100
        ls_box = [[-1, 1], [-2, 3], [5.5, 6.7]]
        i_seed = 2016

        ls_variables = ['var1', 'var2', 'var3']
        for var, rng in zip(ls_variables, ls_box):
            test_samples.set_continuous_variable(var, rng[0], 0, rng[1])
        test_samples.generate_samples(ls_variables, test_sampler, num_points=i_num_points, seed=i_seed)

        # Variable does not exist in testSamples (string argument)
        self.assertRaises(KeyError, test_samples.get_points, 'invalidVar')

        # Variable does not exist in testSamples (list argument len 1)
        self.assertRaises(KeyError, test_samples.get_points, ['invalidVar'])

        # Variable does not exist in testSamples (list argument len 2)
        self.assertRaises(KeyError, test_samples.get_points, ['invalidVar1', 'invalidVar2'])

        # Variable does not exist in testSamples (list argument 1 invalid 1 valid)
        self.assertRaises(KeyError, test_samples.get_points, ['invalidVar', ls_variables[0]])

        # Invalid type
        self.assertRaises(TypeError, test_samples.get_points, 123)

        # Invalid type in list
        self.assertRaises(TypeError, test_samples.get_points, [123])

    def test_getLength_valid(self):
        test_samples = uqp.sampling.composite_samples.Samples()
        test_sampler = uqp.sampling.sampler.LatinHyperCubeSampler()

        i_num_points = 100
        ls_box = [[-1, 1], [-2, 3], [5.5, 6.7]]
        i_seed = 2016

        ls_variables = ['var1', 'var2', 'var3']
        for var, rng in zip(ls_variables, ls_box):
            test_samples.set_continuous_variable(var, rng[0], 0, rng[1])
        test_samples.generate_samples(ls_variables, test_sampler, num_points=i_num_points, seed=i_seed)

        np_expected1 = np.array([i_num_points] * len(ls_variables))
        np_expected2 = np.array([i_num_points] * 1)
        np_expected3 = np.array([i_num_points] * 2)

        # No arguments
        np_actual1 = test_samples.get_length()
        np.testing.assert_array_equal(np_actual1, np_expected1)

        # list argument with both
        np_actual2 = test_samples.get_length(ls_variables)
        np.testing.assert_array_equal(np_actual2, np_expected1)

        # list argument with one
        np_actual3 = test_samples.get_length([ls_variables[0]])
        np.testing.assert_array_equal(np_actual3, np_expected2)

        # list argument with two
        np_actual4 = test_samples.get_length([ls_variables[0], ls_variables[1]])
        np.testing.assert_array_equal(np_actual4, np_expected3)

        # string argument
        np_actual5 = test_samples.get_length(ls_variables[0])
        np.testing.assert_array_equal(np_actual5, np_expected2)

    def test_getLength_invalid(self):
        test_samples = uqp.sampling.composite_samples.Samples()
        test_sampler = uqp.sampling.sampler.LatinHyperCubeSampler()

        i_num_points = 100
        ls_box = [[-1, 1], [-2, 3], [5.5, 6.7]]
        i_seed = 2016

        ls_variables = ['var1', 'var2', 'var3']
        for var, rng in zip(ls_variables, ls_box):
            test_samples.set_continuous_variable(var, rng[0], 0, rng[1])
        test_samples.generate_samples(ls_variables, test_sampler, num_points=i_num_points, seed=i_seed)

        # Variable does not exist in testSamples (string argument)
        self.assertRaises(KeyError, test_samples.get_length, 'invalidVar')

        # Variable does not exist in testSamples (list argument len 1)
        self.assertRaises(KeyError, test_samples.get_length, ['invalidVar'])

        # Variable does not exist in testSamples (list argument len 2)
        self.assertRaises(KeyError, test_samples.get_length, ['invalidVar1', 'invalidVar2'])

        # Variable does not exist in testSamples (list argument 1 invalid 1 valid)
        self.assertRaises(KeyError, test_samples.get_length, ['invalidVar', ls_variables[0]])

        # Invalid type
        self.assertRaises(TypeError, test_samples.get_length, 123)

        # Invalid type in list
        self.assertRaises(TypeError, test_samples.get_length, [123])

    def test_getRange_valid(self):
        test_samples = uqp.sampling.composite_samples.Samples()

        test_samples.set_continuous_variable('test1', -1.1, 0.1, 1.1)

        np.testing.assert_array_equal(test_samples.get_ranges('test1'), np.array([[-1.1, 1.1]]))

        test_samples.set_continuous_variable('test2', -2.3, .3, 3.2)
        np.testing.assert_array_equal(test_samples.get_ranges(['test1', 'test2']), np.array([[-1.1, 1.1], [-2.3, 3.2]]))
        np.testing.assert_array_equal(test_samples.get_ranges(), np.array([[-1.1, 1.1], [-2.3, 3.2]]))

        for var, test in zip(test_samples.get_variable_list(), ['test1', 'test2']):
            self.assertEqual(var, test)

    def test_getRange_invalid(self):
        test_samples = uqp.sampling.composite_samples.Samples()
        test_sampler = uqp.sampling.sampler.LatinHyperCubeSampler()

        i_num_points = 100
        ls_box = [[-1, 1], [-2, 3], [5.5, 6.7]]
        i_seed = 2016

        ls_variables = ['var1', 'var2', 'var3']
        for var, rng in zip(ls_variables, ls_box):
            test_samples.set_continuous_variable(var, rng[0], 0, rng[1])
        test_samples.generate_samples(ls_variables, test_sampler, num_points=i_num_points, seed=i_seed)

        # Variable does not exist in testSamples (string argument)
        self.assertRaises(KeyError, test_samples.get_ranges, 'invalidVar')

        # Variable does not exist in testSamples (list argument len 1)
        self.assertRaises(KeyError, test_samples.get_ranges, ['invalidVar'])

        # Variable does not exist in testSamples (list argument len 2)
        self.assertRaises(KeyError, test_samples.get_ranges, ['invalidVar1', 'invalidVar2'])

        # Variable does not exist in testSamples (list argument 1 invalid 1 valid)
        self.assertRaises(KeyError, test_samples.get_ranges, ['invalidVar', ls_variables[0]])

        # Invalid type
        self.assertRaises(TypeError, test_samples.get_ranges, 123)

        # Invalid type in list
        self.assertRaises(TypeError, test_samples.get_ranges, [123])

    def test_getVariable_valid(self):
        test_samples = uqp.sampling.composite_samples.Samples()

        test_samples.set_continuous_variable('test1', -1.1, 0.1, 1.1)
        test_samples.set_continuous_variable('test2', -2.3, .3, 3.2)

        actual_variable1 = uqp.sampling.composite_samples.ContinuousVariable('test1', -1.1, 0.1, 1.1,
                                                                         None, None, None, None)

        actual_variable2 = uqp.sampling.composite_samples.ContinuousVariable('test2', -2.3, .3, 3.2,
                                                                         None, None, None, None)

        np.testing.assert_array_equal(test_samples.get_variables(), [actual_variable1, actual_variable2])
        np.testing.assert_array_equal(test_samples.get_variables(['test2', 'test1']),
                                      [actual_variable2, actual_variable1])

        np.testing.assert_array_equal(test_samples.get_variables(['test1']), [actual_variable1])
        np.testing.assert_array_equal(test_samples.get_variables('test1'), [actual_variable1])

        np.testing.assert_array_equal(test_samples.get_variables(['test2']), [actual_variable2])
        np.testing.assert_array_equal(test_samples.get_variables('test2'), [actual_variable2])

    def test_getVariable_invalid(self):
        test_samples = uqp.sampling.composite_samples.Samples()
        test_sampler = uqp.sampling.sampler.LatinHyperCubeSampler()

        i_num_points = 100
        ls_box = [[-1, 1], [-2, 3], [5.5, 6.7]]
        i_seed = 2016

        ls_variables = ['var1', 'var2', 'var3']
        for var, rng in zip(ls_variables, ls_box):
            test_samples.set_continuous_variable(var, rng[0], 0, rng[1])
        test_samples.generate_samples(ls_variables, test_sampler, num_points=i_num_points, seed=i_seed)

        # Variable does not exist in testSamples (string argument)
        self.assertRaises(KeyError, test_samples.get_variables, 'invalidVar')

        # Variable does not exist in testSamples (list argument len 1)
        self.assertRaises(KeyError, test_samples.get_variables, ['invalidVar'])

        # Variable does not exist in testSamples (list argument len 2)
        self.assertRaises(KeyError, test_samples.get_variables, ['invalidVar1', 'invalidVar2'])

        # Variable does not exist in testSamples (list argument 1 invalid 1 valid)
        self.assertRaises(KeyError, test_samples.get_variables, ['invalidVar', ls_variables[0]])

        # Invalid type
        self.assertRaises(TypeError, test_samples.get_variables, 123)

        # Invalid type in list
        self.assertRaises(TypeError, test_samples.get_variables, [123])


#
# ====================================================================
#                                 main
# ====================================================================
#
#  Run the unit tests.
#
if __name__ == '__main__':
    unittest.main()
