#! /usr/bin/env python

# Copyright (c) 2004 - 2018, Lawrence Livermore National Security, LLC.
# All rights reserved.

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import csv
import itertools

import numpy as np

from sampling_methods.sampler import ContinuousSampler, DiscreteSampler, DiscreteOrderedSampler


def parse_file(file_name, file_type='tab'):
    """
    Parses a data file into a Samples object.

    Parses a tab or csv file into a Samples object, making a number of assumptions. A column containing values which can
    be parsed into a float will be a continuous variable. The lower and upper extremes will define its range and the
    median will be its default value. Identity scaling will be assumed. A column that cannot be parsed into a float will
    be a discrete variable. The unique values found will represent the set of possible values.

    Args:
        file_name (string): The name of the file to parse.
        file_type (string): The type of parsing to perform. Currently only .tab and .csv are supported.

    Returns (Samples):
        A Samples object containing the points from the file.
    """
    if file_type.lower() == 'tab':

        with open(file_name, 'r') as file_handle:

            line = next(file_handle)
            line_num = 1
            while line.split()[0] == '##':
                line = next(file_handle)
                line_num += 1

            if line.split()[0] == '#':
                feature_names = line.split()[1:]

                feature_values = []
                for _ in range(len(feature_names)):
                    feature_values.append([])

            else:
                raise ValueError('Parsing Error on line {}: Header must come before data'.format(line_num))

            for line, line_num in zip(file_handle, itertools.count(line_num + 1)):
                tokens = line.split()

                if tokens[0] == '##':  # Treat as comment. Ignore
                    continue

                elif tokens[0] == '#':  # Header line. Error
                    raise ValueError(
                        'Parsing Error on line {}: Cannot have more that one header in line'.format(line_num))

                else:  # Values
                    if len(tokens) != len(feature_names):
                        raise ValueError(
                            'Parsing Error on line {}: Number of values does not match number of features'.format(
                                line_num))
                    for value, token in zip(feature_values, tokens):
                        value.append(token)

    elif file_type.lower() == 'csv':
        with open(file_name, 'r') as file_handle:
            reader = csv.reader(file_handle)

            feature_names = next(reader)

            feature_values = []
            for _ in range(len(feature_names)):
                feature_values.append([])

            for line_num, line in enumerate(reader):
                if len(line) != len(feature_values):
                    raise ValueError(
                        'Parsing Error on line {}: Number of values does not match number of features'.format(
                            line_num + 2))
                for value, token in zip(feature_values, line):
                    value.append(token)
    else:
        raise TypeError('Can only parse tab or csv file')

    samples_obj = Samples()

    for name, values in zip(feature_names, feature_values):
        try:
            np_values = np.array(list(map(int, values)))
            samples_obj.set_discrete_ordered_variable(name, np.unique(np_values), np.unique(np_values)[0])
            samples_obj.dt_variables[name].np_points = np_values
        except ValueError:
            try:
                np_values = np.array(list(map(float, values)))
                low, default, high = np_values.min(), np.median(np_values), np_values.max()
                samples_obj.set_continuous_variable(name, low, default, high)
                samples_obj.dt_variables[name].np_points = np_values
            except ValueError:
                np_values = np.array(values)
                samples_obj.set_discrete_ordered_variable(name, np.unique(np_values), np.unique(np_values)[0])
                samples_obj.dt_variables[name].np_points = np_values

    return samples_obj


class Variable(object):
    """Represents a variable with a set of sample points."""

    def __init__(self, str_name):
        """
        Initialize a Variable with given name and empty points

        Args:
            str_name (string): The name of the Variable
        """
        self.str_name = str_name
        self.np_points = np.array([])

    def __len__(self):
        """Return the number of sample points"""
        return len(self.np_points)

    def __eq__(self, other):
        return (self.str_name == other.str_name) and \
               (self.np_points.shape == other.np_points.shape) and \
               (self.np_points == other.np_points).all()

    def __repr__(self):
        return "Variable '{}': {} points".format(self.str_name, len(self))

    def __str__(self):
        return self.__repr__()


class ContinuousVariable(Variable):
    def __init__(self, str_name, f_low, f_default, f_high,
                 scaling_type, f_scaled_low, f_scaled_default, f_scaled_high):
        """
        Initialize a ContinuousVariable with given name and range parameters

        Args:
            str_name (string): The name of the variable
            f_low (float): The lower limit of the variable
            f_default (float): The default value of the variable
            f_high (float): The upper limit of the variable
            scaling_type (string): Scaling function type (log, lin, or some other)
            f_scaled_low (float): The scaled lower limit
            f_scaled_default (float): The scaled default
            f_scaled_high (float): The scaled upper limit
        """

        super(ContinuousVariable, self).__init__(str_name)

        self.ls_range = [f_low, f_high]
        self.f_default = f_default

        self.ls_scaled_range = [f_scaled_low, f_scaled_high]
        self.f_scaled_default = f_scaled_default

        self.scaling_type = scaling_type

    def __eq__(self, other):
        if isinstance(other, ContinuousVariable):
            return (Variable.__eq__(self, other)) and \
                   (self.ls_range == other.ls_range) and \
                   (self.f_default == other.f_default) and \
                   (self.ls_scaled_range == other.ls_scaled_range) and \
                   (self.f_scaled_default == other.f_scaled_default) and \
                   (self.scaling_type == other.scaling_type)
        else:
            return False

    def __repr__(self):
        return "Continuous" + super(ContinuousVariable, self).__repr__() + ", range: {}, default: {}".format(
            self.ls_range,
            self.f_default)


class DiscreteVariable(Variable):
    def __init__(self, str_name, ls_values, default):
        super(DiscreteVariable, self).__init__(str_name)
        self.ls_values = list(ls_values)
        self.default = default

    def __eq__(self, other):
        if isinstance(other, DiscreteVariable):
            return (Variable.__eq__(self, other)) and \
                   (self.ls_values == other.ls_values)
        else:
            return False

    def __repr__(self):
        return "Discrete" + super(DiscreteVariable, self).__repr__() + ", values: {}, default: {}".format(
            self.ls_values,
            self.default)


class DiscreteOrderedVariable(DiscreteVariable):
    # def __init__(self, str_name, ls_values):
    #     ls_values = sorted(list(ls_values))
    #     super(DiscreteOrderedVariable, self).__init__(str_name, ls_values)
    pass


class Samples(object):
    """A composite class for generating and collating sample points"""

    def __init__(self):
        self.dt_variables = {}
        self.ls_variables_order = []

    def __iter__(self):
        index = 0
        while True:
            try:
                val = {var: self.get_points(var)[index][0] for var in self.ls_variables_order}
                index += 1
            except IndexError:
                return
            else:
                yield val

    def __repr__(self):
        return '\n'.join([str(var) for var in self.get_variables()])

    def __str__(self):
        return self.__repr__()

    def generate_samples(self, variables, sampler, **kwargs):
        """
        Creates a set of sample points based on variables and sampler given.

        Sample points are stored in each individual variable.

        Args:
            variables ([string]): The list of variables to add sample points to
            sampler (sampling.Sampler): The sampler to generate sample points
            **kwargs: Set of keywords to be passed to the sampler

        Raises:
            KeyError
            TypeError
        """
        if not isinstance(sampler, ContinuousSampler) and not isinstance(sampler, DiscreteSampler):
            raise ValueError("Sampler type must be at least one of {}".format([ContinuousSampler,
                                                                               DiscreteSampler]))

        for var, i in zip(variables, range(len(variables))):  # For each given variable
            if isinstance(var, str):  # Make sure each variable is actually a string
                if var not in self.dt_variables.keys():  # Make sure variable exists in added variables
                    raise KeyError("Variable \'{}\' does not exist in variables".format(var))
            else:
                raise TypeError("Variable name must be a string. Was given \'{}\'".format(type(var)))

        ls_variable_objects = [self.dt_variables[x] for x in variables]

        if not isinstance(sampler, DiscreteOrderedSampler) and np.array([isinstance(x, DiscreteOrderedVariable)
                                                                         for x in ls_variable_objects]).any():
            raise TypeError("Must use ordered discrete sampler on ordered discrete variable")

        if not isinstance(sampler, DiscreteSampler) and np.array([isinstance(x, DiscreteVariable)
                                                                  for x in ls_variable_objects]).any():
            raise TypeError("Must use discrete sampler on discrete variable")

        if not isinstance(sampler, ContinuousSampler) and np.array([isinstance(x, ContinuousVariable)
                                                                    for x in ls_variable_objects]).any():
            raise TypeError("Must use continuous sampler on continuous variable")

        ls_box = []
        ls_default = []
        ls_values = []

        for var, i in zip(ls_variable_objects, range(len(ls_variable_objects))):
            if isinstance(var, ContinuousVariable):
                ls_box.append(var.ls_range)
                ls_default.append(var.f_default)
                ls_values.append([])
            elif isinstance(var, DiscreteVariable):
                ls_box.append([])
                ls_default.append(None)
                ls_values.append(var.ls_values)
            else:
                raise ValueError("Variable was not of type \'ContinuousVariable\' nor \'DiscreteVariable\'")

        if 'box' in kwargs:
            ls_box = kwargs['box']
            kwargs.pop('box')

        # Generate sample points
        np_points = sampler.sample_points(box=ls_box, default=ls_default,
                                          values=ls_values, variables=variables, **kwargs).T

        for var, i in zip(variables, range(len(variables))):  # For each given variable
            if isinstance(var, str):  # Make sure each variable is actually a string
                if var in self.dt_variables.keys():  # Make sure variable exists in added variables

                    # Add generated points to respective internal variable
                    self.dt_variables[var].np_points = np.append(self.dt_variables[var].np_points, np_points[i])
                else:
                    raise KeyError("Variable \'{}\' does not exist variables".format(var))
            else:
                raise TypeError("Variable name must be a string. Was given \'{}\'".format(type(var)))

    def fill_unsampled(self):
        """
        Fills in variables with default points

        Fills in any variable that is not long enough with the default point. This applies to variables who's length
            is less than that of the longest variable. The default point for that variable will be appended to the end
            enough times to make the variable long enough

        """
        i_max_length = max([len(var) for var in self.dt_variables.values()])
        for obj_var in self.dt_variables.values():
            i_diff = i_max_length - len(obj_var)
            if isinstance(obj_var, ContinuousVariable):
                obj_var.np_points = np.append(obj_var.np_points, [obj_var.f_default] * i_diff)
            elif isinstance(obj_var, DiscreteVariable):
                obj_var.np_points = np.append(obj_var.np_points, [obj_var.default] * i_diff)

    def set_continuous_variable(self, str_name, f_low, f_default, f_high,
                                str_scale_type=None, f_scaled_low=None, f_scaled_default=None, f_scaled_high=None):
        """
        Add new continuous variable to set of variables.

        Adds a new continuous variable to the set of variables. If the variable already exists in the set of variables,
            then that variable is changed to the new value.

        Args:
            str_name (string): The name of the variable
            f_low (float): The lower limit of the variable
            f_default (float): The default value of the variable
            f_high (float): The upper limit of the variable
            str_scale_type (string): Scaling function type (log, lin, or some other)
            f_scaled_low (float): Scaled low value
            f_scaled_default (float): Scaled default value
            f_scaled_high (float): Scaled high value
        """

        obj_new_var = ContinuousVariable(str_name, f_low, f_default, f_high,
                                         str_scale_type, f_scaled_low, f_scaled_default, f_scaled_high)
        self.dt_variables[str_name] = obj_new_var
        if str_name not in self.ls_variables_order:
            self.ls_variables_order.append(str_name)

    def set_discrete_variable(self, str_name, ls_values, default):
        """
        Add new discrete variable to set of variables.

        Adds a new discrete variable to the set of variables. If the variable already exists in the set of variables,
            then that variable is changed to the new value.

        Args:
            str_name (string): The name of the variable
            ls_values ([~]): The list of values the variable can take. Duplicates will be removed.
            default (~): The default value of the variable
       """
        obj_new_var = DiscreteVariable(str_name, ls_values, default)
        self.dt_variables[str_name] = obj_new_var
        if str_name not in self.ls_variables_order:
            self.ls_variables_order.append(str_name)

    def set_discrete_ordered_variable(self, str_name, ls_values, default):
        """
        Add new discrete variable to set of variables.

        Adds a new discrete variable to the set of variables. If the variable already exists in the set of variables,
            then that variable is changed to the new value.

        Args:
            str_name (string): The name of the variable
            ls_values ([~]): The list of values the variable can take. Duplicates will be removed.
            default (~): The default value of the variable
       """
        obj_new_var = DiscreteOrderedVariable(str_name, ls_values, default)
        self.dt_variables[str_name] = obj_new_var
        if str_name not in self.ls_variables_order:
            self.ls_variables_order.append(str_name)

    def config_from_dict(self, config, warn=True):
        """
        Configures Samples object from a dictionary
        """
        if 'variables' in config:
            for var_name, var_attr in config['variables'].items():
                if 'range' in var_attr:
                    if 'scaling' in var_attr:
                        if 'scaled_range' in var_attr:
                            self.set_continuous_variable(str_name=var_name,
                                                         f_low=var_attr['range'][0],
                                                         f_default=var_attr['default'],
                                                         f_high=var_attr['range'][1],
                                                         str_scale_type=var_attr['scaling'],
                                                         f_scaled_low=var_attr['scaled_range'][0],
                                                         f_scaled_high=var_attr['scaled_range'][1])
                        else:
                            raise Exception("'scaled_range' not found for {}".format(var_name))
                    else:
                        self.set_continuous_variable(str_name=var_name,
                                                     f_low=var_attr['range'][0],
                                                     f_default=var_attr['default'],
                                                     f_high=var_attr['range'][1])
                elif 'values' in var_attr:
                    if 'default' in var_attr:
                        self.set_discrete_variable(str_name=var_name,
                                                   ls_values=var_attr['values'],
                                                   default=var_attr['default'])
                    else:
                        self.set_discrete_variable(str_name=var_name, ls_values=var_attr['values'],
                                                   default=var_attr['default'])
                else:
                    raise Exception("Variable '{}' did not have either 'range' or 'values'".format(var_name))

        else:
            if warn:
                raise Warning("The 'variables' tag was not found in given dictionary")

    def config_to_dict(self):
        dt = {}
        for var_name in self.ls_variables_order:
            var = self.dt_variables[var_name]
            if isinstance(var, ContinuousVariable):
                dt[var_name] = {'range': var.ls_range, 'default': var.f_default}
                if var.scaling_type is not None:
                    dt[var_name]['scaling'] = var.scaling_type
                    dt[var_name]['scaled_range'] = var.ls_scaled_range

            elif isinstance(var, DiscreteVariable):
                dt[var_name] = {'values': var.ls_values, 'default': var.default}
            else:
                raise Exception('Variable "{}" was neither Continuous nor Discrete'.format(var_name))

        return {'variables':dt}

    def _get_attr(self, variable, func):

        def get_func(f):
            def wrapper(x):
                if isinstance(x, str):
                    if x in self.dt_variables.keys():
                        return f(x)
                    else:
                        raise KeyError("Variable \'{}\' does not exist variables".format(x))
                else:
                    raise TypeError("Variable name must be a string. Was given \'{}\'".format(type(variable)))

            return wrapper

        func = get_func(func)

        if variable is None:
            return np.array([func(var) for var in self.ls_variables_order])

        elif isinstance(variable, list) or isinstance(variable, np.ndarray):
            return np.array([func(var) for var in variable])

        elif isinstance(variable, str):
            return np.array([func(variable)])

        else:
            raise TypeError("Variable name must be a string or list of strings."
                            "Was given \'{}\'".format(type(variable)))

    def get_points(self, variable=None, scaled=False):
        """
        Return the sample points from given variables.

        Args:
            variable (string, [string]): Variables to get sample points from
            scaled (bool): Whether to return scaled values

        Returns (numpy array):
            Sample points from given variables
        """

        def linear_scaling(val, lo, hi, n_lo, n_hi):
            return n_lo + (val - lo) * (n_hi - n_lo) / (hi - lo)

        def get_points_helper(x):
            var = self.dt_variables[x]
            if scaled:
                if isinstance(var, ContinuousVariable):
                    if var.scaling_type is None:
                        return var.np_points
                    elif isinstance(var.scaling_type, str):
                        if var.scaling_type[:3].lower() == 'lin':
                            return linear_scaling(var.np_points,
                                                  var.ls_range[0],
                                                  var.ls_range[1],
                                                  var.ls_scaled_range[0],
                                                  var.ls_scaled_range[1])

                        elif var.scaling_type[:3].lower() == 'log':

                            return linear_scaling(np.power(np.asarray(var.np_points, dtype=np.float), 10),
                                                  np.power(var.ls_range[0],10),
                                                  np.power(var.ls_range[1],10),
                                                  var.ls_scaled_range[0],
                                                  var.ls_scaled_range[1])

                        else:
                            raise Exception()
                    else:
                        return var.scaling_type(var.np_points)
                else:
                    return var.np_points
            else:
                return var.np_points

        arr = self._get_attr(variable, get_points_helper).T

        if len(arr.shape) != 2:
            raise Exception("Invalid shape. There is a variable that does not has not have enough points. "
                            "You may want to use 'fill_unsampled' to fill in variables and create a valid shape.")

        return arr

    def get_length(self, variable=None):
        """
        Return the lengths of the sample points for given variables.

        Args:
            variable (string, [string]): Variables to get sample points from

        Returns (numpy array):
            The lengths of the sample points for the given variables
        """

        return self._get_attr(variable, lambda x: len(self.dt_variables[x]))

    def get_ranges(self, variable=None, scaled=False):
        """
        Return the ranges of sample points for given variables.

        Args:
            variable (string, [string]): Variables to get ranges from
            scaled (bool): Whether to return scaled range

        Returns (numpy array):
            The ranges for the given variables
        """

        def get_ranges_helper(x):
            var = self.dt_variables[x]
            if scaled and isinstance(var, ContinuousVariable) and var.scaling_type is not None:
                return np.array(var.ls_scaled_range)
            else:
                return np.array(var.ls_range)

        return self._get_attr(variable, get_ranges_helper)

    def get_variables(self, variable=None):
        """
        Return the variables as Variables Objects.

        Args:
            variable (string, [string]): Variables to get

        Returns [Variable]:
            A list of Variables
        """

        return self._get_attr(variable, lambda x: self.dt_variables[x])

    def get_variable_list(self):
        """
        Returns a list of the variable names in the order which they were added.

        Returns ([string]):
            The variable names in the order which they were added.
        """
        return self.ls_variables_order

    def write_csv(self, file_path, scaled=False):

        with open(file_path, 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            names = self.get_variable_list()
            points = self.get_points(scaled=scaled)
            csv_writer.writerow(names)
            for point in points:
                csv_writer.writerow(point)


if __name__ == "__main__":
    pass
