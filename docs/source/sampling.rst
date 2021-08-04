Sampling
========

Introduction to Sampling in the LLNL UQPipeline
-----------------------------------------------

Sampling refers to the selection of values for parameters from a subset of :math:`N` dimensional space, where :math:`N` is the number of perturbed parameters.
In the context of running an ensemble, there is a one to one correspondence between a sampled point and ensemble run.
The individual values of the point are mapped to their corresponding parameter for that ensemble run.

Documentation on the sampling component is located here: :ref:`sampling_component:Sampling Component`.

Simple Strategies
-----------------

The LLNL UQPipeline offers a number of strategies for sampling that provide different coverages of parameter space and different utilities.
To specify a strategy, use the :code:`uqmd` dictionary's :code:`sampling_strategy` key.
This key should be given the value of a list of strategy dictionaries.
Each of these strategy dictionaries has a key called :code:`sampling_type` who's value is the particular strategy to use.
Additionally, each strategy will have a unique set of keys that can be given, some being required and some being optional.

Example: Simple Latin Hypercube
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   uqmd = {
       'sampling_strategy': [
           {'sampling_type': 'stdlhs', 'num_sample_points': 10}
       ]
   }

Composite Strategies
--------------------

The key :code:`sampling_strategy` takes values in the form of a list.
To sample with more than one strategy at a time, simply include the additional strategies in the list.


Example: Simple Composite Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   uqmd = {
       'sampling_strategy': [
           {'sampling_type': 'stdlhs', 'num_sample_points': 10},
           {'sampling_type': 'corners'}
       ]
   }

Complex Composite Strategies
----------------------------

The strategy dictionary may also have a key called :code:`variables` who's value is a list of the particular variables or parameters on which to perform the strategy.
If this key is omitted or given an empty list, then it is assumed that the strategy will be performed on all parameters.
Using this key, one can choose to sample on only a subset of the possible variables or parameters.
Thus, complex strategies can be created by sampling on different subsets of variables.

If, at the end of sampling, a parameter has fewer values sampled than the paramater with the most values, the default value will be filled in so that it has enough values.
The user should take care to keep track of how many values each parameter will have at the end of sampling.

Example: Latin Hypercube + Gaussian Composite Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   uqmd = {
       'sampling_strategy': [
           {'sampling_type': 'stdlhs', variables: ['x', 'y'], 'num_sample_points': 10},
           {'sampling_type': 'pdf', variables: ['z'], 'num_sample_points': 10, 'dist': 'norm'}
       ]
   }

.. _smp_vs_raw:

:code:`samplepoints` vs. :code:`rawsamplepoints`
------------------------------------------------
When giving points directly to the pipeline, there is a choice between using :code:`samplepoints` and :code:`rawsamplepoints`.
The difference between these strategies is in how they handle scaling.

When using :code:`samplepoints` the points given are assumed to be in the scale of the simulation code.
In other words, these points should not be scaled. They will be substituted into the input deck without any alterations.
To reflect this usage, a synonym for :code:`samplepoints` is :code:`sim_samplepoints`.

When using :code:`rawsamplepoints` the points given are assumed to be in the scale of a design of experiments.
In other words, these points should be scaled using the scaling function provided in params_to_use when substituted into the input deck.
This allows a user to create a design of experiments (a generic ensemble) which can be used in multiple different situation.
To reflect this usage, a synonym for :code:`rawsamplepoints` is :code:`doe_samplepoints` (short for design of experiments).

Table of Sampling Strategies
----------------------------


+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|Name                     |sampling_type            |Required Keys                                 |Optional Keys                                 |
+=========================+=========================+==============================================+==============================================+
|:ref:`stdlhs`            |:code:`stdlhs`           |- :code:`num_sample_points`/:code:`num_points`|- :code:`seed`                                |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`geolhs`            |:code:`geolhs`           |- :code:`num_sample_points`/:code:`num_points`|- :code:`seed`                                |
|                         |                         |- :code:`geo_degree`                          |                                              |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`montecarlo`        |:code:`montecarlo`       |                                              |                                              |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`quasi_rn`          |:code:`quasi_rn`         |- :code:`num_sample_points`/:code:`num_points`|- :code:`technique`                           |
|                         |                         |                                              |- :code:`at_most`                             |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`centered`          |:code:`centered`         |- :code:`num_sample_points`/:code:`num_points`|- :code:`technique`                           |
|                         |                         |                                              |- :code:`seed`                                |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`list`              |:code:`list`             |- :code:`list_type`                           |                                              |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`default_value`     |:code:`default_value`    |- :code:`num_sample_points`/:code:`num_points`|                                              |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`corners`           |:code:`corners`          |                                              |- :code:`num_sample_points`/:code:`num_points`|
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`uniform`           |:code:`uniform`          |- :code:`num_sample_points`/:code:`num_points`|- :code:`equal_area_divs`                     |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`multidim`          |:code:`multidim`         |- :code:`interval`/:code:`num_divisions`      |- :code:`equal_area_divs`                     |
|                         |:code:`cartesian_cross`  |                                              |                                              |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`samplepoints`      |:code:`samplepoints`     |- :code:`samples`                             |                                              |
|                         |:code:`sim_samplepoints` |                                              |                                              |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`rawsamplepoints`   |:code:`rawsamplepoints`  |- :code:`samples`                             |                                              |
|                         |:code:`doe_samplepoints` |                                              |                                              |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`pdf`               |:code:`pdf`              |- :code:`num_sample_points`/:code:`num_points`|- :code:`loc`                                 |
|                         |                         |- :code:`dist`                                |- :code:`scale`                               |
|                         |                         |                                              |- :code:`seed`                                |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`multi_normal`      |:code:`multi_normal`     |- :code:`num_sample_points`/:code:`num_points`|- :code:`mean`                                |
|                         |                         |                                              |- :code:`covariance`                          |
|                         |                         |                                              |- :code:`seed`                                |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`external`          |:code:`extmthd`          |- :code:`location`                            |                                              |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`user`              |:code:`user`             |- :code:`user_samples_file`                   |                                              |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`dace`              |:code:`dace`             |                                              |                                              |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`nond`              |:code:`nond`             |                                              |                                              |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+
|:ref:`moat`              |:code:`moat`             |                                              |                                              |
+-------------------------+-------------------------+----------------------------------------------+----------------------------------------------+

.. _stdlhs:

Standard Latin Hypercube
------------------------
:code:`stdlhs`

Creates a set of points in a random latin hypercube

Ranges in each dimension are divided in to N sub-intervals, where N is the number of points. For each
sample point, an interval is randomly selected from each dimension, and the point is selected uniformly
within the intersection of those intervals. Once a point has been sampled from an interval, that interval
can no longer be selected to be sampled from.

- Required Keys
    * :code:`num_sample_points`/:code:`num_points`: The number of sample points

- Optional Keys
    * :code:`seed`: Random seed

.. _geolhs:

Geometric Latin Hypercube
-------------------------

:code:`geolhs`

Creates a set of points in a random latin hypercube. See :ref:`stdlhs` for more details.

Intervals are bunched closer to the edges or the center depending on :code:`geo_degree`.

- Required Keys
    * :code:`num_sample_points`/:code:`num_points`: The number of sample points
    * :code:`geo_degree`: Determines how much to transform intervals. Values greater than 1 cause points to bunch near the edges. Points less than 1 cause points to bunch near the center.

- Optional Keys
    * :code:`seed`: Random seed

.. _montecarlo:

Monte Carlo
-----------

:code:`montecarlo`

Creates a set of points using standard uniform monte carlo.

Each dimension is sampled independently from a uniform distribution for each point.

- Required Keys
    * :code:`num_sample_points`/:code:`num_points`: The number of sample points

- Optional Keys
    * :code:`seed`: Random seed

.. _quasi_rn:

Quasi Random Number
-------------------

:code:`quasi_rn`

Creates a set of points using quasi-random numbers

Produces a quasi-random set of points that tends to evenly cover space. This strategy is deterministic and will always produce
the same sequence given the same inputs.

- Required Keys
    * :code:`num_sample_points`/:code:`num_points`: The number of sample points

- Optional Keys
    * :code:`technique`: Which type of sequence to use; either 'Sobol' or 'Halton'. Sobol is the default.
    * :code:`at_most`: A tolerance parameter for Halton Sequences. Default is 10000

.. _centered:

Centered
--------

:code:`centered`

Creates a set of points centered using a default point or generated points

Generates a line of points in across a single dimension, while holding the other dimensions constant at the
center point. Points in the varying-dimension will vary across the range in that dimension.

- Required Keys
    * :code:`num_sample_points`/:code:`num_points`: The number of sample points per dimension. If 2-tuple: First number is number of sample points per dimension, second number is number of latin hyper cube points to generate.

- Optional Keys
    * :code:`technique`: Whether to perform standard or latin hypercube centered sampling. Use 'lhs_vals' for latin hypercube centered sampling. Leave out otherwise. See :ref:`stdlhs` for more details.
    * :code:`seed`: Random seed for latin hypercube centered sampling

.. _list:

List
----

:code:`list`

Creates a set of points varying each dimension one at a time

Generates a a set of points with each dimension taking on its high and low values once, keeping all
other dimensions constant at the default point. Can also include point with all dimensions set at the high
value and the low value, as well as the default point itself.

- Required Keys
    * :code:`list_type`: The type of sampling to perform
        - :code:`listoat`: One at a Time with default
        - :code:`listd`: Only default
        - :code:`listldh`: Low, default, and high
        - :code:`listdh`: Default, and high
        - :code:`listlh`: Low, and high

.. _default_value:

Default Value
-------------

:code:`default_value`

Creates a set of default points.

Generates a set of N points simply repeating the default point.

- Required Keys
    * :code:`num_sample_points`/:code:`num_points`: The number of sample points

.. _corners:

Corners
-------

:code:`corners`

Creates a set of corner points.

Generates a set of points at the corners of the bounding box. If the number of points requested does not equal
the number of corners (2^N for N equal to the number of dimensions), then the sampler will return the number requested. If the
number of points is greater than 2^N, then the set of 2^N will be repeated until :code:`num_sample_points` have been given.

- Optional Keys
    * :code:`num_sample_points`/:code:`num_points`: The number of sample points

.. _uniform:

Uniform
-------

:code:`uniform`

Creates a set points in a line from the low corner to the high corner.

Each dimension is divided into N evenly spaced divisions with the points placed on the edges of these divisions. If 'equal_area_divs' is set
the points will be placed in the middle of the divisions instead of on the edges. The result is a line of
points from the corner of all low extents to the corner of all high extents.

- Required Keys
    * :code:`num_sample_points`/:code:`num_points`: The number of sample points

- Optional Keys
    * :code:`equal_area_divs`: Whether to place points in the center of the division areas or at the edges

.. _multidim:

Cartesian Cross
---------------

:code:`multidim`
:code:`cartesian_cross`

Creates a set of points that is the the Cartesian product of the given variables.

- Required Keys
    * :code:`interval`/:code:`num_divisions`: The number of divisions in each dimensions.
        - A single number implies that many divisions for each dimension.
        - A list of numbers implies that many divisions in each corresponding dimension.
        - A list of lists implies the exact values to use in each corresponding dimension.

- Optional Keys
    * :code:`equal_area_divs`: Whether to place points in the center of the division areas or at the edges

.. _samplepoints:

Sample Points
-------------

:code:`samplepoints`
:code:`sim_samplepoints`

Create a set of points using a given list of points.

Generates the points given from :code:`samples`. Generated points are given straight to the simulation.

See :ref:`smp_vs_raw` for more information on usage.

- Required Keys
    * :code:`samples`: The list of sample points to use

.. _rawsamplepoints:

Raw Sample Points
-----------------

:code:`rawsamplepoints`
:code:`doe_samplepoints`

Create a set of points using a given list of points.

Generates the points given from :code:`samples`. Generated points constitute a design of experiments.

See :ref:`smp_vs_raw` for more information on usage.

- Required Keys
    * :code:`samples`: The list of sample points to use

.. _pdf:

Probability Distribution
------------------------

:code:`pdf`

Creates a set of points from a probability distribution

Generates points from scipy's stat distributions. The parameters for location and scale
are passed directly to scipy's implementation. Each dimension will be sampled independently.

- Required Keys
    * :code:`num_sample_points`/:code:`num_points`: The number of sample points
    * :code:`dist`: Name of the distribution to use

- Optional
    * :code:`loc`: Location parameter. List of values will be sent to each dimension independently
    * :code:`scale`: Scale parameter. List of values will be sent to each dimension independently
    * :code:`seed`: Random seed

.. _multi_normal:

Multivariate Normal
-------------------

:code:`multi_normal`

Create a set of points from a multi-variate normal distribution

Generates points sampled from scipy's multivariate gaussian normal distribution.

- Required Keys
    * :code:`num_sample_points`/:code:`num_points`: The number of sample points

- Optional
    * :code:`mean`: N-dimensional vector of the distribution mean
    * :code:`covariance`: N-by-N symmetric positive semi-definite matrix of the distribution covariance
    * :code:`seed`: Random seed

.. _external:

External
--------

:code:`extmthd`

Creates a set of points using an externally defined method.

An external script that contains the method :code:`GenerateSamples(logger, uqi, uqcs)` is used to generate a set of samples.

- Required Keys
    * :code:`location`: The relative path to the external script

.. _user:

User
----

:code:`user`

Creates a set of points from a tab file.

Generates a set of points from an external tab file.

- Required Keys
    * :code:`user_samples_file`: The relative path to the external tab file

.. _dace:

Design and Analysis of Computer Experiments
-------------------------------------------

:code:`dace`

DAKOTA method

- Optional Keys:
    * See `Dakota manual <https://dakota.sandia.gov/content/latest-reference-manual>`_ for info on keys.

.. _nond:

NOND
----

:code:`nond`

DAKOTA method

- Optional Keys:
    * See `Dakota manual <https://dakota.sandia.gov/content/latest-reference-manual>`_ for info on keys.

.. _moat:

Morris One-at-a-Time
--------------------

:code:`moat`

DAKOTA method

- Optional Keys:
    * See `Dakota manual <https://dakota.sandia.gov/content/latest-reference-manual>`_ for info on keys.
