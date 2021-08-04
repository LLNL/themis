Tutorial
========

Sampler
-------

Samplers are used to create simple sample sets. The method :code:`sample_points()` takes a set of arguments and returns
a numpy array. Each sampler will have its own implementation, and so will take a unique set of arguments.

**Example**

To create a set of 10 points using the latin hypercube strategy in the unit square:

.. code:: python

   from uqp.sampling.sampler import LatinHyperCubeSampler
   points = LatinHyperCubeSampler.sample_points(num_points=10, box=[[0, 1], [0, 1]])

Full description of samplers and their arguments can be found in this `notebook <_static/Demostration_of_Samplers.html>`_.

Composite Samples
-----------------

A Composite Samples object makes it easier to generate more complicated sample sets. With it, you can specify a set of
variables and sample using those variables multiple times.

A full tutorial of the Sampling component can be found in this `notebook <_static/Sampling_Documentation.html>`_.

Examples
--------

Two Continuous Variables
~~~~~~~~~~~~~~~~~~~~~~~~

We want to sample over 2 continuous variables, :code:`X` and :code:`Y`.
A typical sampling strategy is to look at the corners, the faces, and some points in the interior.
We'll let :code:`X` have a range [0.1, 0.2] and :code:`Y` have a range [1.5, 2.5].
We must also decide on a default value for each variable. We'll let the default be the midpoint of the range.

To start, we must create a :code:`composite_samples` object to hold our sample points.

.. code:: python

   samples = composite_samples.Samples()

Next, we can define our two variables in :code:`samples`.

.. code:: python

   samples.set_continuous_variable('X', 0.1, 0.15, 0.2)
   samples.set_continuous_variable('Y', 1.5, 2.0, 2.5)

Now that the variables are defined in :code:`samples` we can start generating our sample set.
Each sampler implements a sampling strategy.
We need 3 samplers to generate our sample set: :code:`CornerSampler`, :code:`OneAtATimeSampler`, and :code:`LatinHyperCubeSampler`
We'll instantiate each of these so we can use them later.

.. code:: python

   corner_sampler = sampling.CornerSampler()
   face_sampler = sampling.OneAtATimeSampler()
   lhs_sampler = sampling.LatinHyperCubeSampler()

To actually generate the sample points we use the :code:`generate_samples` method.
This method takes a list of variable names and a sampler (with a set of keyword arguments).
It calls the sampler's :code:`sample_points` method using the keyword arguments and stores those points in the :code:`samples` object
under the specified variables.

.. code:: python

   samples.generate_samples(['X', 'Y'], corner_sampler)
   samples.generate_samples(['X', 'Y'], face_sampler, do_oat=True)
   samples.generate_samples(['X', 'Y'], lhs_sampler, num_points=10)

Now our sample set is stored in the :code:`samples` object.
To access these points we'll use :code:`get_points`.
This method simply returns all the points that have been generated.

.. code:: python

   sample_points = samples.get_points()

