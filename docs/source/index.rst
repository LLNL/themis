Uncertainty Quantification Pipeline (UQP) Documentation
=======================================================

The UQP components interface is made up of the ``uqp`` and ``themis`` packages. The ``uqp`` package
is made up of three subpackages, each with their own API.
The three subpackages deal with sampling strategies on a parameter space (``uqp.sampling``);
generating surrogate models (``uqp.surrogate_model``); and performing uncertainty quantification (``uqp.uq_component``).

The :ref:`themis` component manages the execution of simulations.
Given a set of inputs (sample points) to run a simulation on,
this component will execute them in parallel, monitor their progress, and collect the results.

The :ref:`sampling_component` component is used to generate sample points in order to explore a parameter space.
For instance, if a simulation takes two inputs, x and y, and you want to run a set of simulations with x-values
between 5 and 20 and y-values between 0.1 and 1000, the sampling component can generate sample points (which in
this case means (x,y) pairs) for you. You can specify how many total sample points you want, and how you want
them to be chosen--the sampling component offers a large number of different sampling strategies.
If, on the other hand, you already have sample points you wish to use,
the component can simply read them in from a file.

The :ref:`surrogate_model_component` component is designed to be
used after a number of simulations have run to completion. This component can generate statistical
models which can be used to predict the results of future simulation runs, and to perform
sensitivity and uncertainty quantification analyses.

The ``uqp`` and ``themis`` packages work with Python 2 and 3.
On LC RZ and CZ systems, they are available at ``/collab/usr/gapps/uq/``.
On LANL's Trinitite, they are available at ``/usr/projects/packages/UQPipeline/``. Demo usage (on LC):

.. code:: python

	import sys
	sys.path.append("/collab/usr/gapps/uq")

	from uqp.sampling import composite_samples
	from themis import Themis
	from uqp.uq_component import uqp_mcmc

For detail on each component, click on the links in the navigation menu on the left.


.. toctree::
   :maxdepth: 2
   :Caption: Contents

   sampling_component
   themis/index
   uq_component


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`