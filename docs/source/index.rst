.. uqp documentation master file, created by
   sphinx-quickstart on Thu Oct 11 15:43:19 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the LLNL UQ Pipeline Documentation!
==============================================

The LLNL Uncertainty Quantification (UQ) Pipeline, or UQP, is a Python-based scientific workflow system for running and analyzing concurrent UQ simulations on high-performance computers.  Using a simple, non-intrusive interface to simulation models, it provides the following capabilities:

* generating parameter studies
* generating one-at-a-time parameter variation studies
* sampling high dimensional uncertainty spaces
* generating ensemble of simulations leveraging LC's HPC resources
* analyzing ensemble of simulations output
* constructing surrogate models
* performing sensitivity studies
* performing statistical inferences
* estimating parameter values and probability distributions

The pipeline has been used for simulations in the domains of Inertial Confinement Fusion, National Ignition Facility experiments, climate, as well as other programs and projects.

The pipeline is composed of the following capabilities:

* sampling
* ensemble manager
* uncertainty quantification methods

We have made these capabilities available in individual software components.

Components Interface
====================

The UQP components interface is made up of 4 packages
 * ``sampling_methods``
 * ``sensitivity_methods``
 * ``uq_methods``
 * ``themis``

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

The :ref:`sensitivity_component` component is component is also designed to be
used after a number of simulations have run to completion. This component is used to assess the sensitivity each
input parameter has on the output.

The :ref:`uncertainty_quantification` component is also designed to be
used after a number of simulations have run to completion. This component is used to predict the results of future simulation runs.

The ``uqp`` packages work with Python 2 and 3.
On LC RZ and CZ systems as an example, they are available at ``/collab/usr/gapps/uq/``.
On LANL's Trinitite, they are available at ``/usr/projects/packages/UQPipeline/``. Demo usage (on LC):

.. code:: python

        import sys
        sys.path.append("/collab/usr/gapps/uq")

        from sampling_methods import composite_samples
        from themis import Themis
        from uq_methods import uqp_mcmc

For detail on each component, click on the links in the navigation menu on the left.


.. toctree::
   :maxdepth: 2
   :hidden:

   themis/index
   sampling_component
   uq_component
   sensitivity/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`