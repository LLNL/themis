.. themis documentation master file, created by
   sphinx-quickstart on Thu Oct 11 15:43:19 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Themis Documentation!
====================================

The LLNL Themis package is a Python-based scientific workflow ensemble manager for running concurrent UQ simulations on high-performance computers. Using a simple, non-intrusive interface to simulation models, it provides the following capabilities:

* generating ensemble of simulations leveraging LC's HPC resources
* analyzing ensemble of simulations output

Themis has been used for simulations in the domains of Inertial Confinement Fusion, National Ignition Facility experiments, climate, as well as other programs and projects.

Themis
======

At the most general level, the Themis ensemble manager does nothing but run a single executable (usually a
batch script) a number of times while varying the input---the combination of the executable plus
its various sets of inputs is called an "ensemble." Through the API, a user can indicate to Themis
the executable to be run, how many times Themis should run it, and how to vary the input;
then Themis will begin running the executable with the various inputs.

More specifically, however, Themis is designed to perform that process efficiently with
executables (sometimes referred to as applications, here and throughout) designed for high-performance computers.
It is also meant to provide a simple and uniform interface
to the different resource managers (e.g. Slurm, LSF, Flux, ...) managing those high-performance computers.

It can be tedious to manually launch a HPC application more than a handful of times and collect results;
Themis is designed to automate that process for you.
It has the robustness to manage applications that take weeks and hundreds of nodes
to run, and it has the speed and scalability to complete :ref:`thousands of runs per second <performance_info>` with minimal overhead.
It also supports collecting results from completed runs, and dynamically
adding new runs as results become available (of interest for machine-learning-driven parameter studies).

For users who just want to launch a batch script a handful of times, and don't need all the power (and complexity) of the
ensemble manager, there :ref:`is an interface for that <laf>` as well.

New users should take a look at the :ref:`themis_quickstart` page for help on getting started;
the :ref:`ensemble_manager_examples` page might be helpful as well.

The ensemble manager supports any non-HPC Unix machine (including, e.g., Mac OSX laptops),
and any Unix HPC machine running Slurm, Flux, or IBM's LSF. It is, however, currently only installed and
tested on LLNL's HPC resources, as well as LANL's Trinitite.

Please reach out to a member of the UQP team if you would like the ensemble
manager to support a new HPC resource manager.

The ``themis`` package manages the execution of simulations.
Given a set of inputs (sample points) to run a simulation on,
this package will execute them in parallel, monitor their progress, and collect the results.

The ``themis`` package work with Python 2 and 3.
On LC RZ and CZ systems as an example, they are available at ``/collab/usr/gapps/uq/``.
On LANL's Trinitite, they are available at ``/usr/projects/packages/UQPipeline/``. Demo usage (on LC):

.. code:: python

        import sys
        sys.path.append("/collab/usr/gapps/uq")

        from themis import Themis


For detail on each component, click on the links in the navigation menu on the left.


.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart
   examples
   api
   runtime_api
   application_interface
   faq
   laf
   version_history


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`