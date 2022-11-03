.. _ThemisPythonAPI:

====================
Python API Reference
====================

.. py:class:: themis.Themis

    The ``themis.Themis`` class can be used to launch ensembles with
    a single batch allocation. It is central to everything that can
    be done with the ``themis`` package.

Creating a New Ensemble
-----------------------
To create a new Themis ensemble, there are two necessary steps:

#.  Create an iterable of ``themis.CompositeRun`` or ``themis.Run`` objects
    to describe the runs of the ensemble.
#.  Call one of the ``Themis.create*`` methods listed below,
    passing it the iterable of ``CompositeRun``/``Run`` objects that you just created,
    and whatever optional arguments you choose. (If ``Run`` objects are passed, the
    ``application`` argument must be supplied as well.)

A directory should now have been created named
".themis_setup" (or, if you set the optional ``setup_dir`` argument in step 2, with
an arbitrary name and location). This directory will be needed in the future to
interact with the ensemble. See the :ref:`themis_setup_directory` section for more.

.. autoclass:: themis.Step
.. autoclass:: themis.CompositeRun
.. autoclass:: themis.Run

.. automethod:: themis.Themis.create
.. automethod:: themis.Themis.create_overwrite
.. automethod:: themis.Themis.create_resume


Interacting with an Existing Ensemble
-------------------------------------
Once an ensemble has been created,
you can interact with it by creating a new ``themis.Themis`` object, or
through a command-line interface. Either way, you may need the path to the
setup directory (the ``setup_dir``), which defaults to ".themis_setup".

.. automethod:: themis.Themis.__init__

Status Information
^^^^^^^^^^^^^^^^^^
Each run of a Themis ensemble can have one of four statuses: queued, which
is any run which has not yet completed; successful; failed; aborted, or killed.

These statuses are represented by five constants:
``themis.Themis.RUN_QUEUED``, ``themis.Themis.RUN_SUCCESS``,
``themis.Themis.RUN_FAILURE``, ``themis.Themis.RUN_ABORTED``,
and ``themis.Themis.RUN_KILLED``.

.. automethod:: themis.Themis.filter_by_status
.. automethod:: themis.Themis.count_by_status
.. automethod:: themis.Themis.progress

Ensemble Summaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automethod:: themis.Themis.write_csv
.. automethod:: themis.Themis.write_yaml
.. automethod:: themis.Themis.write_json
.. automethod:: themis.Themis.as_dataframe

Manipulating and Adding Runs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automethod:: themis.Themis.dequeue_runs
.. automethod:: themis.Themis.requeue_runs
.. automethod:: themis.Themis.add_runs

Miscellaneous
^^^^^^^^^^^^^
.. automethod:: themis.Themis.run_dirs
.. automethod:: themis.Themis.on_completion


Executing an Existing Ensemble
---------------------------------
Executing a Themis ensemble is as simple as calling either the ``execute_alloc`` or
``execute_local`` method. ``execute_local`` is used to execute on the current machine
or set of nodes without requesting a new allocation. To execute
within a new batch allocation, describe the desired allocation with
``themis.Allocation`` and pass that to ``mgr.execute_alloc``.

The ensemble should then begin running as soon as the allocation is acquired from
the machine's resource manager, or immediately if ``execute_local`` was used.

The ``dry_run`` function can be used to confirm that an ensemble is configured
properly without committing to execution.

.. automethod:: themis.Themis.dry_run
.. automethod:: themis.Themis.execute_local
.. automethod:: themis.Themis.execute_alloc


Utility Functions
-----------------
The following miscellaneous functions are designed to make using Themis easier.

Creating/Restarting Ensembles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To check if a directory contains a valid Themis ensemble, there is the
``exists`` function:

.. automethod:: themis.Themis.exists

And for removing a Themis ensemble:

.. automethod:: themis.Themis.clear

Debugging
^^^^^^^^^
The most difficult aspect of creating an ensemble is the
:ref:`application interface <application_interface>`. In order to
facilitate development and debugging, ``Themis`` exposes
the following methods.

.. automethod:: themis.Themis.call_post_run
.. automethod:: themis.Themis.call_post_ensemble
.. automethod:: themis.Themis.call_prep_ensemble
.. automethod:: themis.Themis.set_result
