.. _application_interface:

Application Interfaces
======================

.. note::
  It is *strongly recommended* that you use the :ref:`Themis runtime API <themis_runtime_api>`
  instead of defining an application interface. Defining an application interface
  makes things harder to read and debug.


Themis is a general-purpose tool, and it has no understanding of the
application it is executing. It does not know, for instance, what kind of results it produces,
or anything similar. To understand the application,
it relies on the user (you) writing interface code. How simple this is depends a great deal on your application, but
we hope it is relatively straightforward.

The interface code should take the form of a python module (a ".py" text file) which defines functions
with specific names. The path to the module should be passed to Themis when the ensemble is created;
Themis will import it, and at defined points during the ensemble, those functions will be called.
The interface functions are responsible for interfacing with the application--whether it is
preparing the application for a new run or collecting results from an existing run.

The functions which may be defined by an application interface (each of them is optional)
are described below. If you create an application interface, you may define as many or as few of these
as you want. The usefulness of each of the functions depends on your particular application and your intentions.
Each of these functions should be callable with no arguments.

Recognized Functions
----------------------------------

The ``prep_ensemble`` function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function is only ever called once. It is called as soon
as the ensemble starts on the batch allocation, and before it launches any runs.

This function might be used to build some common files that all of the runs will
share---for instance, generating a mesh. However, for most cases, you could generate
those files yourself on the login node or inside an allocation
before launching the ensemble instead of putting the same code in ``prep_ensemble``.

Runs will not begin to be executed until ``prep_ensemble`` returns; therefore, in the
interests of not wasting compute resources, ``prep_ensemble`` should execute quickly.

No value should be returned from ``prep_ensemble``; any return value will be ignored.

``prep_ensemble`` will be called only once. If an ensemble is restarted,
``prep_ensemble`` will not be called a second time.


The ``post_ensemble`` function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function is called after the entire ensemble is finished,
and all existing runs have completed.

Might be used to, for instance, clean up some files. Like ``prep_ensemble``, your
compute allocation will be almost entirely inactive while this function runs.
Speed is therefore in your best interest.

This function is generally called only once. However, should ``post_ensemble``
add new runs via the :code:`themis.Themis.add_runs` method, Themis
will complete those new runs, then call
``post_ensemble`` again. In this case, be careful you do not get into an infinite loop.

No value should be returned from ``post_ensemble``; any return value will be ignored.


The ``prep_run`` function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Called before each new application run starts.
This might be used to prepare an individual application instance
for execution.

When this function is called, the current working directory will be the
directory where this run of the application will be executed.


The ``post_run`` function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Called after each application run completes successfully (i.e.
with a returncode of 0). This function is meant to be used to
determine whether that run succeeded, and then (if it did succeed)
harvest output of interest.

If this function returns ``1e99``, the run will be restarted, or marked as a failure
if the restarts have been exhausted. Otherwise, the object returned will be stored
internally and made available through a :code:`themis.Themis` instance.
If you do not write a post_run function, Themis will not collect any results.

Any combination of built-in or standard library types
(int, float, str, dict, list, tuple, or decimal.Decimal, to name the most common)
can be returned and safely stored: the object will be returned to you unchanged.
The same guarantee cannot be made for custom types (e.g. instances of a class defined in your code).

When this function is called, the current working directory will be the
directory where this run of the application was executed.

.. _ensemble_user_utils:

The user_utils Module
------------------------------------------------------------------------------------
.. automodule:: themis.user_utils
    :members:

Examples
--------
These examples are meant to be simple, to illustrate how these functions work and
what can be done with them.

Adding new samples in the post_ensemble function:

.. code:: python

  import themis
  from themis import user_utils

  def new_sample(i):
      raise NotImplementedError("This part is up to you")

  def post_ensemble():
      """Add points until there are 50 total runs"""
      manager = user_utils.themis_handle()
      total_runs = manager.count_by_status()
      if total_runs < 50:
          manager.add_runs(
              [
                  themis.Run(new_sample(i), None, tasks=5)
                  for i in range(total_runs, total_runs + 5)
              ]
          )

Template
----------
A template application interface, to be filled in as you see fit.

.. code:: python

  from themis import user_utils


  def prep_ensemble():
    """Prepare ensemble-wide files or other persistent data."""
    pass


  def post_ensemble():
    """Shut down and clean up the ensemble."""
    pass


  def prep_run():
    """Prepare an application run for execution."""
    pass


  def post_run():
    """Finish an application run, determine whether it succeeded, and return results."""
    pass

Common Issues
-----------------

Global Variables
~~~~~~~~~~~~~~~~~~~
The behavior of any of the four functions documented here should not depend on global
variables set by another of the four functions. For instance, ``prep_run``
should **NOT** set a global variable that ``post_run`` checks. If you attempt to do this,
*it will not work*. The python process that calls ``prep_run`` will not be the same
as the one that calls ``post_run``; therefore, the global variable initialized by ``prep_run``
will be lost. The same is true of any of the other functions.
If you wish to pass some kind of state between the various application interface functions,
you must do it some other way (perhaps by writing to a file).

Imports
~~~~~~~~~~~
Because the application interface is imported by Themis,
:code:`sys.path` may not be laid out how you expect; imports that you think should work,
or that work when you use your application interface locally, may not work at all when
Themis uses it. The only real solution to this problem is to be experienced
with Python's import system. Knowing how to import a script or package at a specific absolute path
is very useful. Consult a UQP developer if the problems are persistent.

