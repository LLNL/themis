.. _themis_quickstart:

================
Quickstart Guide
================

.. seealso::
  The :ref:`ensemble_manager_examples` page, which goes over a handful of examples in detail.

General Concepts
==================

As stated on the landing page, a Themis ensemble consists of nothing more than running a single executable
a number of times while varying the parameters for each run. The parameters that can be varied,
and how it varies them, are up to you, the user; this guide will show you how to do it. First, however,
some general concepts that are essential to understanding how Themis works.

.. _ensemble_quickstart_run_dirs:

Run Directories
---------------
Before each run is launched by Themis, a new directory is created called the "run directory".
The run will be launched in that directory; since each run is given its own directory, it is less
likely to interfere with the files of other runs (imagine if all the runs wrote out a
file named "results.json").
If the run requires some supporting files, Themis can copy or link those files into the
run directory.

Usually, run directories are just given unhelpful names like "1", "2", "3", etc.; however,
they can (by means of an optional argument) be named however you like.

.. _ensemble_quickstart_samples:

Run Variations
--------------
There are three main types of parameters that Themis can vary across runs:

#.  The 'sample', which is a collection of variables. The variable names and their values are up to you.
    The purpose and use of a run's sample is covered in detail :ref:`below <text_replacement>`.
#.  The resources for your application, such as the number of MPI tasks to launch.
    The full list of options are: MPI tasks, cores per MPI task, GPUs per MPI task, and the timeout
    (rarely used, timeout represents the maximum runtime, in minutes, for the run). Usage is covered in detail
    :ref:`below <themis_resources>`.
#.  The command-line arguments to pass to your application, like ``--input /path/to/file.xml --verbose``.
    This one is pretty straightforward---it's just a single string.

In Python, all of the variation can be done through the ``themis.Run`` object,
which represents the sample, resources, and command-line arguments for
a single run of the ensemble. The sample is usually represented as a dictionary,
e.g. ``{'viscocity': 45.8, 'hydrostatics': 1740}``
(this means that the variable "viscocity" is assigned the value of 45.8, and "hydrostatics" a value of 1740).

On the command-line interface, each run is represented as a row in a CSV. Generally,
each column in the CSV is assumed to represent a single variable in the samples, and the
resources and command-line arguments are assumed to be constant across all runs. However, if
each run requires different resources or command-line arguments, they can be put into the CSV if the
``--vary-all`` flag is set.

Themis Ensemble-Wide Parameters
-------------------------------
Themis supports a few ensemble-wide parameters that modify the behavior of the ensemble
at a general level. You can specify, for instance, a list of files to hard-copy into each
run directory; a template string that yields the names of the run directories; or the
maximum number of times to restart a run upon failure.

.. _themis_setup_directory:

Themis's Setup Directory
------------------------
When you create a new Themis ensemble, Themis creates and populates a directory,
the "setup directory." Themis uses this directory to store all its data and
configuration information. By default
Themis will create the setup directory in ``./.themis_setup``, and whenever
you invoke Themis, it will look for the directory there. You can tell Themis to
create the directory wherever you want; however, when you want to interact with
a particular Themis ensemble, you will need to specify the path to the setup
directory for that ensemble.

Themis stores its logs in the setup directory; any file ending in ``.log`` may
be helpful if you something goes wrong with your ensemble.

It is generally recommended that you do not modify any of the files in the setup
directory. However, when you are done with an ensemble or want to start fresh,
feel free to delete the setup directory entirely.

.. _text_replacement:

Samples and Text Replacement
============================
Themis uses its samples to find specific tokens
(or variables if you prefer) in text files and replace them with
their value. By default, Themis will only perform this find-and-replace on the application itself
(if it is a batch script); however,
you can direct Themis to run its find-and-replace routine on other text files with an optional
argument, ``run_parse`` in Python or ``--parse`` in the CLI, which takes a sequence of one or more paths
to UTF-8 text files (in any format: xml, json, a python script, ...).


An example will make this much clearer. Suppose the names of the parameters I wish
to vary are 'viscocity' and 'hydrostatics'.
Then one of the ``run_parse`` files I pass to Themis might be a file looking like this:

.. code:: python

  hydrokinetics: 25000

  viscocity: %%viscocity%%
  hydrostatics: %%hydrostatics%%

  pressure: 18

The double percent characters ("%%") indicate to Themis that this
is a variable whose value should be replaced.
The ensemble manager looks at the sample for the current run, and if it finds a 'viscocity'
entry and a 'hydrostatics' entry, it will replace those variables with their values.
So, for instance, suppose Themis is executing run #57, and the sample point for run #57
looks something like ``{'viscocity': 45.8, 'hydrostatics': 1740}``.
Then the input deck will be copied (leaving the original unchanged),
then parsed and converted to the following before run #57:

.. code:: python

  hydrokinetics: 25000

  viscocity: 45.8
  hydrostatics: 1740

  pressure: 18

Of course, in order for this change to make any difference to the behavior of your application,
your application needs to actually read this file. How that is achieved is up to you.
It might be, for instance, that the input deck has a fixed name your application recognizes,
or perhaps the path to the input deck is accepted as a command-line argument.

Suppose, however, that the application also takes the path to the input deck as
a command-line argument named ``--setup``. Then you might launch the ensemble like so:

.. code:: python

  import themis

  input_deck = "/path/to/my/input/deck.txt"
  args = "--setup " + os.path.basename(input_deck)
  # assume the variable `samples` is already defined
  runs = [
    themis.Run(sample, args)
    for sample in samples
  ]

  mgr = themis.Themis.create("/path/to/my/application", runs=runs, run_parse=input_deck)
  mgr.execute_alloc()  # describe the batch allocation here

The ``os.path.basename`` function (which, in this case, would return just ``deck.txt``)
is used because, for each run,
the input deck will be parsed and copied into the same directory that the application is executed in.
Therefore, each run will pick up the correct version of the input deck---not the original one,
stored at ``/path/to/my/input/deck.txt``, which would still contain the "%%" characters.


Advanced Text Replacement with ``jinja``
----------------------------------------
If `jinja <https://palletsprojects.com/p/jinja/>`_ is installed
in the current virtual environment, Themis will use jinja as its text-replacement
engine instead of its (very basic) native support. This allows you to write things like
``%% my_mapping['my_key'].my_attribute %%`` and much more. See the
`jinja documentation <https://jinja.palletsprojects.com>`_ for more. Themis uses
most of the default ``jinja`` settings, except that the "print statement" is written
with "%%" instead of "{{" and "}}".


.. _themis_resources:

Allocating and Varying Resources
================================
Themis needs to know the resource requirements of your application in order to launch it correctly, and
to effectively manage the entire set of resources available to Themis.

In the examples above, we have implicitly assumed that the application Themis will execute has no
special resource requirements---that it isn't an MPI program, it isn't massively multithreaded,
and it isn't going to use GPUs. That assumption is not usually valid for HPC applications.
In order to specify resource requirements for a run to Themis, the ``themis.Run``
constructor takes some optional arguments. Suppose all of the runs of our ensemble should use 50 MPI tasks,
with 2 cores per task and 3 gpus per task. This can be specified as follows:

.. code:: python

  import themis

  runs = [
    themis.Run(tasks=50, cores_per_task=2, gpus_per_task=3)
    for _ in range(10)
  ]

On the command line, this would look like ``$ themis create foo.exe params.csv -n50 -c2 -g3``.

The Double Meaning of Resources
-------------------------------
In many cases, the application passed to Themis is not an MPI executable, or going to use GPUs.
It is instead a script (often called a "batch script") that will *launch* an MPI/GPU application,
e.g. with ``srun``. In that case, Themis needs to know,
not the resource requirements for the script (which are usually trivial) *but the resource requirements
of the applications that the script will launch*. It needs to know this in order to effectively throttle
the number of scripts active at any one time, and to describe the script's resource
requirements to the resource manager.

.. seealso::

   The section that goes into detail about Themis's :ref:`two application types <batch_script_howto>`.

Varying Resources Across Runs
-----------------------------
Now that you know how to allocate the same number of resources to each run of an ensemble,
varying resources among the runs is simple, at least in Python.
Here is an example showing how to give each run twice as many MPI tasks as the last run:

.. code:: python

  import themis

  # assume the variable `samples` is already defined
  runs = [themis.Run(sample, tasks=2 ** i) for i, sample in enumerate(samples)]

The ensemble can now be launched as follows:

.. code:: python

  mgr = themis.Themis.create("/path/to/my/application", runs)
  mgr.execute_local()

On the command line, to vary resources across runs, you would need to put the resource
requirements for each run into the CSV defining the samples, and then add the
``--vary-all`` flag to the ``create`` or ``add`` subcommands.
For instance, you might add "tasks" and "gpus_per_task" columns to the CSV.
