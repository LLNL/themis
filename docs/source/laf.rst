.. _laf:

====================================
Launching Batch Scripts Individually
====================================
The Themis ensemble manager is powerful, but it can also be complicated.
For users who just want to submit a batch script a handful of times with modifications, the
``themis.laf.BatchSubmitter`` may suffice.

This is sometimes referred to as "LAF" (launch and forget) mode.

Quickstart
============
The ``themis.laf.BatchSubmitter`` class launches ensembles by taking a single batch script, modifying it repeatedly,
and submitting each new version to the resource manager. So, for instance, if there are *N* runs
in your ensemble, ``BatchSubmitter`` will create *N* batch allocations, each running a slightly modified
version of the original batch script.

The ``BatchSubmitter`` class shares three things in common with the ``Themis`` class:
:ref:`samples <ensemble_quickstart_samples>`,
:ref:`run directories <ensemble_quickstart_run_dirs>`,
and :ref:`text replacement <text_replacement>`.

The sample of each run is used to modify the batch script; the ``BatchSubmitter`` class modifies the batch script by
parsing and replacing variables declared with "%%"---just as described :ref:`here <text_replacement>`.
For instance, you might have a sample like ``{"nodes": 5, "tasks": 120, "viscocity": 58.3}``,
and in your batch script you might have:

.. code-block:: none

  #SBATCH -N %%nodes%%
  [...]
  TASKS=%%tasks%%
  srun -n $TASKS path/to/my_application --visc=%%viscocity%%

Then your batch script will become, after parsing:

.. code-block:: none

  #SBATCH -N 5
  [...]
  TASKS=120
  srun -n $TASKS path/to/my_application --visc=120

The only way the ``BatchSubmitter`` class can modify runs is by parsing files. Themis, by contrast,
can also modify the resources and command-line arguments for each run. However, when you submit a
batch script to a resource manager, you cannot pass command-line arguments, and the resources are
usually defined inside the batch script itself (e.g. "#SBATCH -N 10").

General Usage
-------------
Suppose I have my batch script stored at ``/path/to/my/batch/script.sh``, and
I have defined the samples I wish to use. Let's assume it's a Slurm batch script. Then an ensemble
can be launched like so:

.. code:: python

  from themis import laf
  from trata.composite_samples import parse_file

  # assume the samples are defined in a csv file stored at "samples/my_samples.csv"
  samples = parse_file("samples/my_samples.csv", "csv")
  sub = laf.BatchSubmitter("/path/to/my/batch/script.sh", samples, "slurm")
  # no jobs are submitted until `.execute()` is called...
  batch_job_ids = sub.execute()
  print(batch_job_ids)

Pretty simple. All the ``BatchSubmitter`` class needs to know is the path to the batch script that
it will parse, the resource manager for the batch script, and the samples.

The same example, executed through the command-line interface, would look like this:

``python [...]/core/ensemble/laf.py my/batch/script.sh slurm samples/my_samples.csv``

API Reference
================================
The ``themis.laf.BatchSubmitter`` class has an interface similar to that of the
``themis.Themis``. Note, however, ``BatchSubmitter`` creates numerous allocations simultaneously,
while the ``Themis`` creates only one at a time. The
``themis.Themis`` class also supports
:ref:`running batch scripts in parallel <batch_script_howto>` inside
of a single allocation.

All that is needed to launch an ensemble with the ``BatchSubmitter`` class is:

#.  The path to the batch script you want to launch.
#.  The name of the resource manager you wish to submit it to (Slurm, LSF, Moab, ...).
#.  An iterable of dictionaries defining the samples for each run. Each dictionary
    corresponds to a new run.

Pass those three objects to the constructor, along with whatever optional arguments you like,
call the ``.execute()`` method, and your ensemble will begin.

.. autoclass:: themis.laf.BatchSubmitter
    :members:


Command-Line Interface
======================
The ``BatchSubmitter`` class can be used through a command-line interface.
Execute ``python [...]/uqp/core/ensemble/laf.py --help`` at the command line for usage information.


Examples
========
Now that the quickstart guide has gone over general usage information, and the API reference
has covered the meaning of each argument, the following
sections will go over simple, concrete examples in more detail.

Hello World
-----------
In this example, we'll launch
5 LSF batch scripts, each of which will ``echo`` a message and then quit.

The csv file containing the samples, stored at ``./my_samples.csv``:

.. code-block:: none

  message
  hello world
  hola mundo
  bonjour monde
  buongiorno mondo
  vale munde

The top row, "message," is the column header---not a sample itself.

Next, the batch script, stored at ``./script.sh``:

.. code-block:: bash

  #!/bin/bash
  #BSUB -q pdebug
  #BSUB -W 5
  #BSUB -nnodes 1

  lrun -n1 echo "%%message%%" > run.log


And lastly, the driver script for the ``BatchSubmitter``, stored at ``./driver.py``:

.. code:: python

  from themis import laf
  from trata.composite_samples import parse_file

  samples = parse_file("my_samples.csv", "csv")
  sub = laf.BatchSubmitter("script.sh", samples, "lsf")
  print(sub.execute())

Now, to execute:

.. code-block:: none

  $ python driver.py
  [71760, 71761, 71762, 71763, 71764]

Instead of writing a Python script, we could instead use the command-line interface. In that
case all we need to do is execute a single command:

.. code-block:: none

  $ python [...]/ensemble/laf.py script.sh lsf my_samples.csv
  Batch job IDs are: 71760, 71761, 71762, 71763, 71764

Examining the results, after completion:

.. code-block:: none

  $ ls
  driver.py my_samples.csv  runs    script.sh
  $ ls runs/
  0  1  2  3  4
  $ ls runs/0/
  run.log    script.sh
  $ cat runs/0/run.log
  hello world
  $ cat runs/4/run.log
  vale munde

Using the Optional Arguments
----------------------------

Naming Run Directories
^^^^^^^^^^^^^^^^^^^^^^
Now suppose we aren't happy with the ``runs/####`` naming scheme. Let's say we want them to be
named like ``languages/[insert language here]``. Then we first need to augment the ``samples.csv`` file:

.. code-block:: none

  language,message
  english,hello world
  spanish,hola mundo
  french,bonjour monde
  italian,buongiorno mondo
  latin,vale munde

Lastly, we need to inform the ``BatchSubmitter`` (in ``driver.py``) about how it should name the run directories,
using the optional ``run_dir_names`` argument:

.. code:: python

  sub = laf.BatchSubmitter(
    "script.sh",
    samples,
    "lsf",
    run_dir_names="languages/{language}"
  )

Now to execute, with either ``python driver.py`` or
``python -m themis.laf script.sh lsf my_samples.csv -r "languages/{language}"``.

Examining the results:

.. code-block:: none

  $ ls
  driver.py languages my_samples.csv  script.sh
  $ ls languages/
  english french  italian latin spanish
  $ cat languages/french/run.log
  bonjour monde

Symlinking and Copying Files Into the Run Directories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Suppose we want to populate each run directory with a couple of files:
``support/reqd_1.txt``, ``support/reqd_2.txt``, and ``support/reqd_3.txt``.

These files are read-only, so they don't need to be copied, just symlinked. We can specify them to the
``BatchSubmitter`` like so:

.. code:: python

  sub = laf.BatchSubmitter(
    "script.sh",
    samples,
    "lsf",
    run_symlink="support/reqd_*.txt"
  )

Or ``python -m themis.laf script.sh lsf my_samples.csv --symlink support/reqd_*.txt``.

This yields the following:

.. code-block:: none

  $ ls runs/1/
  reqd_1.txt  reqd_2.txt  reqd_3.txt  run.log   script.sh
  $ ls runs/3/
  reqd_1.txt  reqd_2.txt  reqd_3.txt  run.log   script.sh

If the files were not read-only (and so needed to be hard-copied rather than symlinked),
we would use ``run_copy`` in Python or ``--copy`` on the command line;
if they needed to be parsed for variables declared with "%%", then we would use
the ``run_parse`` argument or the ``--parse`` command-line option.
