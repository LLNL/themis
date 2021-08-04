====================================
FAQ, Tips, Tricks, and Miscellaneous
====================================

.. _batch_script_howto:

Themis's Two Application Types
-------------------------------
Themis supports running two kinds of applications: batch scripts and "regular" applications.
In theory they are both just executables, but in practice they are quite different.
Batch scripts are usually simple shell scripts, but (more importantly) they interact with the system's
resource manager to launch other applications, e.g. by calls to ``srun`` or ``flux``;
"regular" applications do not interact with the resource manager, at least not
to launch other applications. The class of "regular" applications includes most HPC
applications (i.e. highly-optimized, enormous, and compiled applications that use all
kinds of parallelism, like CUDA or other GPU languages/libraries, OpenMP, MPI).

Running HPC Applications With Themis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When you want Themis to execute an HPC application, or really any kind of application
except a batch script, the process is quite straightforward.
Pass the path to the application to Themis (e.g. "/usr/bin/ls"), specify its resource requirements
(e.g. 40 MPI tasks with 2 cores per task for OpenMP multithreading), and any command-line
arguments for the application. Then when you launch Themis it will execute your application
with the specified resources and command-line arguments.

You will also need to set ``app_is_batch_script=False`` argument when using Themis's Python
interface, or the ``--no-batch-script`` flag when using Themis from the command line.

Running Batch Scripts With Themis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When you want Themis to execute a batch script---again, a script that will make calls to
the resource manager, usually to launch a HPC application---the process is a little more
complicated. For the most part, however,
you should be able to use your batch script without any modifications. All that needs to be done is:

#.  Pass the path to your batch script to Themis as the ``application`` argument. Now the batch script is treated as the application;
    any command-line arguments you pass to the application are passed *to your batch script*,
    not to the underlying application(s).
#.  For the ``tasks``, ``cores_per_task``, ``gpus_per_task``, and ``timeout`` arguments to ``Run`` objects, specify the
    size and duration of the allocation that the batch script needs.
    For instance, if the batch script will launch a single application with 2 MPI tasks and 18 cores per task, specify that the
    batch script will need just that: ``tasks=2, cores_per_task=18``.
    If the batch script will launch one application with 36 cores, then one with 50 cores, and then one with 20 cores,
    specify that the batch script will need 50 cores---that is the maximum resource usage of the script.

Your application is now assumed to be a script, i.e. a text file. It will be parsed for variables declared with "%%".

Aside from these steps, there are some gotchas:

* The batch script should be careful about what resource-manager-specific variables it reads.
  For instance, one user's batch script checked the ``SLURM_JOB_NUM_NODES`` variable. This will lead to unexpected results, because the batch script will
  be sharing an allocation with other batch scripts. The number given by ``SLURM_JOB_NUM_NODES`` is the number of nodes available to the entire ensemble;
  unlike usual, it is *not* the number of nodes available to that batch script.

* For Slurm on LC machines, any ``srun`` commands should have the following arguments: ``--exclusive --mpibind=off``.
* For Slurm on LANL machines, any ``srun`` commands should have the following arguments: ``--exclusive --gres=craynetwork:0``
  (but see below).

The "themis_launch" Special Variable
------------------------------------
Dealing with the idiosyncracies of different resource managers is annoying.
To remove some part of the annoyance, Themis provides a special variable "themis_launch".
When you place "%%themis_launch%%" in your batch script, Themis will expand it into a resource-manager
specific launch command. On LANL machines running Slurm, for instance, it would be expanded into something like
``srun -n8 –exclusive –gres=craynetwork:0``. The ``-n8`` part comes from the resource
requirements you tell Themis about when you set up the ensemble.

For a full example, suppose we are running the application ``/usr/bin/echo``. Then our batch script
might look like

.. code:: none

    %%themis_launch%% /usr/bin/echo "this is my message" > my_file.txt

Then this might become, on a Flux system:

.. code:: none

    flux mini run -n3 -c2 /usr/bin/echo "this is my message" > my_file.txt

What Makes a Run Count as Failed?
----------------------------------
The mechanism Themis uses to determine whether a run has failed is simple,
but it doesn’t always do exactly what you want. When Themis launches your application,
whether it is a batch script or an MPI application, Themis waits for it to complete and then
checks the return code. Your application is an operating system process,
and when an OS process exits, it has to return an integer. Zero is success, everything else is failure.

So, to take it case by case, say Themis launches your batch script. If your batch
script returns 0, Themis marks it as a success and that’s it---all done.
If your batch script returns anything else, then one of two things can happen.
Themis accepts an ensemble-wide parameter (``max_restarts``) that tells it how many
times to restart individual runs. If the run hasn’t yet hit that restart maximum,
Themis will not mark the run as failed, but will launch it again.
Alternatively, if the run has hit the restart maximum, Themis marks the run as failed
and won’t touch the run again until you explicitly tell it to.
The default number of times that Themis will restart a run is 0.

However, if you are using an :ref:`application interface <application_interface>`, then things get a little more complicated.
If your ``prep_run`` or ``post_run`` functions raise an exception, Themis will mark
the run as failed.

.. _performance_info:

Performance
-------------
The overhead of using Themis is negligible for applications that run for more than a handful of seconds. Moreover, Themis
achieves strong scaling: suppose you want to run a 10,000-member ensemble, and suppose further that Themis can complete it
in *S* seconds when a single node is allocated. Then if 10 nodes are allocated, Themis will be able to complete it in *S/10* seconds.
For 100 nodes, *S/100* seconds, and so on.

Themis is regularly tested for its throughput on tens of thousands of single-core, split-second applications.
Themis is only 10% slower than the fastest the batch system could possibly run the applications.

For those curious about the internals, Themis launches a hierarchy of multithreaded workers
which are responsible for preparing, executing, monitoring, and completing individual runs.

The "concurrency" Argument
----------------------------
This section is only relevant to users of the old UQ pipeline ensemble manager.
The Themis ensemble manager determines "concurrency" (i.e. the number of applications it will execute at once)
solely based on the number of available resources. If the allocation has 16 cores available and your application
needs 7 cores, the ensemble manager will execute two instances of your application, then wait for additional
resources to free up before it launches any more.

.. _exporting_app_interface:

Exporting Information to the Application Interface
--------------------------------------------------
This section is only relevant to users of the old UQ pipeline ensemble manager.
The old ensemble manager had a method named ``setup_appl_dicts`` that could be used to pass information to the
application interface. Users passed dictionaries to setup_appl_dicts, and then collected them in their
application interface---it served as a simple way to export simple constants to the application interface.
This is no longer supported with Themis, but it's fairly easy to do it yourself. Here's
a recipe:

First, pick a file name. This file will be written before the ensemble starts, and read
by the application interface. You'll probably need to hard-code the file name, unfortunately.

.. code:: python

    import pickle

    def dump_app_interface_info(file_name):
        """Pickle an object and dump it to `file_name`."""
        with open(file_name, "w") as file_handle:
            pickle.dump(data, file_handle)

And in the application interface:

.. code:: python

    import os
    import pickle
    from themis import user_utils

    def get_app_interface_info(file_name):
        """Collect and return information pickled in `file_name`."""
        path_to_file = os.path.join(user_utils.root_dir(), file_name)
        if os.path.isfile(path_to_file):
            with open(path_to_file, "r") as file_handle:
                return pickle.load(file_handle)
        else:
            return None
