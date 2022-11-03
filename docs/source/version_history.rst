Version History
===============


Version 1.2 (WIP)
-----------------

*	Added support for multi-step runs, i.e. runs that consist of multiple applications
	executed in sequence, rather than a single application. Added the
	``themis.CompositeRun`` and ``themis.Step`` classes to represent the new kind of runs
	in Python, and the corresponding ``create-composite`` and  ``add-composite``
	commands for shells.
	The ``Run`` class is now a subclass of ``CompositeRun``.
* 	The ``application`` argument to ``themis.Themis`` is now optional, and ignored
	when ``CompositeRun`` instances are passed in.
*	``themis.user_utils.run()`` and ``themis.runtime.fetch_run()`` now return
	``CompositeRun`` instances instead of ``Run`` instances.
* 	Removed the "shrink_on_repeat" argument to ``execute_alloc`` and its CLI
	equivalent, as it cannot be supported in its former implementation.
* 	Added a ``max_concurrency`` argument to the ``execute.*`` methods/commands,
	which tells Themis the maximum number of runs it can execute simultaneously
	in the given machine or allocation. Defaults to a reasonable value.
*	Added a ``allow_multiple`` argument to the ``execute.*`` methods/commands,
	which tells Themis to allow multiple Themis backends to be active simultaneously
	(i.e. multiple allocations cooperating on the same ensemble).
* 	Changed the ``display`` CLI command to account for steps, and made column widths
	individually customizable.
*	Added a ``hard`` option to the ``requeue`` and ``restart`` methods/commands, which
	tells Themis whether to reset the run back to step 0, or to continue execution from
	the most recent step.


Version 1.1 (May 2021)
-----------------------

*	Added the ability to abort a run if it exits with one of a set of
	user-specified return codes. For instance, if the returncode 50 is specified by
	a user as abort-triggering, and Themis is executing a Bash script
	which in turn executes the statement ``exit 50;`` Themis will abort that run.
	An aborted run will not be restarted by Themis. In Python this is the
	``abort_on`` argument to ``Themis.create(...)``, and in the CLI it is
	the ``--abort-on`` option to ``themis create``.
* 	Added the ability to execute a command as soon as every run in the ensemble has
	completed. If runs are added afterwards, the command will fire off again.
	In the CLI this is ``themis completion [command]``, in Python it is
	``Themis.on_completion(...)``.
* 	Added the ability to, when an allocation is expiring, request a *smaller*
	allocation than the current one. A smaller allocation will only be requested
	if there are too few remaining runs to fill an allocation of the current size.
	Python: ``Themis.execute_alloc(..., shrink_on_repeat=True)``. CLI:
	``themis execute-alloc ... --shrink-on-repeat``.


1.1 Command-Line Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^

* 	Added the ``themis runtime parse`` command for running Themis's token-replacement
	algorithm on files dynamically at runtime. This command can only be invoked by
	an application launched by Themis, since the tokens searched for are
	run-specific. It may prove useful if the files to parse are different for each run.
* 	Added the ``themis runtime collect`` command for reading the contents of a file
	and setting its contents (a bytestring) as the result of the current run.
	This command can only be invoked by an application launched by Themis.


1.1 Python Interface
^^^^^^^^^^^^^^^^^^^^

* 	Added the :ref:`themis.runtime <themis_runtime_api>` module for performing
	run-specific actions while Themis is running. This is a generalization of the
	:ref:`application interface <application_interface>` and it is strongly recommended
	that users transition away from the application interface model.
*	The ``Themis.run_dirs`` method now returns ``(run_id, directory_path)`` pairs
	instead of only the directory.


Version 1.0 (November 2020)
---------------------------
This entry in the version history shows the changes
from version 0.1, which was the first pass at the UQP's
ensemble manager component. Version 1.0 marks a major jump.
Themis is now its own stand-alone tool, and the Python interface has
been overhauled. There a handful of backwards-compatibility-breaking
changes to the command-line interface as well, but the changes are
much less substantial.

1.0 Command-Line Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^

*	The former ``execute`` and ``allocation`` methods have been transformed into
	``execute-alloc`` (which allows you to specify an allocation) and
	``execute-local`` (which does not).
* 	A new ``add`` command for adding runs to an existing ensemble. Most of the
	arguments are shared with ``create``.
* 	A new ``--vary-all`` flag to ``create`` which allows you to specify the
	resource requirements for each run individually, by putting that information
	into the parameter file (i.e. the CSV).
* 	The run statuses "pending" and "active" have been merged into "queued".
* 	Run IDs now start at 1 instead of 0.
* 	By default, the application used in the ensemble is now assumed to be a batch
	script. If it isn't, use the ``--no-batch-script`` flag to the ``create``
	command.
* 	The ``write`` command no longer takes a file path. Instead, it prints the result
	to stdout. Use stdout redirection to put the information into a particular file.
	(For instance, ``write json >& themis_data.json`` instead of
	``write json themis_data.json``.)
* 	A new ``--parallel`` option to ``execute-local`` and ``execute-alloc``, determining
	how much scheduler parallelism Themis should use. Basically, a higher number can
	yield better performance, but at the expense of resource fragmentation. The default
	is 0, which should be good enough for most multi-core applications that run for
	a handful of minutes or more.
* 	The default directory for storing Themis's setup files is now ``.themis_setup``
	instead of ``.ensemble_setup``.
*	Added the ``--early-stop N`` option to ``execute-alloc`` to stop Themis
	from submitting and monitoring runs when there
	are only *N* minutes left in an allocation.

1.0 Python Interface
^^^^^^^^^^^^^^^^^^^^

* 	Renamed package from ``uqp.ensemble`` to ``themis``.
* 	Most of the functionality of ``uqp.ensemble.manager`` is now found in ``themis``.
* 	The ``EnsembleManager``, ``EnsembleResults``, and ``EnsembleRestart`` classes
	have been combined into ``themis.Themis``. To create a new ensemble, use
	``themis.Themis.create()`` or a variant. To interact with an existing
	ensemble, use ``themis.Themis(path)``.
* 	The run statuses ``RUN_PENDING`` and ``RUN_ACTIVE`` have been
	merged into ``RUN_QUEUED``.
* 	Run IDs now start at 1 instead of 0.
* 	By default, the application used in the ensemble is now assumed to be a batch script.
*	The former ``execute`` and ``allocation`` methods have been transformed into
	``execute_alloc`` (which allows you to specify an allocation) and
	``execute_local`` (which does not).
*	Added the ``early_stop`` parameter to ``execute_alloc`` to stop Themis
	from submitting and monitoring runs when there
	are only some integer *N* minutes left in an allocation.
* 	The former ``write_*`` methods (yaml, csv, and json) now take a file-like object
	instead of a path string.
* 	A new ``parallelism`` option to the ``themis.Themis.execute_*`` methods, determining
	how much scheduler parallelism Themis should use. Basically, a higher number can
	yield better performance, but at the expense of resource fragmentation. The default
	is 0, which should be good enough for most multi-core applications that run for
	a handful of minutes or more.
* 	The default directory for storing Themis's setup files is now ``.themis_setup``
	instead of ``.ensemble_setup``.

1.0 Bug Fixes
^^^^^^^^^^^^^

*	Themis now runs on LANL's Trinity and Trinitite with no modifications,
	however the ``--parallel`` (CLI) and ``parallelism`` (Python) options must be
	left to the default setting.
