"""
This package contains the Themis ensemble manager.

Themis is used to run an HPC application in parallel
inside of a single allocation. Usually this goes along with
varying the inputs of each run of the application.

The primary user interface to this package is contained in the
``themis`` module, which module provides the `Themis` class
for creating, restarting, and monitoring ensembles.

The ``themis.user_utils`` module provides an interface for passing
information back and forth between the application interface and Themis.
This module is only meant to be used by application interface
functions, which are themselves called by the ensemble manager while
it runs. Consult the documentation for a complete explanation.

The ``themis.laf`` module provides a Python and command-line interface to
launching allow_multiple batch scripts to a resource manager. Run that module directly
to produce the command-line interface.

Any other modules contained in this package are not intended for user use.

Running this package directly produces the command-line interface for
the ensemble manager.

Consult the documentation on each of those classes for information on usage.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import glob
import time
import json
import csv
import numbers
import shutil

from themis import database
from themis.database.utils import EnsembleDatabase
from themis import utils
from themis.utils import Run, CompositeRun, Step
from themis.resource import allocators
from themis import resource
from themis import versions
from themis import backend
from themis.backend.worker import prepper, finisher


__all__ = ["Themis", "user_utils", "runtime", "laf"]

__version__ = 1.2

_DEFAULT_MAX_CONCURRENCY = 100


def _get_app_spec_or_raise(setup_dir):
    try:
        return database.get_app_spec(setup_dir)
    except (IOError, OSError):
        raise ValueError("setup not found in {!r}".format(setup_dir))


def _validate_app_interface(app_interface):
    """Attempt to import the app interface"""
    try:
        utils.import_app_interface(app_interface)
    except ImportError as err:
        raise ImportError("Couldn't import the application interface: " + str(err))


class Themis(object):  # pylint: disable=too-many-public-methods
    """Represents and interfaces with an existing Themis ensemble.

    When creating a new ensemble, use one of the ``Themis.create*`` class methods,
    which will return a ``themis.Themis`` instance.
    The class constructor is used for getting a handle on an existing Themis
    ensemble---i.e. after ``create`` has been called.

    Use one of the ``Themis.execute_*`` methods to start an ensemble. The
    ``dryrun`` method is useful for checking that everything is in order.
    """

    #: This status matches runs which have not yet completed.
    RUN_QUEUED = EnsembleDatabase.RUN_QUEUED
    #: This status matches runs which have completed successfully.
    RUN_SUCCESS = EnsembleDatabase.RUN_SUCCESS
    #: This status matches runs which have failed.
    RUN_FAILURE = EnsembleDatabase.RUN_FAILURE
    #: This status matches runs which have been killed manually.
    RUN_KILLED = EnsembleDatabase.RUN_KILLED
    #: This status matches runs which returned one of the ``abort_on``.
    RUN_ABORTED = EnsembleDatabase.RUN_ABORTED

    STATUS_ENUMS = (RUN_QUEUED, RUN_SUCCESS, RUN_FAILURE, RUN_KILLED, RUN_ABORTED)

    # maps enums to human-readable strings, and vice versa
    _TRANSLATOR = {
        RUN_QUEUED: "queued",
        RUN_SUCCESS: "successful",
        RUN_FAILURE: "failed",
        RUN_KILLED: "killed",
        RUN_ABORTED: "aborted",
    }
    _TRANSLATOR.update({val: key for key, val in _TRANSLATOR.items()})

    @classmethod
    def translate_enum(cls, enum):
        """Convert a ``RUN_*`` enum to a human-readable string, or vice-versa."""
        return cls._TRANSLATOR[enum]

    @classmethod
    def create(  # pylint: disable=too-many-arguments,too-many-locals
        cls,
        application=None,
        runs=None,
        run_parse=None,
        run_copy=None,
        run_symlink=None,
        run_dir_names=None,
        app_interface=None,
        setup_dir=utils.DEFAULT_SETUP_DIR,
        max_restarts=0,
        max_failed_runs=None,
        app_is_batch_script=True,
        use_flux=False,
        abort_on=None,
    ):
        """Create a new Themis ensemble.

        Returns a ``themis.Themis`` instance representing the new ensemble.

        Only when one of the ``execute_*`` method is called does the ensemble actually
        begin.

        :param application: Path to the application to be executed. For instance,
            ``"/usr/bin/ls"``, ``"ls"``, or ``"../../bin/ls"`` might all be valid.
            This option is only relevant if ``themis.Run``
            (rather than ``CompositeRun``) objects are passed to this ensemble.
            If only ``CompositeRun`` objects are passed,
            the application is taken on a per-step basis from ``step.args[0]``.
        :type application: optional, str
        :param runs: An iterable of ``themis.CompositeRun`` objects,
            representing the runs of this ensemble.
            Runs within the ensemble will be executed concurrently.
            Each run is assigned a unique integer ID
            (the "run ID") corresponding to its index in the iterable. If no runs
            are passed, an empty ensemble is created.
        :type runs: optional
        :param run_parse: A file path or iterable of file paths. Unix-style path
            patterns are supported as well. The files specified should be text files;
            they will be hard-copied into the run directories and parsed.
            See :ref:`here <text_replacement>` for more. ``None``, the
            default, specifies that no files will be parsed.
        :type run_parse: str or iterable of str, optional
        :param run_copy: A file path or iterable of file paths. Unix-style path
            patterns are supported as well. The files/directories
            specified will be hard-copied into the run directories. ``None``, the
            default, specifies that no files will be copied.
        :type run_copy: str or iterable of str, optional
        :param run_symlink: A file path or iterable of file paths. Unix-style path
            patterns are supported as well. The files/directories
            specified will be symlinked into the run directories. ``None``, the
            default, specifies that no files will be symlinked.
        :type run_symlink: str or iterable of str, optional
        :param run_dir_names: The file system paths of the run directories.
            This argument should be a `python format string
            <https://docs.python.org/3/library/string.html#formatstrings>`_, where
            the field names correspond to the names of the variables in the samples
            argument. For instance, if the variables in the samples argument are
            "hydrostatics" and "viscocity", you might pass in the string
            ``"hydro={hydrostatics}/visc={viscocity}"``. This string will be
            formatted each run to yield the run directory; so one directory
            might be ``hydro=17.6/visc=35``. Note that
            the posix directory separator character "/"
            in the example string means that the resulting run directory
            will be in fact a sequence of two directories.
            The default value of ``None`` lets the naming
            scheme be determined internally.
        :type run_dir_names: str, optional
        :param app_interface: A file path to an
            :ref:`application interface <application_interface>` module.
        :type app_interface: str, optional
        :param setup_dir: A file system path indicating the directory in which to
            put the ensemble's setup files.
        :type setup_dir: str, optional
        :param max_restarts: the maximum number of times an individual run should be
            restarted if it fails---i.e. if the application exits with a
            nonzero return code.
            Setting this argument to ``None`` allows infinite restarts.
        :type max_restarts: int, optional
        :param max_failed_runs: the maximum total failures across the ensemble.
            If this number is exceeded, the ensemble will abort. The default
            of ``None`` allows infinite failures.
        :type max_failed_runs: int, optional
        :param app_is_batch_script: A boolean indicating whether the
            application given by the ``application`` parameter is
            a script that will launch parallel applications. For instance, if
            the application is a Slurm batch script that contains sruns,
            then this argument should be set to True. See the
            :ref:`batch script info page <batch_script_howto>` for more detail.
        :type app_is_batch_script: bool, optional
        :param use_flux: a boolean indicating whether to use Flux as the resource
            manager for the ensemble instead of the machine's
            native manager (e.g. Slurm).
            Can also be a string identifying the path to the Flux installation to use.
        :type use_flux: bool or str, optional
        :param abort_on: a sequence of positive integers. Each integer
            identifies a OS process return code. If ``application`` exits with
            one of those return codes, that run will be marked as ``RUN_ABORT``
            and will not be restarted.
        :returns: a ``Themis`` object representing a new ensemble
        :raises ValueError: if an ensemble exists in ``setup_dir``.
        """
        if cls.exists(setup_dir):
            raise Exception(
                "It looks like an ensemble was already "
                "created in the directory {0!r}.".format(setup_dir)
            )
        if not os.path.exists(setup_dir):
            os.makedirs(setup_dir)
        setup_dir = os.path.abspath(setup_dir)
        _validate_app_interface(app_interface)
        # initialize self
        if run_dir_names is None:
            run_dir_names = os.path.abspath(utils.DEFAULT_RUN_DIR_NAMES)
        else:
            run_dir_names = os.path.abspath(run_dir_names)
        if use_flux:
            resource_mgr = resource.identify_resource_manager(resource.Flux.identifier)
            flux_path = resource.validate_flux_path(use_flux)
        else:
            resource_mgr = resource.identify_resource_manager()
            flux_path = None
        run_parse, run_copy, run_symlink = utils.validate_run_files(
            run_parse, run_copy, run_symlink,
        )
        if app_is_batch_script and application:
            try:  # if the application is both parsed and symlinked, things get ugly
                run_symlink.remove(utils.validate_application(application))
            except ValueError:
                pass
        # many arguments are only needed on the compute nodes, and so are just stored
        # in the database; store them in a dict, to be later passed to the db
        spec = {"on_completion": None, "root_dir": os.getcwd(), "save_interval": 30}
        spec["application"] = (
            utils.validate_application(application) if application else None
        )
        spec["run_parse"] = run_parse
        spec["run_symlink"] = run_symlink
        spec["run_copy"] = run_copy
        spec["version_created"] = sys.version_info.major
        spec["app_interface"] = (
            None if app_interface is None else utils.existing_file(app_interface)
        )
        spec["resource_mgr"] = resource_mgr.identifier
        spec["flux_path"] = flux_path
        spec["setup_dir"] = setup_dir
        spec["run_dir_names"] = run_dir_names
        spec["max_restarts"] = utils.range_check(
            int(utils.convert_none(max_restarts, -1)), min_val=-1, name="max_restarts"
        )
        spec["max_failed_runs"] = utils.range_check(
            int(utils.convert_none(max_failed_runs, -1)),
            min_val=-1,
            name="max_failed_runs",
        )
        spec["app_is_batch_script"] = bool(app_is_batch_script)
        spec["database"] = {
            "type": "sqlite",
            "path": os.path.join(spec["setup_dir"], "ensemble_database.db"),
            "files": (os.path.join(spec["setup_dir"], "ensemble_database.db"),),
        }
        abort_on = () if abort_on is None else abort_on
        if not all((isinstance(code, int) and code > 0 for code in abort_on)):
            raise ValueError("`abort_on` must contain only positive integers")
        spec["abort_on"] = abort_on
        database.write_app_spec(spec, spec["setup_dir"])
        database.default_database(spec).create()
        database.get_status_store(spec["setup_dir"]).create()
        instance = cls(spec["setup_dir"])
        if runs:
            instance.add_runs(runs)
        return instance

    @classmethod
    def create_overwrite(cls, *args, **kwargs):
        """Create a new ensemble, removing an old one if it exists.

        Takes the same arguments as ``Themis.create()``.

        :returns: a ``Themis`` object representing a new ensemble
        """
        setup_dir = kwargs.get("setup_dir", utils.DEFAULT_SETUP_DIR)
        if cls.exists(setup_dir):
            cls.clear(setup_dir)
        return cls.create(*args, **kwargs)

    @classmethod
    def create_resume(cls, *args, **kwargs):
        """Create a new ensemble, or get a handle to an existing one.

        If an ensemble exists, return a ``Themis`` handle to it.
        Otherwise, create a new ensemble with the given arguments
        and return a ``Themis`` handle to it.

        Takes the same arguments as ``Themis.create()``.

        :returns: a ``Themis`` object representing either a new or an existing ensemble
        """
        setup_dir = kwargs.get("setup_dir", utils.DEFAULT_SETUP_DIR)
        if cls.exists(setup_dir):
            return cls(setup_dir)
        return cls.create(*args, **kwargs)

    @classmethod
    def clear(cls, setup_dir=utils.DEFAULT_SETUP_DIR):
        """Clear files generated by the ensemble manager for its own use.

        This function won't remove any user-provided files--for instance,
        input decks or required files. However, it will remove the
        ensemble manager's own log files, and any other files it uses
        to store its information. This includes stored samples and
        results; therefore, be careful that this is what you actually
        want to do.
        """
        if not cls.exists(setup_dir):
            return None
        app_spec = database.get_app_spec(setup_dir)
        expected_files = [
            os.path.join(setup_dir, database.APP_SPEC_NAME),
            os.path.join(setup_dir, database.STATUS_STORE_NAME),
        ]
        expected_files.extend(glob.glob(os.path.join(setup_dir, "themis_*.log")))
        expected_files.extend(
            glob.glob(os.path.join(setup_dir, "ensemble_launch_script.sh"))
        )
        expected_files.extend(app_spec["database"]["files"])
        for path in expected_files:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        return None

    @classmethod
    def exists(cls, setup_dir=utils.DEFAULT_SETUP_DIR):
        """Return ``True`` if an ensemble's setup exists in ``setup_dir``.

        :param setup_dir: has the same meaning as the ``setup_dir`` parameter
            to the ``themis.Themis.__init__`` method.
        """
        try:
            cls(setup_dir)
        except ValueError:
            return False
        else:
            return True

    def __init__(self, setup_dir=utils.DEFAULT_SETUP_DIR):
        """Constructor. Used to get a handle to an existing Themis ensemble."""
        self._setup_dir = os.path.abspath(setup_dir)
        self._spec = _get_app_spec_or_raise(self._setup_dir)
        self._status_store = database.get_status_store(self._setup_dir)
        self._db = database.default_database(self._spec)
        if self._spec["version_created"] > sys.version_info.major:
            raise RuntimeError(
                "Ensemble created with Python {} but running in Python {}".format(
                    self._spec["version_created"], sys.version_info.major
                )
            )

    def __repr__(self):
        """Return str representation of constructor call"""
        return "{}({!r})".format(type(self).__name__, self._spec["setup_dir"])

    @property
    def setup_dir(self):
        """Return the setup directory where Themis stores its data."""
        return self._spec["setup_dir"]

    def execute_local(  # pylint: disable=too-many-arguments
        self,
        blocking=False,
        timeout=None,
        parallelism=None,
        allow_multiple=False,
        max_concurrency=None,
    ):
        """Execute the ensemble locally. Return ``None``.

        :param blocking: if ``True``, block until the ensemble exits.
        :type blocking: bool
        :param timeout: length in minutes to run Themis for. Default is no time limit
        :type timeout: int, optional
        :param parallelism: if set to a positive integer N, reserve N cores for Themis
            itself, and run Themis in parallel on those cores. If you suspect Themis's
            performance is bottlenecking your ensemble, increase the parallelism.
        :type parallelism: int, optional
        :param allow_multiple: if ``True``, enable multiple concurrent executions
            (i.e. multiple simultaneous calls to ``execute_alloc`` or
            ``execute_local``). As this makes certain checks and behavior
            impossible, it is not the default.
        :type allow_multiple: bool, optional
        :param max_concurrency: the maximum number of concurrent runs Themis should
            allow at any given time. Too low a number may cause your ensemble to
            progress very slowly, much too large a number may eventually cause
            performance degradation on certain systems.
            If ``None``, pick a reasonable default value.
        :type max_concurrency: int, optional
        """
        self._execute(
            allocators.Allocation(timeout=timeout, repeats=0),
            allocator=allocators.InteractiveAllocator(),
            blocking=blocking,
            parallelism=parallelism,
            allow_multiple=allow_multiple,
            max_concurrency=max_concurrency,
        )

    def execute_alloc(  # pylint: disable=too-many-arguments
        self,
        nodes=1,
        partition=None,
        bank=None,
        name="themis",
        timeout=None,
        repeats=0,
        parallelism=None,
        early_stop=None,
        allow_multiple=False,
        max_concurrency=None,
    ):
        """Request an allocation and launch the ensemble within it.

        Requests an allocation from current machine's resource manager and returns
        the job ID.

        :param nodes: sets the allocation size in terms of total nodes.
        :type nodes: int
        :param partition: the compute partition to use for the ensemble.
        :type partition: str
        :param bank: the bank to use for the ensemble, e.g. "wbronze". The default value
            of None allows the resource manager to choose the bank.
        :type bank: str
        :param name: the name to give the allocation.
        :type name: str
        :param timeout: the time limit to request for the allocation, in minutes.
        :type timeout: int
        :param repeats: the number of times to replicate the allocation if time expires
            but the ensemble is not yet complete.
        :param parallelism: if set to a positive integer N, reserve N cores for Themis
            itself, and run Themis in parallel on those cores. If you suspect Themis's
            performance is bottlenecking your ensemble, increase the parallelism.
        :type parallelism: int, optional
        :param early_stop: a positive integer indicating the number of minutes "early"
            that Themis should stop launching new runs. Once Themis has run for
            ``(timeout - early_stop)`` minutes, it will go to sleep until the
            allocation expires. While Themis is sleeping, no new runs will be launched,
            and updates to existing runs will be ignored. This option is generally
            only useful if an application actively checks the remaining time
            on an allocation.
        :type early_stop: int, optional
        :param allow_multiple: if ``True``, permit multiple concurrent executions
            (i.e. multiple simultaneous calls to ``execute_alloc`` or
            ``execute_local``). As this makes certain checks and behavior
            impossible, it is not the default.
        :type allow_multiple: bool, optional
        :param max_concurrency: the maximum number of concurrent runs Themis should
            allow at any given time. Too low a number may cause your ensemble to
            progress very slowly, much too large a number may eventually cause
            performance degradation on certain systems.
            If ``None``, pick a reasonable default value.
        :type max_concurrency: int, optional
        :returns: the job ID of the allocation
        """
        alloc = allocators.Allocation(
            nodes=nodes,
            partition=partition,
            bank=bank,
            name=name,
            timeout=timeout,
            repeats=repeats,
        )
        return self._execute(
            alloc,
            parallelism=parallelism,
            early_stop=early_stop,
            allow_multiple=allow_multiple,
            max_concurrency=max_concurrency,
        ).job_id

    def _execute(  # pylint: disable=too-many-arguments
        self,
        allocation,
        allocator=None,
        blocking=False,
        parallelism=None,
        early_stop=None,
        allow_multiple=False,
        max_concurrency=None,
        debug=False,
    ):
        if self.count_by_status(self.RUN_QUEUED) <= 0:
            raise RuntimeError("There are no runs to execute")
        resource_mgr = resource.identify_resource_manager(
            self._spec["resource_mgr"], path=self._spec["flux_path"]
        )
        if max_concurrency is None or max_concurrency < 0:
            max_concurrency = _DEFAULT_MAX_CONCURRENCY
        if allocator is None:
            allocator = resource_mgr.allocator()
        alloc_id = self._status_store.add_allocation(allocation)
        parallelism = parallelism if parallelism is not None else 0
        timeout = allocation.timeout if allocation.timeout is not None else 0
        early_stop = int(early_stop) if early_stop is not None else 0
        backend_commands = [
            resource_mgr.commands_to_launch_backend(
                timeout,
                parallelism,
                alloc_id,
                early_stop,
                allow_multiple,
                self._spec["setup_dir"],
                max_concurrency,
            )
        ]
        alloc_dir = resource.backend_dir(self._spec["setup_dir"], alloc_id)
        if not os.path.exists(alloc_dir):
            os.makedirs(alloc_dir)
        allocator.start(
            allocation, backend_commands, alloc_dir, alloc_dir,
        )
        if blocking and debug:
            allocator.wait(debug=debug)
        elif blocking:
            allocator.wait()
        return allocator

    def dry_run(self, *run_ids, **kwargs):
        """Perform one or more dry runs. If `run_ids` is empty, dry-run every run.

        Populate each run directory with
        the "run_symlink" and the "input deck" files, and call the
        application interface's ``prep_run`` function, if it exists.

        This is done in serial, so is safe to do on login nodes of HPC clusters.

        :param run_ids: the run IDs of the runs to dry-run.
        :param verbosity: If > 0, print messages about the progress of the dry runs.
        """
        verbosity = kwargs.get("verbosity", 0)
        if not run_ids:
            run_ids = self.filter_by_status(self.RUN_QUEUED)
        run_mapping = self._db.user_facing_run_info(run_ids)
        for run_id in run_ids:
            self._call_worker(run_id, run_mapping[run_id], "prep_run", verbosity)

    def _call_worker(self, run_id, run, app_interface_attr, verbosity):
        """Call a worker function directly, and return its return value."""
        backend.export_to_user_utils(self._spec, self._status_store, run_id, run)
        run_dir = utils.get_run_dir(run_id, self._spec["run_dir_names"], run.sample)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        app_interface_func = getattr(
            utils.import_app_interface(self._spec["app_interface"]),
            app_interface_attr,
            None,
        )
        if verbosity:
            message = "Executing {action} #{id} in {cwd!r}..."
            print(
                message.format(
                    action="dry run"
                    if app_interface_attr == "prep_run"
                    else app_interface_attr,
                    id=run_id,
                    cwd=os.path.relpath(run_dir),
                )
            )
        if app_interface_attr == "prep_run":
            return prepper.preparation(
                app_interface_func, run_dir, self._spec, run.sample, run.steps
            )
        if app_interface_attr == "post_run":
            return finisher.user_post_run(app_interface_func, run_dir)
        raise ValueError("Unrecognized interface attribute")

    def call_post_run(self, run_id):
        """Call the application interface's ``post_run`` function for a single run.

        :param run_id: the run ID of the run to execute ``post_run`` for.
        :type run_id: int
        :return: the return value of the ``post_run`` function. ``None`` if the
            function does not exist.
        """
        run_info = database.default_database(self._spec).get_run_info(run_id)
        if run_info is None:
            raise ValueError("No run has the id " + str(run_id))
        return self._call_worker(run_id, run_info.run, "post_run", 0,)

    def call_prep_ensemble(self):
        """Call the application interface's ``prep_ensemble`` function, if it exists."""
        self._call_ensemble_func("prep_ensemble")

    def call_post_ensemble(self):
        """Call the application interface's ``post_ensemble`` function, if it exists."""
        self._call_ensemble_func("post_ensemble")

    def _call_ensemble_func(self, user_func_name):
        user_func = getattr(
            utils.import_app_interface(self._spec["app_interface"]),
            user_func_name,
            None,
        )
        if not callable(user_func):
            return
        backend.export_to_user_utils(self._spec, self._status_store)
        if user_func_name == "prep_ensemble":
            backend.user_prep_ensemble(user_func, os.getcwd())
        elif user_func_name == "post_ensemble":
            backend.user_post_ensemble(user_func, os.getcwd())
        else:
            raise ValueError("Unrecognized interface attribute")

    def filter_by_status(self, *statuses):
        """Return an iterable of run IDs representing runs with a given status.

        The return value is not guaranteed to be perfectly accurate--there
        may be considerable delay in propagating a run's status change to
        the caller of this method.

        :param statuses: zero or more ``themis.Themis.RUN_*`` objects.
        :return: an iterable of integer run IDs, representing runs with
            one of the given statuses. If statuses is empty, return all run IDs.
        """
        return self._db.runs_with_completion_status(*statuses)

    def count_by_status(self, *statuses):
        """Return a count of runs with a given status.

        This method is equivalent to calling ``len()`` on the results
        of ``filter_by_status``; however, this method may be considerably
        faster for large ensembles.

        :param statuses: same meaning as in ``filter_by_status``.
        :return: the number of runs with one of the given statuses.
            If statuses is empty, return the total number of runs.
        """
        return self._db.count_runs_by_completion(*statuses)

    def progress(self):
        """Return a 2-tuple of ints representing the number of (completed, total) runs.

        Convenience wrapper around the ``count_by_status`` method.

        A completed run is one that has either succeeded, failed, or been killed.
        """
        completed = self.count_by_status(
            self.RUN_FAILURE, self.RUN_SUCCESS, self.RUN_KILLED
        )
        return (completed, self.count_by_status())

    def requeue_runs(self, run_ids, hard=False):
        """Requeue one or more completed runs.

        Restarted runs have their status reset to ``RUN_QUEUED``
        and will be executed at the earliest opportunity.
        If the run has not completed by the time this method is called,
        no action is taken.

        :param run_ids: an iterable of integer run IDs.
        :param hard: if ``True``, reset the given runs' progress back to step 0.
            If ``False``, maintain the runs' progress, but re-attempt execution.
        """
        self._db.mark_runs_to_restart(run_ids, hard)

    def restart_runs(self, *args, **kwargs):
        """Alias for ``requeue_runs``."""
        return self.requeue_runs(*args, **kwargs)

    def dequeue_runs(self, run_ids):
        """Dequeue one or more incomplete (i.e. queued) runs.

        Dequeued runs have their status set to ``RUN_KILLED`` after a delay.
        If the run has already finished when this method is called, no action is taken.
        If the run has not yet started, that run will never be started.

        :param run_ids: an iterable of integer run IDs.
        """
        self._db.mark_runs_to_kill(run_ids)

    def kill_runs(self, *args, **kwargs):
        """Alias for ``dequeue_runs``."""
        return self.dequeue_runs(*args, **kwargs)

    def add_runs(self, runs):
        """Add runs to the ensemble.

        The new runs will be executed at the earliest opportunity---either
        immediately or at the time of the next restart.

        :param runs: an iterable of ``Run`` objects.
        """
        runs = utils.validate_runs(
            runs,
            self._spec["run_dir_names"],
            application=self._spec["application"],
            batch_script=self._spec["app_is_batch_script"],
        )
        self._db.add_runs(runs)

    def runs(self, run_ids):
        """Get information about a set of runs.

        :param run_ids: an iterable of integer run IDs.
        :returns: a mapping from ``run_ids`` to ``Run`` objects. The objects will
            have additional ``result`` and ``status`` attributes.
        """
        return self._db.user_facing_run_info(run_ids)

    def as_dataframe(self, include_none=True):
        """Return a pandas DataFrame containing information about each run.

        The DataFrame will be indexed on run IDs, and have columns for
        each sample label.

        :param include_none: whether to include rows with no result
        :type include_none: bool, optional

        :raises ImportError: if ``pandas`` cannot be imported
        """
        from pandas import DataFrame

        mappings = self._db.as_mappings(include_none)
        for mapping in mappings:
            mapping.update(mapping["sample"])
            del mapping["sample"]
        dframe = DataFrame(mappings)
        dframe.set_index("run_id", inplace=True)
        dframe.drop(columns=["status"], inplace=True)
        return dframe

    def _info_to_write(self):
        mappings = self._db.as_mappings(True)
        for mapping in mappings:
            mapping["status"] = self.translate_enum(mapping["status"])
        return mappings

    def write_json(self, stream):
        """Write a JSON document containing a full set of data about the ensemble.

        :param stream: a file-like object that supports writing ``str``
        """
        runs = self._info_to_write()
        json.dump(runs, stream, indent=2, default=utils.Step.encode)

    def write_yaml(self, stream):
        """Write a YAML document containing a full set of data about the ensemble.

        :param stream: a file-like object that supports writing ``str``

        :raises ImportError: if ``yaml`` cannot be imported
        """
        import yaml

        runs = self._info_to_write()
        return yaml.dump(runs, stream)

    def write_csv(self, stream):
        """Write a CSV containing a full set of data about the ensemble.

        :param stream: a file-like object that supports writing ``str``

        :raises ValueError: if the results cannot be formatted into a csv
        """
        run_dicts = self._info_to_write()
        all_sample_labels = set()
        all_result_labels = set()
        for run_dict in run_dicts:
            all_sample_labels |= set(run_dict["sample"].keys())
            run_dict.update(run_dict["sample"])
            del run_dict["sample"]
            if isinstance(run_dict["result"], (numbers.Number, str)):
                run_dict["result"] = (run_dict["result"],)
            if isinstance(run_dict["result"], versions.Mapping):
                all_result_labels |= run_dict["result"].keys()
                run_dict.update(run_dict["result"])
                del run_dict["result"]
            elif isinstance(run_dict["result"], versions.Sequence):
                result_labels = [
                    "output_{}".format(i) for i in range(len(run_dict["result"]))
                ]
                for key, val in zip(result_labels, run_dict["result"]):
                    run_dict[key] = val
                del run_dict["result"]
                all_result_labels |= set(result_labels)
        fieldnames = ["run_id", "status", "tasks", "cores_per_task", "gpus_per_task"]
        fieldnames.extend(all_sample_labels)
        fieldnames.extend(sorted(all_result_labels))
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for run_dict in run_dicts:
            writer.writerow(run_dict)

    def set_result(self, run_id, result):
        """Set the result for a run of the ensemble.

        :param run_id: the ID of the run
        :type run_id: int
        :param result: the result to add
        """
        self._db.add_result((run_id,), (result,))

    def run_dirs(self, nonexistent=False):
        """Return an iterable of (run_id, working_directory) pairs in arbitrary order.

        :param nonexistent: if ``True``, include directories that haven't yet been
            created.
        :type nonexistent: bool, optional
        """
        for mapping in self._db.as_mappings(True):
            run_dir = utils.get_run_dir(
                mapping["run_id"], self._spec["run_dir_names"], mapping["sample"]
            )
            if nonexistent or os.path.exists(run_dir):
                yield mapping["run_id"], run_dir

    def on_completion(self, args, cwd=os.curdir, stdout=None, stderr=None):
        """Provide an executable for Themis to launch once the ensemble has completed.

        The executable will be invoked during a call to ``Themis.execute_*`` once
        every run has finished.

        Only the most recent call to this function is remembered.

        :param args: an iterable of str defining the executable and its arguments,
            e.g. ``["/usr/bin/python", "-vvv", "myscript.py" "--foobar"]``.
        :param cwd: the working directory for the executable when it is launched
        :type cwd: str
        :param stdout: file to redirect stdout to;
            default redirects to a Themis log file
        :type stderr: str
        :param stderr: file to redirect stderr to;
            default redirects to a Themis log file
        :type stderr: str
        """
        if not all((isinstance(arg, str) for arg in args)):
            raise ValueError("arguments must be strings")
        if not os.path.isdir(cwd):
            raise ValueError("Not a directory: {!r}".format(cwd))
        executable = utils.which(args[0])
        if executable is None:
            raise ValueError("Cannot find executable {!r}".format(args[0]))
        args = list(args)  # make a shallow copy for modification
        args[0] = executable
        self._spec["on_completion"] = {
            "args": args,
            "cwd": os.path.abspath(cwd),
            "stdout": os.path.abspath(stdout) if stdout is not None else None,
            "stderr": os.path.abspath(stderr) if stderr is not None else None,
        }
        database.write_app_spec(self._spec, self._spec["setup_dir"])
