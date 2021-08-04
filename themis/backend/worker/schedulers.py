"""
This module defines an interface for submitting runs to be executed.

The interface consists of the function setup_scheduler, which returns a
Scheduler object. Scheduler objects have in turn their own interface consisting of
the methods submit, shutdown, free_runs, kill_runs, active_runs, done, and
save_progress. See the ThreadedAppRunBatcher class for detail.

Some terminology that should be used consistently throughout this module:

    A step is the execution of an application. This involves not just executing
        the executable, but also setting the working directory, arguments, and
        so on. bundle of information about the creation of an application. A step
        corresponds to a single `srun` or `flux mini run` etc.
    A run is a series of steps executed in order: step 1 followed by step 2 followed
        by step 3 ... and so on. Each run corresponds to a sample, and is uniquely
        identified by a run ID.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import logging
import time
import threading
import functools
from collections import deque

import themis.utils
from themis import resource
from themis import backend
from themis.database.utils import EnsembleDatabase
from themis.backend.worker import prepper


LOGGER = logging.getLogger(__name__)


def setup_scheduler(app_spec, server_args, max_active_runs, do_prep, do_post):
    """Create and return a Scheduler object.

    For information on Scheduler objects and their interface, consult the
    ThreadedAppRunBatcher class.
    """
    resource_mgr = resource.identify_resource_manager(app_spec["resource_mgr"])
    LOGGER.debug("Using resource manager %s", resource_mgr)
    step_creator = _StepCreator(app_spec, server_args, do_prep, do_post)
    scheduler = JobSubmissionThread(
        max_active_runs,
        functools.partial(
            _create_run, step_creator, app_spec["max_restarts"], app_spec["abort_on"]
        ),
        resource_mgr.executor(app_spec["setup_dir"], server_args),
        app_spec["max_restarts"],
    )
    scheduler.start()
    return scheduler


def _create_run(step_creator, max_restarts, abort_on, run_info, executor):
    """Create and return a new _AppRun object that will execute a run.

    :param step_creator: the StepCreator instance to be used for creating the steps
        for each new run.
    :param max_restarts: the number of times a step of each run may be
        restarted
    """
    return _AppRun(
        run_info.run_id,
        run_info.status,
        max_restarts - run_info.restarts,
        step_creator(run_info),
        abort_on,
        executor,
    )


# pylint: disable=too-many-instance-attributes
class JobSubmissionThread(threading.Thread):
    """Thread that creates and manages runs.

    :param waited_queue: a collections.Deque holding runs that have completed
    :param run_creator: callable for creating runs
    :param max_active_runs: the maximum number of concurrent runs to create
    """

    def __init__(self, max_active_runs, run_creator, executor, max_restarts):
        threading.Thread.__init__(self)
        self.daemon = True
        self.name = type(self).__name__
        self.profiler = None
        self.__exit_event = threading.Event()
        self.__executor = executor  # constant
        self.__run_creator = run_creator  # constant
        self.__max_active_runs = max_active_runs  # constant
        self.__max_restarts = max_restarts  # constant
        self.__runs = {}  # thread-private
        self.__done_lock = threading.Lock()  # public and thread-safe
        self.__new_run_queue = deque()  # public and thread-safe
        self.__statuses_to_write = deque()  # public and thread-safe
        self.__runs_to_kill = deque()  # public and thread-safe
        self.__done = False  # writes should be protected by __done_lock
        self.__jobids_to_runs = {}  # thread-private

    def run(self):
        """Profile the _run method if an environment variable is set."""
        self.profiler = backend.get_profiler()
        backend.profile(self.profiler, self._run)

    def submit(self, new_run_infos):
        """Submit runs; return when all have been submitted."""
        for run_info in new_run_infos:
            self.__new_run_queue.append(run_info)
        if new_run_infos:
            with self.__done_lock:
                self.__done = False

    def kill_runs(self, run_ids):
        """Kill the runs given by run_ids, provided they are currently active.

        Adds the run IDs to a data structure the thread will read from.
        The actual killing is left to that thread.
        """
        for run_id in run_ids:
            self.__runs_to_kill.append(run_id)

    def count_pending_runs(self):
        """Return the number of runs which have not yet been started."""
        return len(self.__new_run_queue)

    def complete_existing_runs(self):
        """Block until the scheduler has completed all active runs.

        Submitted but not active runs will not be scheduled."""
        while self.__new_run_queue:
            try:
                self.__new_run_queue.popleft().run_id
            except IndexError:  # queue may be emptied by another thread
                pass
        LOGGER.debug(
            "Draining runs from %s", type(self).__name__,
        )
        while not self.done():
            time.sleep(0.5)
        LOGGER.debug("%s drained", type(self).__name__)

    def save_progress(self, ensemble_db):
        """Save progress to the database."""
        run_ids_to_save = []
        statuses_to_save = []
        while self.__statuses_to_write:
            app_run = self.__statuses_to_write.popleft()
            run_ids_to_save.append(app_run.run_id)
            statuses_to_save.append(
                (
                    app_run.state,
                    app_run.completion_state,
                    self.__max_restarts - app_run.remaining_restarts,
                )
            )
        completed_ids_and_status = [
            (run_id, status[1])
            for run_id, status in zip(run_ids_to_save, statuses_to_save)
        ]
        ensemble_db.set_run_status(run_ids_to_save, statuses_to_save)
        return completed_ids_and_status

    def done(self):
        """Return True if there is no more work to be done."""
        return self.__done

    def shutdown(self, ensemble_db):
        """Shut down the instance."""
        self.__exit_event.set()
        self.join()
        for run in self.__runs.values():
            self.__statuses_to_write.append(run)
        self.save_progress(ensemble_db)
        return self.profiler

    def _run(self):
        """Loop through runs, attempting to advance them to the next state.

        The loop will not end until self._shutdown is set to True,
        Which must be done externally--there is no internal trigger to stop.
        """
        while not self.__exit_event.is_set():
            while len(self.__runs) < self.__max_active_runs and self.__new_run_queue:
                run_info = self.__new_run_queue.popleft()
                new_run = self.__run_creator(run_info, self.__executor)
                self.__runs[run_info.run_id] = new_run
                self.__jobids_to_runs[new_run.job_id] = new_run
            run_update = self.__executor.wait(timeout=0.1)
            if run_update is not None:
                self.__send_update(*run_update)
            if not self.__runs:
                with self.__done_lock:
                    if not self.__new_run_queue:
                        self.__done = True
            # if there are runs to kill, kill them
            while self.__runs_to_kill:
                run_id_to_kill = self.__runs_to_kill.popleft()
                if run_id_to_kill in self.__runs:
                    self.__runs[run_id_to_kill].kill_async()

    def __send_update(self, job_id, exit_status):
        """Update a run with the completion of one of its jobs."""
        run_to_update = self.__jobids_to_runs.pop(job_id, None)
        if run_to_update is None:
            raise ValueError(
                "Job ID {} (exitcode {}) doesn't match anything".format(
                    job_id, exit_status
                )
            )
        if run_to_update.run_complete(exit_status):
            self.__statuses_to_write.append(run_to_update)
            del self.__runs[run_to_update.run_id]
        else:
            self.__jobids_to_runs[run_to_update.job_id] = run_to_update


# pylint: disable=too-few-public-methods
class _StepCreator(object):
    """Instances of this class create Step objects.

    Instances store information which is constant across runs;
    Then, to create steps for a run, only the run RunInfo object
    for that run needs to be passed in.

    :param app_spec: the app_spec dictionary for the ensemble.
    :param server_info: the sequence of arguments to be used for
        connecting to the server
    """

    _THIS_FILE_LOCATION = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, app_spec, server_info, do_prep, do_post):
        self._cwd = os.getcwd()
        self._app_spec = app_spec
        self._server_info = server_info
        # optimizations: no point launching a worker if it has nothing to do
        self._do_post = do_post
        self._do_prep = do_prep

    def __call__(self, run_info):
        """Return an interable of steps for the run given by run_info."""
        steps = []
        run_dir = themis.utils.get_run_dir(
            run_info.run_id, self._app_spec["run_dir_names"], run_info.sample
        )
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir)
        if self._do_prep:
            steps.append(self._worker_step(run_info.run_id, run_dir, "prepper.py"))
        else:
            prepper.preparation(
                None, run_dir, self._app_spec, run_info.sample, run_info.steps
            )
        for step in run_info.steps:
            steps.append(self._user_step(step, run_dir))
        if self._do_post:
            steps.append(self._worker_step(run_info.run_id, run_dir, "finisher.py"))
        return steps

    @staticmethod
    def _user_step(step, run_dir):
        """Create steps for users' applications"""
        # if step.cwd is relative, make it relative to run dir
        step.cwd = os.path.join(run_dir, step.cwd)
        if not os.path.isdir(step.cwd):
            os.makedirs(step.cwd)
        step.args[0] = themis.utils.get_application(
            step.args[0], run_dir, step.batch_script,
        )
        return (step, None)

    def _worker_step(self, run_id, run_dir, worker_name):
        """Create a worker task step. The step is assumed to require a single core."""
        args = [
            sys.executable,
            os.path.join(self._THIS_FILE_LOCATION, worker_name),
            str(run_id),
            self._cwd,
            "--server-args",
        ]
        args.extend(self._server_info)
        if worker_name == "prepper.py":
            message = "Prepping run {}".format(run_id)
        else:
            message = "Finishing run {}".format(run_id)
        return (themis.utils.Step(args, batch_script=False, cwd=run_dir), message)


# pylint: disable=too-many-instance-attributes
class _AppRun(object):
    """Instances of this class represent and execute a single run of an ensemble.

    A run consists of one or more steps. Steps are represented by Step objects
    (naturally) and are expected to have a certain set of attributes.

    Upon construction, instances of this class execute their steps in order.
    A step is restarted upon failure (up to max_restarts times), and a step is only
    launched if there are resources available.

    Instances will not progress their steps in the background; the advance method
    must be repeatedly called in order for runs to make any progress.

    :param run_id: the run ID for this run
    :param state: the state of the current run. Usually 0, but may be different if
        the current run started previously, failed, and is now being restarted.
    :param steps: a sequence of Step objects representing the steps making up this run
    :param launcher: the Launcher class to be used for launching steps in this run
    :param max_restarts: the number of times a step may be restarted before the run
        is terminated.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, run_id, state, remaining_restarts, steps, abort_on, executor):
        self.run_id = run_id
        self.state = state
        self.completion_state = EnsembleDatabase.RUN_QUEUED
        self._steps = steps
        self.remaining_restarts = remaining_restarts
        self._abort_on = abort_on
        self._executor = executor
        self.job_id = None
        self._done = False
        self._killed = False
        self._attempt_execution()

    def __repr__(self):
        """Return string representation of constructor call"""
        return "{}({}, {})".format(type(self).__name__, self.run_id, self.state)

    def _attempt_execution(self):
        """Attempt to execute the next step of this run"""
        curr_step, message = self._steps[self.state]
        if message is None:
            LOGGER.info("Launching run %i", self.run_id)
        else:
            LOGGER.info(message)
        self.job_id = self._executor.submit(self.run_id, curr_step)

    def run_complete(self, returncode):
        """Take action given that the most recent job has completed."""
        if returncode == 0:
            self.state += 1
            # check if the job has finished the final stage
            if self.state == len(self._steps):
                self.completion_state = EnsembleDatabase.RUN_SUCCESS
                LOGGER.info("Run %i completed successfully", self.run_id)
                self._done = True
                return True
            # if it hasn't, advance it to the next stage
            self._attempt_execution()
            return False
        # step hit fatal error, abort
        if returncode in self._abort_on:
            LOGGER.info("Run %i aborted with return code %s", self.run_id, returncode)
            self._done = True
            self.completion_state = EnsembleDatabase.RUN_ABORTED
            return True
        # The step exited with errors. check whether to restart
        if self._killed:
            LOGGER.info("Run %i was killed", self.run_id)
            self._done = True
            self.completion_state = EnsembleDatabase.RUN_KILLED
            return True
        LOGGER.info(
            "Failure in run %i on step %i: returncode %s",
            self.run_id,
            self.state,
            returncode,
        )
        if self.remaining_restarts != 0:
            # if remaining_restarts was -1, restart forever
            self.remaining_restarts -= 1
            self._attempt_execution()
            return False
        LOGGER.info("Run %i has hit the restart limit", self.run_id)
        self.completion_state = EnsembleDatabase.RUN_FAILURE
        self._done = True
        return True

    def kill_async(self):
        """Kill the run."""
        if not self._done:
            self._done = True
            self._killed = True
            self._executor.kill(self.job_id)
