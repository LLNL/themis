"""
This script prepares an ensemble for execution, and launches the
individual applications making up the ensemble.

Many variable names are used repeatedly throughout this module. They are
intended to have a consistent meaning. Some of them are as follows:
    :param ensemble_db: an EnsembleDatabase object
    :param app_spec: a constant mapping containing information about the ensemble.
        Should have a fixed set of keys; see the ensemble.database module for more.
    :param app_interface: a user's application interface module, or None.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import time
import logging
import socket
import argparse
import contextlib

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.insert(0, os.path.abspath(__file__).rsplit(os.sep, 4)[0])

# pylint: disable=wrong-import-position
import themis.utils
from themis import database
from themis.versions import reprlib
from themis import backend
from themis.backend.network import BackendClient, CERT_NAME
from themis.backend.worker import schedulers
from themis.backend.worker import network


LOGGER = logging.getLogger(__name__)


class ProgressSaver(object):
    """Instances of this class are used to save progress in the submission loop.

    The interface consists only of the __call__ method, which should be
    called regularly. It only saves progress if progress hasn't been in the past
    ``frequency`` seconds; otherwise, does nothing.

    :param max_failed_runs: The maximum number of tolerated failures
    :param frequency: the frequency (in seconds) of saves
    """

    def __init__(self, server, scheduler, max_failed_runs, frequency):
        """Initialize a new instance."""
        self._server = server
        self._scheduler = scheduler
        self._max_failed_runs = max_failed_runs
        self._save_interval = frequency
        self._time_of_last_save = time.time()

    def __repr__(self):
        """Return str representation of constructor call"""
        return "{}({}, {}, {}, {})".format(
            type(self).__name__,
            self._server,
            self._scheduler,
            self._max_failed_runs,
            self._save_interval,
        )

    def __call__(self, ensemble_db, status_store):
        """Save server and scheduler progress; return True if an exit condition
        was reached.

        Only saves progress if progress hasn't been in the past ``frequency``
        seconds; otherwise, does nothing.

        The three exit conditions recognized:
        1)  The server or scheduler thread dies
        2)  A STOP_ENSEMBLE message is received
        3)  Too many runs fail

        :param curr_time: the current Unix standard time.
        """
        curr_time = time.time()
        if curr_time - self._time_of_last_save > self._save_interval:
            self._time_of_last_save = curr_time
            return self.save(ensemble_db, status_store)
        return False

    def save(self, ensemble_db, status_store):
        """Save progress."""
        if self._check_alive() or self._check_messages(status_store, ensemble_db):
            return True
        # get newly failed runs from the scheduler. The set enforces uniqueness
        completed_ids_and_status = self._scheduler.save_progress(ensemble_db)
        num_new_failed_runs = sum(
            [
                status == ensemble_db.RUN_FAILURE
                for _, status in completed_ids_and_status
            ]
        )
        self._server.remove_completed_runs(
            [run_id for run_id, _ in completed_ids_and_status]
        )
        self._server.save_progress(ensemble_db)
        return self._check_failed_runs(status_store, num_new_failed_runs)

    def _check_alive(self):
        """Return True if the server or scheduler has died."""
        if not (self._server.is_alive() and self._scheduler.is_alive()):
            LOGGER.error("Threading exception, exiting...")
            return True
        return False

    def _check_messages(self, status_store, ensemble_db):
        """Check user messages for killing runs or stopping the ensemble.

        Kill any runs required by the messages.

        If a STOP_ENSEMBLE message is received, return True once current runs
        have cleared (i.e. the ensemble has stopped).
        """
        runs_to_kill = ensemble_db.runs_to_kill()
        LOGGER.debug("Killing runs %s", reprlib.repr(runs_to_kill))
        self._scheduler.kill_runs(runs_to_kill)
        if status_store.fetch()["stop"]:
            # if a 'complete_existing_runs' message is received, shut down
            LOGGER.info("Stop ensemble message received")
            self._scheduler.complete_existing_runs()
            return True
        return False

    def _check_failed_runs(self, status_store, num_new_failed_runs):
        """Return True if the number of failed runs exceeds the maximum allowed."""
        if self._max_failed_runs > -1:
            total_failed_runs = status_store.increment_failed_runs(num_new_failed_runs)
            if total_failed_runs > self._max_failed_runs:
                # pass None for app interface so post_ensemble isn't called
                LOGGER.error(
                    "Total failed runs (%s) exceeds maximum of %s. Exiting...",
                    total_failed_runs,
                    self._max_failed_runs,
                )
                # don't call post_ensemble--just let existing runs complete
                self._scheduler.complete_existing_runs()
                return True
        return False


# pylint: disable=too-many-arguments
def enter_submission_loop(
    ensemble_db, status_store, app_spec, queue_size, do_prep, do_post
):
    """Enter a loop of submitting runs and checking for new runs.

    Aside from errors, there are a handful of exit conditions:
    1)  There are no new runs, all existing runs have completed,
            AND the user's post_ensemble has been called and did
            not return any new runs.
    2)  A user called for a STOP_ENSEMBLE; in this case, the loop exits
            once all existing runs have completed.
    3)  The maximum number of failed runs has been exceeded.

    :return: an iterable, with entries being either
        Profiler objects from the cProfile module, or None.
    """
    # create a Scheduler object which takes in run IDs and runs them
    with backend.managed_shutdown(
        network.WorkerServer(CERT_NAME, app_spec), ensemble_db
    ) as server, backend.managed_shutdown(
        schedulers.setup_scheduler(
            app_spec, server.server_address, queue_size, do_prep, do_post
        ),
        ensemble_db,
    ) as scheduler, backend.RunFeeder(
        # Pick a reasonable size for the cache of new runs
        ensemble_db,
        int(2.5 * queue_size),
    ) as run_feeder:
        LOGGER.info("Beginning to submit runs.")
        LOGGER.info("Server listening on %s", server.server_address)
        progress_saver = ProgressSaver(
            server, scheduler, app_spec["max_failed_runs"], app_spec["save_interval"]
        )
        # fetch new runs from the RunFeeder whenever the scheduler has less than
        # min_pending_runs (some reasonable factor of max_concurrency) to execute
        min_pending_runs = max(2, queue_size)
        while True:
            pending_runs = scheduler.count_pending_runs()
            if pending_runs < min_pending_runs:
                # get a sequence of new RunInfo objects to submit, and their statuses
                new_runs = run_feeder.new_runs(min_pending_runs - pending_runs)
                # submit the runs (whether new or restarts) to the scheduler and server
                scheduler.submit(new_runs)
                server.submit(new_runs)
                run_feeder.poll_cache()
                if run_feeder.is_empty() and scheduler.done():
                    # there are no new runs to submit
                    if not run_feeder.fill_cache():
                        # all existing runs have completed, and no new runs yet
                        # Now check if the user's post_ensemble adds new runs
                        LOGGER.info("Ensemble complete. Shutting down and exiting...")
                        break
            # check for messages and save progress
            if progress_saver(ensemble_db, status_store):
                # the call only returns true if exit condition #2 or #3 occurs
                LOGGER.info("Stop condition triggered, exiting submission loop")
                break
            time.sleep(0.02)
        return (scheduler.profiler, server.profiler)


def main(app_spec, ensemble_db, queue_size):
    """Perform all setup actions before entering the main event loop.

    :param max_runtime: the max max_runtime (in minutes) for this submitter
    """
    status_store = database.get_status_store(app_spec["setup_dir"])
    LOGGER.info("Using database: %s", ensemble_db)
    # import the application interface
    app_interface = themis.utils.import_app_interface(app_spec["app_interface"])
    do_post = hasattr(app_interface, "post_run")
    do_prep = hasattr(app_interface, "prep_run") or len(app_spec["run_parse"]) > 2
    # enter a loop of submitting runs and then checking for new runs
    return enter_submission_loop(
        ensemble_db, status_store, app_spec, queue_size, do_prep, do_post
    )


def setup_parser():
    """Set up the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Execute runs. Requires a database "
            "holding information about the ensemble."
        )
    )
    parser.add_argument(
        "--id",
        metavar="N",
        type=str,
        help="the (unique) id of this scheduler run",
        default="0",
    )
    parser.add_argument(
        "-c",
        "--max-concurrency",
        metavar="N",
        type=int,
        help="the maximum number of runs to execute concurrently",
        required=True,
    )
    parser.add_argument(
        "--server-args", nargs="*", help="server connection information", type=str
    )
    parser.add_argument(
        "--setup-dir", help="Path to the Themis setup directory", default=os.curdir,
    )
    return parser


def get_identifier(args):
    """Return the ID of this instance and the number of cores and GPUs it controls."""
    if os.getenv("FLUX_TREE_ID") is not None:
        # if running inside of Flux tree, there will be no resource args,
        # only env vars exported by Flux
        identifier = os.getenv("FLUX_TREE_ID")
        sys.stderr = open("themis_worker_{}.log".format(identifier), "a")
        sys.stdout = sys.stderr
    else:
        identifier = str(args.id)
    return identifier


@contextlib.contextmanager
def exit_on_signal():
    """On SignalCaught exception, exit Python."""
    try:
        yield
    except backend.SignalCaught as sig:
        LOGGER.info("Signal %i received, exiting...", sig.signum)
        raise


def setup():
    """Collect this submitter's setup and pass it to main().

    Profile main if the profiling env var is set.
    """
    args = setup_parser().parse_args()
    identifier = get_identifier(args)
    LOGGER.info("Worker starting...")
    LOGGER.info("Received args %s", args)
    if "FLUX_URI" in os.environ:
        LOGGER.info(
            "Flux remote URI is %s, local URI is %s",
            "ssh://"
            + socket.gethostname()
            + os.environ["FLUX_URI"].replace("local://", ""),
            os.environ["FLUX_URI"],
        )
    app_spec = database.get_app_spec(args.setup_dir)
    backend.handle_signals()
    with exit_on_signal():
        LOGGER.info("Maximum concurrent runs: %i", args.max_concurrency)
        if args.server_args:
            ensemble_db = BackendClient(CERT_NAME, *args.server_args)
        else:
            ensemble_db = database.default_database(app_spec)
        profiler, additional_profilers = backend.profile_from_new(
            main, app_spec, ensemble_db, args.max_concurrency
        )
        backend.dump_profile_data(
            "submitter_profile_stats_{}.pstat".format(identifier),
            profiler,
            *additional_profilers
        )
    LOGGER.info("Exited cleanly.")


if __name__ == "__main__":
    with backend.managed_logging(sys.stderr):
        setup()
