"""Script comprising the entry point to the backend of themis"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import logging
import functools
import argparse
import time
import contextlib
import signal
import socket

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 3)[0])


# pylint: disable=wrong-import-position
from themis import database
from themis import utils
from themis import resource
from themis import backend
from themis.backend import network
from themis.backend import RunFeeder, SignalCaught


LOGGER = logging.getLogger(__name__)


def create_new_allocation(status_store, resource_mgr, args):
    """Create a new allocation to follow this one."""
    curr_allocation = status_store.decrement_repeats(args.alloc_id)
    if curr_allocation.repeats == -1:
        LOGGER.info("No remaining allocation repeats. Exiting...")
        return
    LOGGER.info("Replicating existing allocation...")
    backend_commands = [
        resource_mgr.commands_to_launch_backend(
            curr_allocation.timeout,
            args.parallelism,
            args.alloc_id,
            args.early_stop,
            args.multiple,
            args.setup_dir,
            args.max_concurrency,
        )
    ]
    resource_mgr.allocator().start(
        curr_allocation, backend_commands, os.getcwd(), os.getcwd()
    )


@contextlib.contextmanager
def manage_workers(launch_workers):
    """Context manager for terminating workers on error."""
    workers = launch_workers()
    try:
        yield workers
    finally:
        workers.terminate(wait=0.3)


def backend_event_loop(run_feeder, server, ensemble_db, launch_workers, exit_condition):
    """The primary event loop driving top-level ensemble progress.

    :param launch_workers: a callable that submits workers that execute the runs
        of the ensemble
    :param exit_condition: a callable that returns True when it is safe to exit
    """
    while True:
        LOGGER.info("Entering new worker round")
        with manage_workers(launch_workers) as workers:
            inner_backend_event_loop(workers, run_feeder, server, ensemble_db)
        LOGGER.info("Worker round complete. Checking for overall completion...")
        if exit_condition() or not run_feeder.fill_cache():
            break
        LOGGER.info("Not yet complete.")
        run_feeder.fill_cache()


def inner_backend_event_loop(workers, run_feeder, server, ensemble_db):
    """An event loop contained within `backend_event_loop`"""
    curr_time = time.time()
    time_of_last_save = curr_time
    save_frequency = 1  # write the server's cache to the database every second
    while workers.poll() is None and server.is_alive():
        run_feeder.poll_cache()
        if curr_time - time_of_last_save > save_frequency:
            server.save_progress(ensemble_db)
        time.sleep(0.01)
        curr_time = time.time()
    server.save_progress(ensemble_db)
    if not server.is_alive():
        raise IOError("Server {!r} died".format(server))


def setup_ensemble(status_store, app_interface, root_dir):
    """Perform ensemble setup: call `prep_ensemble` if necessary."""
    prep_ensemble = getattr(app_interface, "prep_ensemble", None)
    if callable(prep_ensemble) and not status_store.fetch()["prepped"]:
        LOGGER.info("Calling user's prep_ensemble")
        backend.user_prep_ensemble(prep_ensemble, root_dir)
        status_store.set(prepped=True)
    else:
        LOGGER.info("No prep_ensemble to call. Proceeding...")


def can_exit(app_interface, status_store, max_failed_runs, root_dir, on_completion):
    """Return True if it is permissible to exit the submission loop."""
    status = status_store.fetch()
    if status["stop"]:
        LOGGER.info("'Stop ensemble' message received. Exiting...")
        return True
    if status["failed_runs"] >= max_failed_runs > -1:
        LOGGER.info("Max failed runs exceeded. Exiting...")
        return True
    post_ensemble = getattr(app_interface, "post_ensemble", None)
    exit_ok = True  # ok to exit unless post_ensemble or on_completion defined
    # post_ensemble and on_completion may add new runs
    if isinstance(on_completion, dict) and on_completion.get("args") is not None:
        backend.invoke_completion_script(on_completion)
        exit_ok = False
    if callable(post_ensemble):
        backend.user_post_ensemble(post_ensemble, root_dir)
        exit_ok = False
    if exit_ok:
        LOGGER.info("No post_ensemble/on_completion to call. Exiting...")
    return exit_ok


def use_ssl():
    """Create an SSL certificate for use by servers"""
    try:
        network.create_ssl_certificate()
    except OSError:
        LOGGER.exception("Couldn't generate SSL certificate:")


def early_stop_grace_period(runtime_end, grace_period_mins):
    """Go to sleep to keep an allocation alive without submitting any new runs."""
    grace_period_secs = grace_period_mins * 60
    grace_period_end = runtime_end + grace_period_secs
    remaining_grace_period_time = grace_period_end - time.time()
    if remaining_grace_period_time > 0:
        LOGGER.info("Entering grace period of %f seconds", remaining_grace_period_time)
        time.sleep(remaining_grace_period_time)


@contextlib.contextmanager
def allocation_manager(status_store, rmgr, runtime_end, args):
    """Upon catching a signal, enter the grace period and then create a new alloc."""
    try:
        yield
    except SignalCaught:
        early_stop_grace_period(runtime_end, args.early_stop)
        create_new_allocation(status_store, rmgr, args)


def setup_parser():
    """Setup the command-line parser for the script."""
    parser = argparse.ArgumentParser(
        description=(
            "Themis backend. Requires a database "
            "holding information about the ensemble."
        )
    )
    parser.add_argument(
        "-t",
        "--runtime",
        metavar="N",
        type=float,
        help="the max number of minutes to run for",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--early-stop",
        metavar="N",
        type=float,
        help="the number of minutes early to stop executing runs",
        default=0,
    )
    parser.add_argument(
        "-a",
        "--alloc-id",
        metavar="N",
        type=int,
        help="integer ID identifying allocation in status store",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--parallelism",
        metavar="N",
        type=int,
        help="the number of workers to launch",
        default=0,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--max-concurrency",
        metavar="N",
        type=int,
        help="the maximum number of concurrent runs to allow",
        required=True,
    )
    parser.add_argument(
        "--setup-dir", help="Path to the Themis setup directory", default=os.curdir,
    )
    parser.add_argument(
        "--multiple",
        help=(
            "Allow multiple concurrent executions, disabling "
            "the automatic freeing of all queued runs"
        ),
        action="store_true",
    )
    return parser


def collect_args_set_timer():
    """Parse command-line arguments and set a runtime SIGALRM timer."""
    use_ssl()
    args = setup_parser().parse_args()
    args.setup_dir = os.path.abspath(args.setup_dir)
    LOGGER.info(
        "Running on %s for %f minutes, stopping run submission %i minutes early",
        socket.gethostname(),
        args.runtime,
        args.early_stop,
    )
    LOGGER.info("Running with parallelism %i", args.parallelism)
    if "FLUX_URI" in os.environ:
        LOGGER.info(
            "Flux remote URI is %s, local URI is %s",
            "ssh://"
            + socket.gethostname()
            + os.environ["FLUX_URI"].replace("local://", ""),
            os.environ["FLUX_URI"],
        )
    # catch signals to stop running, e.g. when the batch allocation is expiring
    proper_runtime_seconds = ((args.runtime - args.early_stop) * 60) - 5
    if proper_runtime_seconds >= 0:
        signal.alarm(int(proper_runtime_seconds))  # request an alarm signal
    backend.handle_signals()  # set signal handler
    runtime_end = time.time() + proper_runtime_seconds - 5
    return (args, runtime_end)


def main():
    """Set up and run the backend."""
    args, runtime_end = collect_args_set_timer()
    app_spec = database.get_app_spec(args.setup_dir)
    rmgr = resource.identify_resource_manager(
        app_spec["resource_mgr"], path=app_spec["flux_path"]
    )
    ensemble_db = database.default_database(app_spec)
    if not args.multiple:
        # since we are just starting up Themis, if we don't expect any other
        # concurrent backend processes, enable all runs
        ensemble_db.enable_all()
    app_interface = utils.import_app_interface(app_spec["app_interface"])
    status_store = database.get_status_store(args.setup_dir)
    # now export information and setup the objects used to run
    backend.export_to_user_utils(app_spec, status_store)
    setup_ensemble(status_store, app_interface, app_spec["root_dir"])
    with allocation_manager(status_store, rmgr, runtime_end, args), RunFeeder(
        ensemble_db, args.max_concurrency * 10, enable_runs=True
    ) as run_feeder, backend.managed_shutdown(
        network.BackendServer(network.CERT_NAME, run_feeder), ensemble_db
    ) as server:
        LOGGER.info("Set up server %s running on %s", server, server.server_address)
        backend_event_loop(
            run_feeder,
            server,
            ensemble_db,
            functools.partial(
                rmgr.launch_workers,
                args.parallelism,
                server.server_address,
                args.setup_dir,
                args.max_concurrency,
            ),
            functools.partial(
                can_exit,
                app_interface,
                status_store,
                app_spec["max_failed_runs"],
                app_spec["root_dir"],
                app_spec.get("on_completion"),
            ),
        )


def main_wrapper():
    """Setup the logging facility and wrap _main in a profiler."""
    with backend.managed_logging(sys.stderr):
        profilers = backend.profile_from_new(main)
        backend.dump_profile_data("backend_profile.pstat", *profilers)


if __name__ == "__main__":
    main_wrapper()
