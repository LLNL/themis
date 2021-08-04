"""Utility functions for the backend of themis"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import argparse
import logging
import os

from themis import database
from themis import backend
from themis.backend.network import CERT_NAME
from themis.backend.worker import network
import themis.utils


def profile_worker(main_func, worker):
    """This function provides a layer on top of main(), to allow for profiling main().

    In order to properly direct the profile output, need access to the run ID.
    """
    with backend.managed_logging(sys.stderr):
        run_id, server_args, setup_dir = read_command_line("prep an application run")
        profiler, _ = backend.profile_from_new(
            main_func, run_id, server_args, setup_dir
        )
        if run_id < 1000:
            backend.dump_profile_data(
                "submitter_profile_stats_{}_{}.pstat".format(worker, run_id), profiler
            )


def read_command_line(script_description):
    """Parse worker command line arguments and return them.

    :param script_description: the description to display for --help messages
    """
    parser = argparse.ArgumentParser(description=script_description)
    parser.add_argument(
        "run_id", metavar="N", type=int, help="the (unique) id of this application run"
    )
    parser.add_argument("setup_dir", help="the setup dir for the ensemble")
    parser.add_argument(
        "--server-args",
        metavar="ARG",
        type=str,
        nargs="*",
        help="arguments to connect to the server (if it exists)",
    )
    args = parser.parse_args()
    return (args.run_id, args.server_args, args.setup_dir)


def setup_script(run_id, server_args, setup_dir):
    """Set up a worker script.

    Connects to the ensemble database, collects the sample and app spec, imports the
    application interface, and creates the run directory. Returns all the
    collected information.

    :param run_id: ID identifying the worker
    :param server_args: a sequence of arguments used to connect to the ensemble server
    """
    # setup the logger to log into the new directory, and direct output to the log file
    logger = logging.getLogger(__name__)
    logger.info("Server connection information: %s", server_args)
    if not server_args:
        app_spec = database.get_app_spec(setup_dir)
        ensemble_db = database.default_database(app_spec)
        run_info = ensemble_db.get_run_info(run_id)
    else:
        ensemble_db = network.WorkerClient(
            os.path.join(setup_dir, CERT_NAME), *server_args
        )
        app_spec, run_info = ensemble_db.get_worker_setup(run_id)
    sample = run_info.sample
    # import the application interface
    app_interface = themis.utils.import_app_interface(app_spec["app_interface"])
    run_dir = themis.utils.get_run_dir(run_id, app_spec["run_dir_names"], sample)
    logger.info("Using database: %s", ensemble_db)
    status_db = database.get_status_store()
    backend.export_to_user_utils(app_spec, status_db, run_id, run_info.run)
    return (ensemble_db, app_interface, run_dir, app_spec, run_info)
