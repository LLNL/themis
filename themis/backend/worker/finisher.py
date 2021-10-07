"""This script wraps up a finished application run.

Most of what this script does
revolves around calling the user's post_run method and responding to the result.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import logging

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.insert(0, os.path.abspath(__file__).rsplit(os.sep, 4)[0])

# pylint: disable=wrong-import-position
from themis.utils import DirectoryManager
from themis.backend.worker import utils
from themis import versions

_FAILURE = 1e99


def user_post_run(post_run, run_dir):
    """Call the user's post_run function, if it exists."""
    if callable(post_run):
        with DirectoryManager(run_dir):
            return post_run()
    return None


def main(run_id, server_args, setup_dir):
    """Invoke the user's post_run application interface function.

    Store the result in the database indirectly by passing it to a
    Themis server.
    """
    run_db, app_interface, run_dir, _, _ = utils.setup_script(
        run_id, server_args, setup_dir
    )
    logger = logging.getLogger(__name__)
    post_run = getattr(app_interface, "post_run", None)
    logger.info("Calling user's post_run")
    post_run_result = user_post_run(post_run, run_dir)  # don't shield exceptions
    if post_run_result == _FAILURE:
        raise ValueError("Received {} from post_run, exiting...".format(_FAILURE))
    if post_run_result is not None:
        logger.info(
            "Received from post_run: %s", versions.reprlib.repr(post_run_result)
        )
        run_db.add_result(run_id, post_run_result)
    logger.info("Result passed off successfully, exiting normally...")


if __name__ == "__main__":
    utils.profile_worker(main, "finisher")
