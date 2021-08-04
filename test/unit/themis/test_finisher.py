"""
Unit tests for ensemble/worker/finisher.py
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import unittest
import sys
import os
import logging

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

from themis.backend.worker import finisher
from themis.backend import export_to_user_utils, clear_user_utils
from themis.utils import Run


class ShamDatabase(object):
    def __init__(self):
        self.run_id = None
        self.result = None

    def add_result(self, run_id, result):
        self.run_id = run_id
        self.result = result


def post_run_one():
    from themis import user_utils

    return user_utils.run_id()


def post_run_two():
    from themis import user_utils

    return user_utils.run().sample["finisher_tests"]


def post_run_three():
    return os.getcwd()


def post_run_four():
    from themis import user_utils

    return user_utils.run().gpus_per_task


def failing_post_run_one():
    return 1e99


def failing_post_run_two():
    raise RuntimeError("Exception in post_run!")


class FinishRunTests(unittest.TestCase):
    """Unit tests for the finish_run function."""

    @classmethod
    def setUpClass(cls):
        cls.run_id = 17
        cls.app_spec = {"root_dir": os.getcwd(), "setup_dir": None}
        cls.ensemble_db = ShamDatabase()
        cls.ensemble_run = Run({"finisher_tests": 51}, gpus_per_task=71)
        cls.ensemble_run.application = "/foo/bar"
        export_to_user_utils(
            cls.app_spec, cls.ensemble_db, cls.run_id, cls.ensemble_run
        )
        logger = logging.getLogger()
        cls.old_loglevel = logger.getEffectiveLevel()
        logger.setLevel(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logger = logging.getLogger()
        logger.setLevel(cls.old_loglevel)
        clear_user_utils()

    def test_post_run_one(self):
        result = finisher.user_post_run(post_run_one, os.getcwd())
        self.assertEqual(self.run_id, result)

    def test_post_run_two(self):
        result = finisher.user_post_run(post_run_two, os.getcwd())
        self.assertEqual(self.ensemble_run.sample["finisher_tests"], result)

    def test_post_run_three(self):
        run_dir = os.path.join(os.getcwd(), "test_run_dir")
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        result = finisher.user_post_run(post_run_three, run_dir)
        self.assertEqual(run_dir, result)

    def test_post_run_four(self):
        result = finisher.user_post_run(post_run_four, os.getcwd())
        self.assertEqual(self.ensemble_run.gpus_per_task, result)

    def test_failing_post_runs(self):
        with self.assertRaises(RuntimeError):
            finisher.user_post_run(failing_post_run_two, os.getcwd())
        result = finisher.user_post_run(failing_post_run_one, os.getcwd())
        self.assertEqual(result, 1e99)


if __name__ == "__main__":
    unittest.main()
