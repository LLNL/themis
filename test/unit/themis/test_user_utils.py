"""
Unit tests for the user_utils module
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import sys
import os
import shutil

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

from themis import user_utils
from themis import utils
from themis import database
from themis.backend import clear_user_utils, export_to_user_utils


def sham_fetcher(*args, **kwargs):
    return ShamDatabase()


class ShamDatabase(object):

    status_count = 5
    status_iterable = [1, 2, 3]

    def __init__(self):
        self.status = {}

    def add_runs(self, samples):
        pass

    def update_ensemble_status(self, field, value):
        self.status[field] = value

    def count_runs_by_completion(self, *args):
        return self.status_count

    def runs_with_completion_status(self, *args):
        return self.status_iterable


class ShamStatusStore(object):
    def __init__(self):
        self.status = {}

    def set(self, **kwargs):
        for key, val in kwargs.items():
            self.status[key] = val


class UserUtilsTests(unittest.TestCase):
    """Tests for the user_utils module.

    Many of the user_utils functions are expected to behave differently based on the
    caller (prep_run vs prep_ensemble); that should be tested here.
    """

    @classmethod
    def setUpClass(cls):
        cls.app_spec = {
            "run_parse": ["/blah/blah.csv"],
            "run_dir_names": None,
            "root_dir": "blah/blah/blah",
            "database": {
                "type": "sqlite",
                "path": os.path.abspath("ensemble_database.db"),
            },
            "setup_dir": os.getcwd(),
            "version_created": sys.version_info.major,
        }
        cls.run_ids = range(10)
        cls.ensemble_db = ShamDatabase()
        cls.status_store = ShamStatusStore()
        cls.samples = [{"sleep_sec": 2 * i} for i in cls.run_ids]
        cls.runs = [
            utils.CompositeRun(
                sample,
                [
                    utils.Step("/foo", cores_per_task=5, gpus_per_task=7)
                    for _ in range(3)
                ],
            )
            for sample in cls.samples
        ]
        cls.original_default_database_func = staticmethod(database.default_database)
        database.default_database = sham_fetcher
        database.write_app_spec(cls.app_spec)

    @classmethod
    def tearDownClass(cls):
        database.default_database = cls.original_default_database_func

    def tearDown(self):
        clear_user_utils()

    def export(self, run_id=None):
        """Shortcut method for exporting info to user_utils."""
        clear_user_utils()
        if run_id is not None:
            export_to_user_utils(
                self.app_spec, self.status_store, run_id, self.runs[run_id]
            )
        else:
            export_to_user_utils(self.app_spec, self.status_store)

    def test_run_id(self):
        for run_id in self.run_ids:
            self.export(run_id)
            self.assertEqual(user_utils.run_id(), run_id)
        self.export()
        self.assertEqual(user_utils.run_id(), None)

    def test_run(self):
        for run_id in self.run_ids:
            self.export(run_id)
            for step1, step2 in zip(user_utils.run().steps, self.runs[run_id].steps):
                for attr in utils.Step.__slots__:
                    self.assertEqual(getattr(step1, attr), getattr(step2, attr))
        self.export()
        self.assertEqual(user_utils.run(), None)

    def test_results(self):
        self.export()
        results = user_utils.themis_handle()
        self.assertEqual(results.count_by_status(), self.ensemble_db.status_count)
        self.assertEqual(results.filter_by_status(), self.ensemble_db.status_iterable)

    def test_stop_ensemble(self):
        self.export()
        user_utils.stop_ensemble()
        with self.assertRaises(KeyError):
            self.status_store.status["stop"]
        for run_id in self.run_ids:
            self.status_store.status = {}
            self.export(run_id)
            user_utils.stop_ensemble()
            self.assertTrue(self.status_store.status["stop"])


if __name__ == "__main__":
    unittest.main()
