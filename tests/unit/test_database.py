"""Unit tests for ensemble/database.py"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import unittest
import sys
import os
import sqlite3 as sql
from decimal import Decimal

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

from themis import database
from themis.database.sqlitedb import SQLiteDatabase
import themis as manager
from themis import utils


class DatabaseMixin:
    """Base class for running unit tests on concrete database implementations."""

    @classmethod
    def database(cls):
        pass


class SqliteMixin(DatabaseMixin):
    """Mixin for running unit tests on the SQLite database."""

    @classmethod
    def database(cls):
        db = SQLiteDatabase("ensemble_database_two.db")
        db.delete()
        return db


class UtilsTests(unittest.TestCase):
    """Tests for module-level functions in the database module."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.assertCountEqual = cls.assertItemsEqual
        except AttributeError:
            pass

    def test_get_app_spec(self):
        """Test the get_app_spec and write_app_spec module-level functions."""
        app_spec_one = {
            "application"        : "application_example.sh",
            "run_parse"          : ("run_parse", "another_input_deck"),
            "app_interface"      : "example_interface.py",
            "run_symlink"        : ("required_script.py", "blah.py",),
            "app_is_batch_script": True,
            "run_dir_names"      : None,
            "setup_dir"          : "blah",
            "max_concurrency"    : 5,
            "max_restarts"       : 10,
            "resource_mgr"       : "flux",
        }
        app_spec_two = {
            "application": "application_example.sh",
            "run_parse": (),
            "app_interface": "example_interface.py",
            "run_symlink": (),
            "app_is_batch_script": False,
            "run_dir_names": "my_run_dirs",
            "setup_dir": "blah",
            "max_concurrency": 5,
            "max_restarts": 10,
            "resource_mgr": None,
        }
        for app_spec in (app_spec_one, app_spec_two):
            database.write_app_spec(app_spec)
            db_app_spec = database.get_app_spec()
            for key, value in app_spec.items():
                if isinstance(value, Sequence) and not isinstance(value, str):
                    self.assertCountEqual(value, db_app_spec[key])
                else:
                    self.assertEqual(value, db_app_spec[key])


class DatabaseTests(object):
    """Tests for EnsembleDatabase subclasses.

    The entire class shares a single database, but it is cleared
    after each test method, so test methods should not interfere with each other.

    However, if the clearing fails or has a bug, expect strange results from the tests.
    """

    @classmethod
    def setUpClass(cls):
        cls.db = cls.database()
        cls.db.delete()

    def setUp(self):
        """Set up the database used for the tests"""
        self.samples = utils.validate_samples([{"db_test": "blah"} for _ in range(10)])
        self.run_ids = list(range(1, len(self.samples) + 1))
        self.run_ids_to_samples = dict(zip(self.run_ids, self.samples))
        # prepare the ensemble info
        self.resources = {
            "default": {
                "cores_per_task": 17,
                "tasks"         : 1,
                "gpus_per_task" : 20,
                "timeout"       : 10,
            },
            1        : {"cores_per_task": 19, "tasks": 50, "gpus_per_task": 0, "timeout": 10},
            4        : {"cores_per_task": 21, "tasks": 50, "gpus_per_task": 0, "timeout": 10},
        }
        self.runs = [
            utils.CompositeRun(
                sample,
                [
                    utils.Step(
                        "/foo/bar",
                        **self.resources.get(i + 1, self.resources["default"])
                    )
                ],
            )
            for i, sample in enumerate(self.samples)
        ]
        self.db.delete()
        self.db.create()
        self.db.add_runs(self.runs)
        try:
            self.assertCountEqual = self.assertItemsEqual
        except:
            pass

    def tearDown(self):
        self.db.delete()

    def check_no_new_runs(self):
        self.assertEqual(len(self.db.new_runs(100)), 0)

    def check_run_info(self, run_info, status=0):
        self.assertEqual(run_info.sample, self.run_ids_to_samples[run_info.run_id])
        self.assertEqual(run_info.status, status)
        self.assertEqual(run_info.restarts, 0)
        self.assertEqual(run_info.steps[0].args[0], "/foo/bar")
        for key in ("tasks", "cores_per_task", "gpus_per_task", "timeout"):
            self.assertEqual(
                getattr(run_info.steps[0], key),
                self.resources.get(run_info.run_id, self.resources["default"])[key],
            )

    def collect_all_free_runs(self):
        total_runs = len(self.run_ids)
        new_runs = self.db.new_runs(total_runs)
        self.assertEqual(
            sorted([run_info.run_id for run_info in new_runs]), self.run_ids
        )
        self.check_no_new_runs()
        return new_runs

    def test_new_runs(self):
        new_runs = self.collect_all_free_runs()
        for run_info in new_runs:
            self.check_run_info(run_info)

    def test_get_run_info(self):
        for run_id in self.run_ids:
            run_info = self.db.get_run_info(run_id)
            self.assertEqual(run_id, run_info.run_id)
            self.check_run_info(run_info)

    def test_mark_runs_to_restart(self):
        self.collect_all_free_runs()
        self.db.mark_runs_to_kill(self.run_ids)
        self.check_no_new_runs()
        self.db.mark_runs_to_restart(self.run_ids, True)
        new_runs = self.collect_all_free_runs()
        self.assertEqual(
            sorted([run_info.run_id for run_info in new_runs]), self.run_ids
        )

    def test_mark_runs_to_kill(self):
        self.db.mark_runs_to_kill(self.run_ids)
        self.assertEqual(
            self.db.count_runs_by_completion(self.db.RUN_KILLED), len(self.run_ids),
        )
        self.assertEqual(
            sorted(self.db.runs_with_completion_status(self.db.RUN_KILLED)),
            self.run_ids,
        )

    def test_enable_disable_runs(self):
        self.db.disable_runs(self.run_ids)
        self.assertEqual(len(self.db.new_runs(len(self.run_ids))), 0)
        enabled_runs = self.run_ids[:4]
        self.db.enable_runs(enabled_runs)
        new_runs = self.db.new_runs(len(self.run_ids))
        self.assertEqual(len(new_runs), len(enabled_runs))
        self.assertEqual(
            sorted([run_info.run_id for run_info in new_runs]), enabled_runs
        )
        self.db.enable_runs(self.run_ids)
        self.collect_all_free_runs()

    def test_add_runs(self):
        orig_runs = self.collect_all_free_runs()
        self.db.add_runs(self.runs)
        new_runs = self.db.new_runs(len(self.runs))
        self.check_no_new_runs()
        self.assertEqual(len(new_runs), len(self.runs))

    def test_add_get_result(self):
        """Test adding results to, and fetching results from, the database"""
        for run_id in self.run_ids:
            for result in (6, (1, 2, 3), [1, 2, 4], {"a": 12, "b": Decimal("134.5")}):
                self.db.add_result(run_id, result)
                self.assertEqual(self.db.get_result(run_id)[run_id], result)

    def test_set_run_status(self):
        for run_id in self.run_ids:
            run_info = self.db.get_run_info(run_id)
            self.check_run_info(run_info)
            for step in range(5):
                self.db.set_run_status((run_id,), ((step, self.db.RUN_QUEUED, 0),))
                run_info = self.db.get_run_info(run_id)
                self.check_run_info(run_info, status=step)

    def test_runs_with_completion_status(self):
        for completion_status in (
            self.db.RUN_KILLED,
            self.db.RUN_SUCCESS,
            self.db.RUN_FAILURE,
        ):
            for run_id in self.run_ids:
                self.assertEqual(
                    sorted(
                        list(self.db.runs_with_completion_status(self.db.RUN_QUEUED))
                    ),
                    self.run_ids,
                )
                self.assertEqual(
                    list(self.db.runs_with_completion_status(completion_status)), []
                )
                self.db.set_run_status((run_id,), ((0, completion_status, 0),))
                self.assertEqual(
                    list(self.db.runs_with_completion_status(completion_status)),
                    [run_id],
                )
                # set the run back to original status of 0
                self.db.set_run_status((run_id,), ((0, self.db.RUN_QUEUED, 0),))


class SqliteDatabaseTests(SqliteMixin, DatabaseTests, unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
