"""Tests for ensemble/themis.py, the primary user interface."""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import sys
import os
import warnings

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

import themis
from themis import utils


class BadInputTests(unittest.TestCase):
    """Tests for bad input to the themis.Themis.create method."""

    RUNS = [themis.Run({"X": 1, "Y": 2}) for _ in range(5)]

    @classmethod
    def setUpClass(cls):
        cls.clean_dir = utils.CleanDirectory("manager_unit_tests")
        cls.clean_dir.go_to_new()

    @classmethod
    def tearDownClass(cls):
        cls.clean_dir.go_to_old()

    def setUp(self):
        themis.Themis.clear()

    def test_bad_application(self):
        mgr = themis.Themis.create("python", self.RUNS)
        bad_application = "not_an_app"
        if os.path.exists(bad_application):
            os.remove(bad_application)
        with self.assertRaises(Exception):
            mgr = themis.Themis.create(bad_application, self.RUNS)
        # test something not executable
        with open(bad_application, "w") as file_handle:
            file_handle.write("Hello!")
        with self.assertRaises(Exception):
            mgr = themis.Themis.create(bad_application, self.RUNS)
        with self.assertRaises(Exception):
            mgr = themis.Themis.create(5, self.RUNS)

    def test_bad_resources(self):
        mgr = themis.Themis.create("python", self.RUNS)
        themis.Themis.clear()
        with self.assertRaises(TypeError):
            mgr = themis.Themis.create("python", 5)

    def test_bad_restarts(self):
        mgr = themis.Themis.create("python", self.RUNS, max_restarts=-1)
        themis.Themis.clear()
        with self.assertRaises(ValueError):
            mgr = themis.Themis.create("python", self.RUNS, max_restarts=-6)

    def test_bad_failed_runs(self):
        mgr = themis.Themis.create("python", self.RUNS, max_failed_runs=None)
        themis.Themis.clear()
        with self.assertRaises(ValueError):
            mgr = themis.Themis.create("python", self.RUNS, max_failed_runs=-6)

    def test_bad_run_symlink(self):
        mgr = themis.Themis.create("python", self.RUNS, run_symlink=())
        themis.Themis.clear()
        with self.assertRaises(TypeError):
            mgr = themis.Themis.create("python", self.RUNS, run_symlink=5)
        with self.assertRaises(ValueError):
            mgr = themis.Themis.create("python", self.RUNS, run_symlink=("not_a_file",))

    def test_bad_input_deck(self):
        mgr = themis.Themis.create("python", self.RUNS, run_parse=())
        themis.Themis.clear()
        with self.assertRaises(TypeError):
            mgr = themis.Themis.create("python", self.RUNS, run_parse=5)
        with self.assertRaises(ValueError):
            mgr = themis.Themis.create("python", self.RUNS, run_parse=("not_a_file",))

    def test_overlapping_symlinks_and_input_deck(self):
        # need existing files--use the scripts in this directory
        # since the cwd changed in setUpClass, need to modify __file__
        script_dir = os.path.join("..", os.path.dirname(__file__))
        reqd_files = [
            os.path.join(script_dir, file_name)
            for file_name in os.listdir(script_dir)
            if os.path.isfile(os.path.join(script_dir, file_name))
        ]
        input_deck = list(reqd_files)
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            mgr = themis.Themis.create(
                "python", self.RUNS, run_symlink=reqd_files, run_parse=input_deck
            )
            # Verify we caught. the right warning
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue("should be distinct" in str(w[-1].message))
        # reqd files should all be removed, giving preference to the input deck
        self.assertEqual(len(mgr._spec["run_symlink"]), 0)


if __name__ == "__main__":
    unittest.main()
