"""Integration tests for the ensemble component which can be run on a laptop.

The aim of this module is to test as much as is feasible for a laptop.
Generally, this means anything short of applications that require MPI or GPUs.

These tests should run quickly (a couple of seconds each). They should also be simple--
if tests are cheap, there's no point making each one complicated.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import os
import sys
import json
import pickle
import warnings
import time
import shlex
import shutil
import string
import multiprocessing as mp

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

import themis.utils
from themis import Themis, database
import themis.resource
from themis.__main__ import setup_parsers
from themis.utils import clean_directory_decorator, CleanDirectory, DEFAULT_SETUP_DIR


# global constants
this_file_location = os.path.dirname(os.path.abspath(__file__)) + os.sep
APPLICATIONS = this_file_location + "Applications" + os.sep
INPUTDECKS = this_file_location + "InputDecks" + os.sep
REQDFILES = this_file_location + "ReqdFiles" + os.sep
INTERFACES = this_file_location + "Interfaces" + os.sep
PARAMETERFILES = this_file_location + "Parameterfiles" + os.sep
XY_PARAMETERFILE = os.path.join(PARAMETERFILES, "xy_123abc.csv")
XY_PARAMETERFILE_LEN = len(list(themis.utils.read_csv(XY_PARAMETERFILE)))


def force_noresourcemanager(themis_instance):
    spec = themis_instance._spec
    spec["resource_mgr"] = themis.resource.NoResourceManager.identifier
    database.write_app_spec(spec, spec["setup_dir"])


class StdoutRedirector(object):
    def __init__(self):
        self.buffer = []

    def write(self, obj):
        self.buffer.append(obj)

    def flush(self):
        pass

    def clear(self):
        del self.buffer[:]

    def as_str(self):
        return "".join(self.buffer)


class ThemisLaptopCLIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.testing_dir = CleanDirectory(cls.__name__)
        cls.testing_dir.go_to_new()
        cls.parser = setup_parsers()
        cls.old_stdout = sys.stdout
        sys.stdout = StdoutRedirector()

    @classmethod
    def tearDownClass(cls):
        cls.testing_dir.go_to_old()
        sys.stdout = cls.old_stdout

    @classmethod
    def call_subcommand(cls, arg_list):
        namespace = cls.parser.parse_args(arg_list)
        namespace.handler(namespace)

    def setUp(self):
        sys.stdout.flush()

    @clean_directory_decorator()
    def test_creation_execution(self):
        self.assertFalse(Themis.exists())
        self.call_subcommand(
            ["create", APPLICATIONS + "laptop_batch_script.sh", XY_PARAMETERFILE]
        )
        self.assertTrue(Themis.exists())
        mgr = Themis()
        force_noresourcemanager(mgr)
        self.call_subcommand(["execute-local", "--block"])
        self.assertEqual(mgr.count_by_status(), XY_PARAMETERFILE_LEN)
        self.assertEqual(mgr.count_by_status(mgr.RUN_SUCCESS), XY_PARAMETERFILE_LEN)
        self.call_subcommand(
            ["create", "sleep", XY_PARAMETERFILE, "--no-batch-script", "-a0"]
        )
        self.assertEqual(mgr.count_by_status(), XY_PARAMETERFILE_LEN)
        self.assertEqual(mgr.count_by_status(mgr.RUN_SUCCESS), XY_PARAMETERFILE_LEN)

    @clean_directory_decorator()
    def test_creation_overwrite(self):
        self.assertFalse(Themis.exists())
        self.call_subcommand(
            [
                "create",
                "sleep",
                XY_PARAMETERFILE,
                "--no-batch-script",
                "-a0",
                "--overwrite",
            ]
        )
        self.assertTrue(Themis.exists())
        mgr = Themis()
        self.assertEqual(mgr.count_by_status(), XY_PARAMETERFILE_LEN)
        self.assertEqual(mgr.count_by_status(mgr.RUN_QUEUED), XY_PARAMETERFILE_LEN)
        self.call_subcommand(
            [
                "create",
                "sleep",
                os.path.join(PARAMETERFILES, "xy_123abc_2.csv"),
                "--no-batch-script",
                "-a0",
                "--overwrite",
            ]
        )
        mgr = Themis()
        self.assertEqual(mgr.count_by_status(), 7)
        self.assertEqual(mgr.count_by_status(mgr.RUN_QUEUED), 7)

    @clean_directory_decorator()
    def test_creation_dryrun(self):
        self.assertFalse(Themis.exists())
        self.call_subcommand(
            [
                "create",
                "sleep",
                XY_PARAMETERFILE,
                "--no-batch-script",
                "-a0",
                "--symlink",
                XY_PARAMETERFILE,
            ]
        )
        self.assertTrue(Themis.exists())
        run_dirs = [
            os.path.join(os.getcwd(), "runs", str(run_id)) for run_id in range(1, 6)
        ]
        for run_dir in run_dirs:
            self.assertFalse(os.path.exists(run_dir))
        mgr = Themis()
        self.call_subcommand(["dryrun"])
        for run_dir in run_dirs:
            self.assertTrue(
                os.path.islink(
                    os.path.join(run_dir, os.path.basename(XY_PARAMETERFILE))
                )
            )

    @clean_directory_decorator()
    @unittest.skip("as_str() is not a member method of sys.stdout.  Unclear as to the purpose of the test.")
    def test_display(self):
        self.assertFalse(Themis.exists())
        self.call_subcommand(
            ["create", "sleep", XY_PARAMETERFILE, "--no-batch-script", "-a0"]
        )
        self.call_subcommand(["display", "0", "20", "--all", "-c1"])
        lines = sys.stdout.as_str().split("\n")
        line_len = len(lines[1])
        for line in lines[1:-2]:
            self.assertEqual(len(line), line_len)
        sys.stdout.flush()
        self.call_subcommand(["display", "0", "20", "--all", "-c5"])
        new_lines = sys.stdout.as_str().split("\n")
        new_line_len = len(new_lines[1])
        self.assertGreater(new_line_len, line_len)
        for line in new_lines[1:-2]:
            self.assertEqual(len(line), new_line_len)

    @clean_directory_decorator()
    def test_add(self):
        self.call_subcommand(["create", "sleep", "--no-batch-script"])
        mgr = Themis()
        self.call_subcommand(
            ["add", XY_PARAMETERFILE, "-afoobar", "-n16", "-c17", "-g18"]
        )
        self.assertEqual(mgr.count_by_status(), XY_PARAMETERFILE_LEN)
        self.assertEqual(mgr.count_by_status(mgr.RUN_QUEUED), XY_PARAMETERFILE_LEN)
        self.call_subcommand(
            ["add", XY_PARAMETERFILE, "-afoobar", "-n16", "-c17", "-g18"]
        )
        self.assertEqual(mgr.count_by_status(), 10)
        self.assertEqual(mgr.count_by_status(mgr.RUN_QUEUED), 10)
        runs = mgr.runs(range(1, 11))
        for run in runs.values():
            self.assertEqual(
                run.sample["Y"], string.ascii_letters[int(run.sample["X"])]
            )
            self.assertEqual(run.steps[0].args[1], "foobar")
            self.assertEqual(run.steps[0].tasks, 16)
            self.assertEqual(run.steps[0].cores_per_task, 17)
            self.assertEqual(run.steps[0].gpus_per_task, 18)

    @clean_directory_decorator()
    def test_add_vary_all(self):
        self.call_subcommand(["create", "sleep", "--no-batch-script"])
        mgr = Themis()
        self.call_subcommand(
            [
                "add",
                os.path.join(PARAMETERFILES, "xy_123abc_vary_all.csv"),
                "--vary-all",
                "-afoobar",
                "-n16",
                "-c17",
                "-g18",
            ]
        )
        for run_id, run in mgr.runs(range(1, 6)).items():
            self.assertEqual(
                run.sample["Y"], string.ascii_letters[int(run.sample["X"])]
            )
            self.assertEqual(run.steps[0].args[1], str(run_id))
            self.assertEqual(run.sample["args"], run.steps[0].args[1])
            self.assertEqual(run.steps[0].tasks, run_id)
            self.assertEqual(run.sample["tasks"], str(run.steps[0].tasks))
            self.assertEqual(run.steps[0].cores_per_task, 17)
            self.assertEqual(run.steps[0].gpus_per_task, run_id)
            self.assertEqual(
                run.sample["gpus_per_task"], str(run.steps[0].gpus_per_task)
            )

    @clean_directory_decorator()
    def test_kill_restart(self):
        self.assertFalse(Themis.exists())
        self.call_subcommand(["create", "sleep", XY_PARAMETERFILE, "--no-batch-script"])
        mgr = Themis()
        self.call_subcommand(["kill"] + "1 2 3 4 5 6 7".split())
        self.assertEqual(mgr.count_by_status(), XY_PARAMETERFILE_LEN)
        self.assertEqual(mgr.count_by_status(mgr.RUN_KILLED), XY_PARAMETERFILE_LEN)
        self.call_subcommand(["restart"] + "1 2 3 4 5 6 7".split())
        self.assertEqual(mgr.count_by_status(mgr.RUN_QUEUED), XY_PARAMETERFILE_LEN)

    @clean_directory_decorator()
    def test_post(self):
        self.call_subcommand(
            [
                "create",
                "sleep",
                XY_PARAMETERFILE,
                "--no-batch-script",
                "--interface",
                INTERFACES + "adding_tuple_results.py",
            ]
        )
        mgr = Themis()
        for run in mgr.runs(range(1, 11)).values():
            self.assertEqual(run.result, None)
        self.call_subcommand(["post"] + "1 2 3 4 5 -s".split())
        run_dirs = [
            os.path.join(os.getcwd(), "runs", str(run_id)) for run_id in range(1, 6)
        ]
        for run_dir in run_dirs:
            self.assertTrue(os.path.exists(run_dir))
        for run in mgr.runs(range(1, 11)).values():
            self.assertEqual(run.result, (1, 2, 3))

    @clean_directory_decorator()
    def test_on_completion(self):
        self.assertFalse(Themis.exists())
        os.mkdir("echo_cwd")
        self.call_subcommand(
            [
                "create",
                "sleep",
                XY_PARAMETERFILE,
                "--no-batch-script",
                "-a0",
                "--symlink",
                XY_PARAMETERFILE,
            ]
        )
        self.assertTrue(Themis.exists())
        self.call_subcommand(
            [
                "completion",
                "--stdout",
                "echo_stdout.txt",
                "--stderr",
                "echo_stderr.txt",
                "--cwd",
                "echo_cwd",
                "echo",
                "hello",
                "world",
            ]
        )
        force_noresourcemanager(Themis())
        self.call_subcommand(["execute-local", "--block"])
        self.assertTrue(os.path.isfile("echo_stdout.txt"))
        self.assertTrue(os.path.isfile("echo_stderr.txt"))
        with open("echo_stdout.txt") as file_handle:
            self.assertTrue("hello world" in file_handle.read())

    @clean_directory_decorator()
    def test_abort_on(self):
        self.assertFalse(Themis.exists())
        self.call_subcommand(
            [
                "create",
                "bash",
                XY_PARAMETERFILE,
                "--no-batch-script",
                "-a-c 'exit 50'",
                "--symlink",
                XY_PARAMETERFILE,
                "--abort-on",
                "50",
            ]
        )
        self.assertTrue(Themis.exists())
        mgr = Themis()
        force_noresourcemanager(mgr)
        self.call_subcommand(["execute-local", "--block"])
        self.assertEqual(mgr.count_by_status(), XY_PARAMETERFILE_LEN)
        self.assertEqual(mgr.count_by_status(mgr.RUN_ABORTED), XY_PARAMETERFILE_LEN)

    @clean_directory_decorator()
    def test_create_composite(self):
        stepfile = os.path.join(PARAMETERFILES, "stepfile.csv")
        self.assertFalse(Themis.exists())
        self.call_subcommand(["create-composite", XY_PARAMETERFILE, stepfile])
        self.assertTrue(Themis.exists())
        mgr = Themis()
        self.call_subcommand(["add-composite", XY_PARAMETERFILE, stepfile])
        force_noresourcemanager(mgr)
        self.call_subcommand(["execute-local", "--block"])
        self.assertEqual(mgr.count_by_status(), XY_PARAMETERFILE_LEN * 2)
        self.assertEqual(mgr.count_by_status(mgr.RUN_SUCCESS), XY_PARAMETERFILE_LEN * 2)
        for run_id, run_dir in mgr.run_dirs():
            for i in range(len(list(themis.utils.read_csv(stepfile)))):
                self.assertTrue(
                    os.path.isfile(os.path.join(run_dir, "step" + str(i), "run.log"))
                )


if __name__ == "__main__":
    unittest.main()
