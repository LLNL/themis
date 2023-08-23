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
import shutil
import multiprocessing as mp

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

import themis
from themis import database
from themis import resource
from themis.resource import allocators
import themis.resource
from themis import utils
from themis import laf

# global constants
this_file_location = os.path.dirname(os.path.abspath(__file__)) + os.sep
APPLICATIONS = this_file_location + "Applications" + os.sep
INPUTDECKS = this_file_location + "InputDecks" + os.sep
REQDFILES = this_file_location + "ReqdFiles" + os.sep
INTERFACES = this_file_location + "Interfaces" + os.sep
THEMIS_DIR = os.path.abspath(os.path.dirname(themis.__file__))


def run_id_range(total_runs):
    return range(1, total_runs + 1)


def run_id_range2(lowerbound, upperbound):
    return range(lowerbound + 1, upperbound + 1)


def force_noresourcemanager(themis_instance):
    spec = themis_instance._spec
    spec["resource_mgr"] = themis.resource.NoResourceManager.identifier
    database.write_app_spec(spec, spec["setup_dir"])


def execute_blocking(themis_instance, **kwargs):
    force_noresourcemanager(themis_instance)
    themis_instance._execute(
        allocators.Allocation(timeout=kwargs.pop("timeout", None), repeats=0),
        allocator=allocators.InteractiveAllocator(),
        debug=True,
        blocking=True,
        **kwargs
    )


class ThemisLaptopIntegrationTests(unittest.TestCase):
    """Test cases for the ensemble manager on laptop-ready examples."""

    @classmethod
    def setUpClass(cls):
        cls.testing_dir = utils.CleanDirectory(cls.__name__)
        cls.testing_dir.go_to_new()

    @classmethod
    def tearDownClass(cls):
        cls.testing_dir.go_to_old()

    def validate_success(self, total_runs=None, mgr=None):
        mgr = themis.Themis() if mgr is None else mgr
        self.assertEqual(
            list(run_id_range(total_runs)),
            sorted(list(mgr.filter_by_status(mgr.RUN_SUCCESS))),
        )
        self.assertEqual(total_runs, mgr.count_by_status(mgr.RUN_SUCCESS))
        self.assertEqual(
            [],
            list(
                mgr.filter_by_status(mgr.RUN_FAILURE, mgr.RUN_KILLED, mgr.RUN_QUEUED, )
            ),
        )
        progress = mgr.progress()
        for entry in progress:
            self.assertEqual(entry, total_runs)

    def token_ensemble(self, run_ids):
        runs = [themis.Run({}, "0") for _ in run_ids]
        mgr = themis.Themis.create(
            runs=runs, application="sleep", app_is_batch_script=False
        )
        return mgr

    @utils.clean_directory_decorator()
    def test_input_deck_parsing(self):
        input_deck = APPLICATIONS + "to_parse.py"
        runs = [
            themis.Run({"X": 1, "Y": 2}, os.path.basename(input_deck)) for _ in range(5)
        ]
        mgr = themis.Themis.create(
            runs=runs,
            application="python3",
            run_parse=input_deck,
            app_interface=INTERFACES + "csv_reader_interface.py",
            run_symlink=REQDFILES + "required_script.py",
            app_is_batch_script=False,
        )
        execute_blocking(mgr)
        self.validate_success(len(runs))

    @utils.clean_directory_decorator()
    def test_run_dir_names(self):
        """Test a do-nothing application, but with custom run dir names.

        Only checks that the runs have the right names.
        """
        run_dir_names = os.path.join(
            os.getcwd(), "runs", "viscocity={viscocity}", "hydrostatics={hydrostatics}",
        )
        samples = [{"viscocity": i // 2, "hydrostatics": i} for i in range(20)]
        # prepare the ensemble info
        runs = [themis.Run(sample, "0") for sample in samples]
        mgr = themis.Themis.create(
            runs=runs,
            application="sleep",
            run_dir_names=run_dir_names,
            app_is_batch_script=False,
        )
        execute_blocking(mgr)
        self.validate_success(len(samples))
        for sample in samples:
            self.assertTrue(os.path.exists(run_dir_names.format(**sample)))

    @utils.clean_directory_decorator()
    def test_custom_run_dirs(self):
        """Test case where the run directories already exist, and are named according to the sample values.

        The application is really just a sham, and executes a custom application based on its command-line argument.
        In this case, the custom application is always just a bash script with the same name as the run directory;
        it just prints "5" to "result.txt".
        """
        materials = ["iron", "bronze", "copper"]
        for run_dir_name in materials:
            os.mkdir(run_dir_name)
            shutil.copyfile(
                APPLICATIONS + "redirected_application.sh",
                os.path.join(run_dir_name, run_dir_name + ".sh"),
            )
            shutil.copymode(
                APPLICATIONS + "redirected_application.sh",
                os.path.join(run_dir_name, run_dir_name + ".sh"),
            )
        # prepare the ensemble info
        runs = [themis.Run({"materials": mat}, "./" + mat + ".sh") for mat in materials]
        mgr = themis.Themis.create(
            runs=runs,
            application=APPLICATIONS + "redirecting_application.sh",
            run_dir_names=os.path.join(os.getcwd(), "{materials}"),
            app_is_batch_script=False,
        )
        execute_blocking(mgr)
        self.validate_success(len(runs))
        self.assertEqual(
            set(run_dir for _, run_dir in mgr.run_dirs()),
            set(os.path.abspath(material) for material in materials),
        )
        for _, run_dir in mgr.run_dirs():
            with open(os.path.join(run_dir, "result.txt")) as file_handle:
                self.assertEqual(5, int(file_handle.readlines()[0]))

    @utils.clean_directory_decorator()
    def test_debug_methods(self):
        """Test the Themis dry_run and call_post_run methods."""
        total_runs = 10
        foo_val = 6
        runs = [themis.Run({"foo": foo_val}, "0") for i in range(total_runs)]
        mgr = themis.Themis.create(
            runs=runs,
            application="sleep",
            app_interface=INTERFACES + "debug_tests_interface.py",
            app_is_batch_script=False,
        )
        self.assertEqual(mgr.call_prep_ensemble(), None)
        with open("debug_tests_prep_ensemble.pkl", "rb") as file_handle:
            self.assertEqual(0, pickle.load(file_handle))
        for run_id in run_id_range(total_runs):
            self.assertEqual(None, mgr.dry_run(run_id))
            self.assertEqual(10 * foo_val, mgr.call_post_run(run_id))
        execute_blocking(mgr)
        self.validate_success(total_runs)
        # now test that the same methods work when restarting
        mgr = themis.Themis()
        for run_id in run_id_range(total_runs):
            self.assertEqual(None, mgr.dry_run(run_id))
            self.assertEqual(10 * foo_val, mgr.call_post_run(run_id))
        self.assertEqual(mgr.call_post_ensemble(), None)
        with open("debug_tests_post_ensemble.pkl", "rb") as file_handle:
            self.assertEqual(total_runs, pickle.load(file_handle))

    @utils.timeout_decorator(120)
    @utils.clean_directory_decorator()
    def test_killing_runs(self):
        """Test case in which the application runs forever, and has to be killed."""
        # prepare the ensemble info
        run_ids = run_id_range(20)
        run_ids_to_kill = run_ids[:10]
        successful_run_ids = list(set(run_ids) - set(run_ids_to_kill))
        runs = [themis.Run({}, "0") for _ in run_ids]
        mgr = themis.Themis.create(
            runs=runs,
            application="sleep",
            app_interface=INTERFACES + "killing_runs_interface.py",
            app_is_batch_script=False,
        )
        # make sure the manager checks for runs to kill more quickly
        mgr._spec["save_interval"] = 1
        database.write_app_spec(mgr._spec, mgr._spec["setup_dir"])
        force_noresourcemanager(mgr)
        alloc = mgr._execute(
            allocators.Allocation(timeout=1),
            allocator=allocators.InteractiveAllocator(),
            debug=True,
        )
        while not mgr.count_by_status(mgr.RUN_SUCCESS):
            time.sleep(0.1)
        mgr.kill_runs(run_ids_to_kill)
        alloc.wait()
        self.assertEqual(
            list(run_ids_to_kill), sorted(mgr.filter_by_status(mgr.RUN_KILLED)),
        )
        self.assertEqual(
            list(successful_run_ids), sorted(mgr.filter_by_status(mgr.RUN_SUCCESS)),
        )

    @utils.clean_directory_decorator()
    def test_allocation_repeats(self):
        """Test the ensemble manager's ability to relaunch itself when
        its allocation expires.
        """
        run_ids = run_id_range(10)
        runs = [themis.Run({}, "2") for _ in run_ids]
        mgr = themis.Themis.create(
            runs=runs, application="sleep", app_is_batch_script=False,
        )
        force_noresourcemanager(mgr)
        mgr._execute(
            allocators.Allocation(repeats=3, timeout=0.15),
            allocator=allocators.InteractiveAllocator(),
            blocking=True,
            max_concurrency=5,
        )
        self.assertTrue(mgr.count_by_status(mgr.RUN_SUCCESS) < len(run_ids))
        self.assertTrue(mgr.count_by_status(mgr.RUN_QUEUED) > 0)
        max_runtime = 30 + time.time()
        while time.time() < max_runtime:
            if mgr.progress()[0] >= len(run_ids):
                break
            time.sleep(0.5)
        self.validate_success(len(run_ids))

    @utils.clean_directory_decorator()
    def test_restarts_manual(self):
        """Test a manual restart of a failed ensemble.

        The application is rigged to fail the first time it is executed.
        """
        run_ids = run_id_range(10)
        runs = [themis.Run({}, APPLICATIONS + "restarts.py") for _ in run_ids]
        mgr = themis.Themis.create(
            runs=runs, application="python", max_restarts=0, app_is_batch_script=False,
        )
        execute_blocking(mgr)
        self.assertEqual(mgr.count_by_status(mgr.RUN_FAILURE), len(run_ids))
        mgr.restart_runs(mgr.filter_by_status(mgr.RUN_FAILURE))
        restart = themis.Themis()
        execute_blocking(restart)
        self.validate_success(len(run_ids))

    @utils.clean_directory_decorator()
    def test_restarts_auto(self):
        """Test case in which the application needs to be restarted.

        The application is rigged to fail the first time it is executed.
        """
        # prepare the ensemble info
        run_ids = run_id_range(10)
        runs = [themis.Run({}, APPLICATIONS + "restarts.py") for _ in run_ids]
        mgr = themis.Themis.create(
            runs=runs, application="python", max_restarts=1, app_is_batch_script=False,
        )
        execute_blocking(mgr)
        self.validate_success(len(run_ids))

    @utils.clean_directory_decorator()
    def test_incomplete_ensemble(self):
        """Test that all active runs are marked as pending on shutdown."""
        run_ids = run_id_range(10)
        runs = [themis.Run({}, "500") for _ in run_ids]
        mgr = themis.Themis.create(
            runs=runs, application="sleep", app_is_batch_script=False
        )
        execute_blocking(mgr, timeout=0.15)
        self.assertEqual(mgr.count_by_status(mgr.RUN_QUEUED), len(run_ids))

    @utils.clean_directory_decorator()
    def test_run_function(self):
        """Test the resume utility function."""
        run_ids = run_id_range(10)
        runs = [themis.Run({}, APPLICATIONS + "restarts.py") for _ in run_ids]
        mgr = themis.Themis.create_resume(
            runs=runs, application="python", max_restarts=0, app_is_batch_script=False,
        )
        execute_blocking(mgr)
        failed_runs = mgr.filter_by_status(mgr.RUN_FAILURE)
        self.assertEqual(len(failed_runs), len(run_ids))
        mgr.restart_runs(failed_runs)
        restart = themis.Themis.create_resume(
            runs=runs, application="python", max_restarts=0
        )
        execute_blocking(restart)
        self.validate_success(len(run_ids))

    @utils.clean_directory_decorator()
    def test_new_function(self):
        run_ids = run_id_range(5)
        results = self.token_ensemble(run_ids)
        with self.assertRaises(Exception):
            self.token_ensemble(run_ids)
        mgr = themis.Themis.create_overwrite(
            runs=[themis.Run({}, "0") for _ in run_ids],
            application="sleep",
            max_restarts=0,
            app_is_batch_script=False,
        )
        execute_blocking(mgr)
        self.validate_success(len(run_ids))

    @utils.clean_directory_decorator()
    def test_manual_restart_adding_runs(self):
        """Test adding_runs through the Themis object."""
        run_ids = run_id_range(10)
        runs = [themis.Run({}, "0") for _ in run_ids]
        mgr = themis.Themis.create(
            runs=runs, application="sleep", max_restarts=0, app_is_batch_script=False,
        )
        execute_blocking(mgr)
        self.validate_success(len(run_ids))
        mgr.add_runs(runs)
        restart = themis.Themis()
        execute_blocking(restart)
        self.validate_success(2 * len(run_ids), restart)

    @utils.clean_directory_decorator()
    def test_max_failed_runs(self):
        """Test case in which all other applications fail, and max_failed_runs kicks in.

        Leverages the fact that the application ``sleep`` requires an argument,
        otherwise it fails.

        The number of runs is set high, and the save_interval set low, to make sure
        that max_failed_runs kills the ensemble. Otherwise, there is a risk that the
        ensemble would complete all its runs before it checks failed runs.
        """
        run_ids = run_id_range(400)
        runs = [themis.Run({}) for _ in run_ids]
        mgr = themis.Themis.create(
            runs=runs,
            application="false",
            max_restarts=0,
            max_failed_runs=10,
            app_is_batch_script=False,
        )
        mgr._spec["save_interval"] = 1
        database.write_app_spec(mgr._spec, mgr._spec["setup_dir"])
        execute_blocking(mgr)
        self.assertGreaterEqual(mgr.count_by_status(mgr.RUN_FAILURE), 10)
        self.assertLess(mgr.count_by_status(mgr.RUN_FAILURE), len(run_ids))
        self.assertEqual(mgr.count_by_status(mgr.RUN_SUCCESS, mgr.RUN_KILLED), 0)

    @utils.clean_directory_decorator()
    def test_adding_runs_in_post_ensemble(self):
        """Test case in which post_ensemble adds new runs."""
        original_run_ids = run_id_range(25)
        run_ids_added_by_post_ensemble = run_id_range2(25, 50)
        all_run_ids = list(original_run_ids) + list(run_ids_added_by_post_ensemble)
        runs = [themis.Run({}, "0") for _ in original_run_ids]
        mgr = themis.Themis.create(
            runs=runs,
            application="sleep",
            app_interface=INTERFACES + "adding_runs_interface.py",
            app_is_batch_script=False,
        )
        execute_blocking(mgr)
        self.validate_success(len(all_run_ids))

    @utils.clean_directory_decorator()
    def test_batch_script_application(self):
        """Test the simplest possible batch script."""
        run_ids = run_id_range(15)
        runs = [themis.Run({}) for _ in run_ids]
        mgr = themis.Themis.create(
            runs=runs,
            application=APPLICATIONS + "laptop_batch_script.sh",
            app_is_batch_script=True,
        )
        execute_blocking(mgr)
        self.validate_success(len(run_ids))

    @utils.clean_directory_decorator()
    def test_laf_laptop(self):
        """Test the BatchSubmitter class on the simplest possible example."""
        run_ids = run_id_range(5)
        mgr = laf.BatchSubmitter(
            batch_script=APPLICATIONS + "laptop_laf_script.sh",
            resource_mgr="none",
            samples=[{"integer": 0} for _ in run_ids],
        )
        job_ids = mgr.execute()
        self.assertEqual(len(job_ids), len(run_ids))

    @utils.clean_directory_decorator()
    def test_laf_dry_run(self):
        run_ids = run_id_range(5)
        run_dir_names = "integer-{integer}"
        samples = [{"integer": i} for i in run_ids]
        mgr = laf.BatchSubmitter(
            batch_script=APPLICATIONS + "laptop_laf_script.sh",
            resource_mgr="none",
            samples=samples,
            run_dir_names=run_dir_names,
            run_symlink=APPLICATIONS + "laptop_batch_script.sh",
        )
        for sample in samples:
            self.assertFalse(os.path.exists(run_dir_names.format(**sample)))
        mgr.dry_run()
        for sample in samples:
            run_dir = run_dir_names.format(**sample)
            self.assertTrue(os.path.isdir(run_dir))
            self.assertTrue(
                os.path.islink(os.path.join(run_dir, "laptop_batch_script.sh"))
            )

    @utils.clean_directory_decorator()
    def test_failing_application(self):
        """Test that a failing application is properly marked as a failure."""
        run_ids = run_id_range(30)
        runs = [themis.Run({}) for _ in run_ids]
        mgr = themis.Themis.create(
            runs=runs, application="sleep", max_restarts=0, app_is_batch_script=False,
        )
        execute_blocking(mgr, parallelism=4, max_concurrency=10)
        self.assertEqual(mgr.count_by_status(mgr.RUN_FAILURE), len(run_ids))
        self.assertEqual(mgr.progress(), (len(run_ids), len(run_ids)))

    @utils.clean_directory_decorator()
    def test_run_symlink(self):
        """Test that run_symlink are properly symlinked into the run directories."""
        run_ids = run_id_range(30)
        runs = [themis.Run({}, APPLICATIONS + "required_files.py") for _ in run_ids]
        mgr = themis.Themis.create(
            runs=runs,
            application="python",
            run_symlink=(REQDFILES + "required_di?", REQDFILES + "required_json.*",),
            app_is_batch_script=False,
        )
        execute_blocking(mgr, parallelism=3, max_concurrency=8)
        self.validate_success(len(run_ids))

    @utils.clean_directory_decorator()
    def test_run_symlink_broken_symlinks(self):
        """Test that removing breaking the run_symlink symlinks doesn't crash the
        ensemble.

        This test is of dubious usefulness. It isn't clear that the ensemble *shouldn't*
        abort.
        """
        files_to_create_then_delete = ("broken_sym1.txt", "broken_sym2.txt")
        for file_name in files_to_create_then_delete:
            with open(file_name, "w") as file_handle:
                file_handle.write("foo\nbar\n")
        run_ids = run_id_range(10)
        runs = [themis.Run({}, "0") for _ in run_ids]
        mgr = themis.Themis.create(
            runs=runs,
            application="sleep",
            run_symlink=files_to_create_then_delete,
            app_is_batch_script=False,
        )
        for file_name in files_to_create_then_delete:
            os.remove(file_name)
        execute_blocking(mgr, parallelism=2)
        self.validate_success(len(run_ids))

    @utils.clean_directory_decorator()
    def test_parsing_application(self):
        """Test that if the app is a batch script, it is parsed."""
        run_ids = run_id_range(5)
        runs = [themis.Run({"integer": 1}) for _ in run_ids]
        mgr = themis.Themis.create(
            application=APPLICATIONS + "laptop_laf_script.sh",
            runs=runs,
            app_is_batch_script=True,
        )
        execute_blocking(mgr)
        self.validate_success(len(run_ids))

    @utils.clean_directory_decorator()
    def test_clear_exists(self):
        """Test the themis.exists and themis.clear functions."""
        self.assertFalse(themis.Themis.exists())
        run_ids = run_id_range(3)
        results = self.token_ensemble(run_ids)
        self.assertTrue(themis.Themis.exists())
        themis.Themis.clear()
        self.assertFalse(themis.Themis.exists())

    @utils.clean_directory_decorator()
    def test_adding_results(self):
        run_ids = run_id_range(5)
        runs = [themis.Run({"sleep_sec": 0}, "0") for _ in run_ids]
        mgr = themis.Themis.create(
            runs=runs,
            application="sleep",
            app_interface=INTERFACES + "adding_tuple_results.py",
            app_is_batch_script=False,
        )
        force_noresourcemanager(mgr)
        execute_blocking(mgr)
        df = mgr.as_dataframe()
        self.assertTrue(all([sleep_sec == 0 for sleep_sec in df["sleep_sec"].values]))
        self.assertTrue(all([result == (1, 2, 3) for result in df["result"].values]))
        self.assertTrue(all([steps[0].tasks == 1 for steps in df["steps"].values]))
        with open("json_test.json", "w") as stream:
            mgr.write_json(stream)
        with open("json_test.json") as file_handle:
            json_results = json.load(file_handle)
        self.assertEqual(
            list(run_ids), sorted([int(mapping["run_id"]) for mapping in json_results])
        )
        for run in json_results:
            self.assertEqual(run["sample"]["sleep_sec"], 0)
            self.assertEqual(run["steps"][0]["tasks"], 1)
        mgr.set_result(1, "new_result")
        df = mgr.as_dataframe()
        self.assertEqual(df.loc[1].result, "new_result")

    @utils.clean_directory_decorator()
    def test_application_reqd_file(self):
        """Test that setting application to be a reqd_run_file doesn't raise an error.

        This could be an issue when application is a batch script, in which case
        it could be treated like both an input deck and a reqd file.
        """
        with warnings.catch_warnings(record=True) as w:
            run_ids = run_id_range(5)
            runs = [themis.Run({"integer": 0}, "0") for _ in run_ids]
            mgr = themis.Themis.create(
                runs=runs,
                application=APPLICATIONS + "laptop_laf_script.sh",
                run_symlink=APPLICATIONS + "laptop_laf_script.sh",
                app_is_batch_script=True,
            )
            for run_id in run_ids:
                self.assertEqual(None, mgr.dry_run(run_id))

    @utils.clean_directory_decorator()
    @unittest.skip("test test_themis_parse_command incorrectly calls Themis' parse command")
    def test_themis_parse_command(self):
        """Test calling `themis parse` from a batch script."""
        themis_path = sys.executable + " " + THEMIS_DIR
        runs = [
            themis.Run({"X": i, "Y": i + 1, "themis": themis_path})
            for i in run_id_range(3)
        ]
        mgr = themis.Themis.create(
            runs=runs,
            application=APPLICATIONS + "runtime_parse.sh",
            run_copy=INPUTDECKS + "runtime_parse_inputdeck.txt",
            app_is_batch_script=True,
        )
        force_noresourcemanager(mgr)
        execute_blocking(mgr)
        self.validate_success(len(runs))
        for run_id, directory in mgr.run_dirs():
            with open(
                os.path.join(directory, "runtime_parse_inputdeck.txt")
            ) as file_handle:
                lines = file_handle.readlines()
                self.assertEqual(int(lines[0]), run_id)
                self.assertEqual(int(lines[1]), run_id + 1)

    @utils.clean_directory_decorator()
    def test_themis_collect_command(self):
        """Test calling `themis parse` from a batch script."""
        themis_path = sys.executable + " " + THEMIS_DIR
        run_ids = run_id_range(3)
        runs = [
            themis.Run({"X": i, "Y": i + 17, "themis": themis_path}) for i in run_ids
        ]
        mgr = themis.Themis.create(
            runs=runs,
            application=APPLICATIONS + "runtime_collect.sh",
            run_parse=INPUTDECKS + "runtime_collect_script.py",
            app_is_batch_script=True,
        )
        force_noresourcemanager(mgr)
        execute_blocking(mgr)
        self.validate_success(len(runs))
        completed_runs = mgr.runs(run_ids)
       
        for run_id in run_ids:
            f = open("runs" + os.sep + str(run_id) + os.sep + "output.json")

            self.assertEqual(
                #json.loads(completed_runs[run_id].result),
                json.load(f),
                {"X": run_id, "Y": run_id + 17},
            )
            f.close()

    @utils.clean_directory_decorator()
    def test_multiple_steps_json(self):
        """Test an ensemble consisting of CompositeRuns with varying step counts."""
        run_ids = run_id_range(3)
        runs = [
            themis.CompositeRun(
                {"A": 1, "B": 2},
                [
                    themis.Step(
                        utils.which("true"), cores_per_task=cpt + 1, batch_script=False
                    )
                    for cpt in range(i)
                ],
            )
            for i in run_ids
        ]
        mgr = themis.Themis.create(runs=runs)
        force_noresourcemanager(mgr)
        execute_blocking(mgr)
        self.validate_success(len(runs))
        with open("json_test.json", "w") as stream:
            mgr.write_json(stream)
        with open("json_test.json") as file_handle:
            json_results = json.load(file_handle)
        for i, run in enumerate(json_results):
            self.assertEqual(len(run["steps"]), i + 1)
            for j, step in enumerate(run["steps"]):
                for attr in themis.Step.__slots__:
                    self.assertEqual(step[attr], getattr(runs[i].steps[j], attr))
        augmented_runs = mgr.runs(run_ids)
        for run_id in run_ids:
            self.assertEqual(augmented_runs[run_id].sample, runs[run_id - 1].sample)
            self.assertEqual(augmented_runs[run_id].status, themis.Themis.RUN_SUCCESS)
            self.assertIsNone(augmented_runs[run_id].result)
            for j, step in enumerate(augmented_runs[run_id].steps):
                for attr in themis.Step.__slots__:
                    self.assertEqual(
                        getattr(step, attr), getattr(runs[run_id - 1].steps[j], attr)
                    )

    @utils.clean_directory_decorator()
    def test_multiple_step_subdirectories(self):
        """Test an ensemble with varying-cwd steps"""
        run_ids = run_id_range(3)
        steps = [
            themis.Step(
                APPLICATIONS + "laptop_batch_script.sh", batch_script=True, cwd=str(i)
            )
            for i in range(3)
        ]
        runs = [themis.CompositeRun({"A": 1, "B": 2}, steps) for _ in run_ids]
        mgr = themis.Themis.create(runs=runs)
        force_noresourcemanager(mgr)
        execute_blocking(mgr)
        self.validate_success(len(runs))
        for run_id, run_dir in mgr.run_dirs():
            for i in range(3):
                self.assertTrue(
                    os.path.isfile(os.path.join(run_dir, str(i), "run.log"))
                )
                self.assertTrue(
                    os.path.isfile(os.path.join(run_dir, "laptop_batch_script.sh"))
                )

    @utils.clean_directory_decorator()
    def test_multiple_steps_absolute_subdirectories(self):
        """Test an ensemble with varying-cwd steps"""
        run_ids = run_id_range(3)

        def create_steps(run_id):
            return [
                themis.Step(
                    utils.which("true"),
                    batch_script=False,
                    cwd=os.path.abspath("run_" + str(run_id) + "_step_" + str(i)),
                )
                for i in range(3)
            ]

        runs = [themis.CompositeRun({"A": 1, "B": 2}, create_steps(i)) for i in run_ids]
        mgr = themis.Themis.create(runs=runs)
        force_noresourcemanager(mgr)
        execute_blocking(mgr)
        self.validate_success(len(runs))
        for run_id in run_ids:
            for step_id in range(3):
                step_dir = os.path.abspath(
                    "run_" + str(run_id) + "_step_" + str(step_id)
                )
                self.assertTrue(os.path.isfile(os.path.join(step_dir, "run.log")))

    @utils.timeout_decorator(10)
    @utils.clean_directory_decorator()
    def test_allow_multiple(self):
        """Test an ensemble with varying-cwd steps"""
        run_ids = run_id_range(50)
        runs = [
            themis.CompositeRun(
                {"A": 1, "B": 2},
                [
                    themis.Step(utils.which("true"), batch_script=False)
                    for _ in range(2)
                ],
            )
            for i in run_ids
        ]
        mgr = themis.Themis.create(runs=runs)
        force_noresourcemanager(mgr)
        for i in range(3):
            try:
                mgr.execute_local(allow_multiple=True, max_concurrency=2)
            except RuntimeError:  # may be raised if all runs already taken
                if i == 0:
                    raise  # but shouldn't have been raised if this was first execution
        while mgr.count_by_status(mgr.RUN_QUEUED):
            time.sleep(0.01)
        self.validate_success(len(runs))


if __name__ == "__main__":
    unittest.main()
