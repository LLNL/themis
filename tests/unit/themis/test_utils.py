"""Tests for ensemble/utils.py"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import os
import sys
import glob

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

from themis.utils import Run, CompositeRun, Step
from themis import utils


SUPPORTING_FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support")


class RunAndStepTests(unittest.TestCase):
    """Unit tests for the Run and Step classes"""

    def test_run_invalid_constructor_types(self):
        """Test that constructor arguments must be of the right type"""
        with self.assertRaises((TypeError, AttributeError)):
            Run(args=sys.executable, timeout=()).steps
        with self.assertRaises((TypeError, AttributeError)):
            Run(args=sys.executable, cores_per_task=[5, 6], timeout=10).steps
        with self.assertRaises((TypeError, AttributeError)):
            Run(args=sys.executable, sample=5).steps
        with self.assertRaises((TypeError, AttributeError)):
            Run(args=sys.executable, timeout="-foo").steps

    def test_step_invalid_constructor_types(self):
        with self.assertRaises(TypeError):
            Step(sys.executable, timeout=())
        with self.assertRaises(TypeError):
            Step(sys.executable, cores_per_task=[5, 6], timeout=10)
        with self.assertRaises(TypeError):
            Step(sys.executable, sample=5)
        with self.assertRaises(ValueError):
            Step(sys.executable, timeout="-foo")

    def test_composite_run(self):
        steps = [Step(sys.executable)]
        run = CompositeRun(None, steps)
        self.assertEqual(run.sample, {})
        self.assertEqual(run.steps, steps)
        with self.assertRaises(TypeError):
            CompositeRun(None, None)
        with self.assertRaises(ValueError):
            CompositeRun(None, [])


class UtilityFunctionTests(unittest.TestCase):
    """Unit tests for functions in ensemble/utils.py"""

    def test_range_check(self):
        with self.assertRaises(ValueError):
            utils.range_check(5, min_val=6)
        with self.assertRaises(ValueError):
            utils.range_check(6, max_val=5)

    def test_type_check(self):
        with self.assertRaises(TypeError):
            utils.type_check(5.5, int, str)
        with self.assertRaises(TypeError):
            utils.type_check("blah", float, int)
        self.assertEqual(5, utils.type_check(5, int))

    def test_import_app_interface(self):
        interface_dir = os.path.join(
            os.path.abspath(__file__).rsplit(os.sep, 2)[0], "integration", "Interfaces"
        )
        for app_interface in glob.glob(os.path.join(interface_dir, "*.py")):
            ai = utils.import_app_interface(app_interface)
            self.assertTrue(isinstance(ai, type(unittest)))  # check that ai is a module

    def test_convert_none(self):
        for value in (1, 2, "blah", 7.8, []):
            self.assertEqual(value, utils.convert_none(None, value))
            self.assertEqual(value, utils.convert_none(value, -1))

    def test_validate_application(self):
        self.assertEqual(sys.executable, utils.validate_application(sys.executable))
        with self.assertRaises(ValueError):
            utils.validate_application(os.path.join("not", "a", "real", "app"))

    def test_which(self):
        self.assertTrue(os.path.isfile(utils.which("python")))
        self.assertEqual(sys.executable, utils.which(sys.executable))

    def test_validate_samples(self):
        for sample_set in (
            [{"label": val} for val in range(4)],
            [{"density": val, "visc": val} for val in range(4)],
        ):
            utils.validate_samples(sample_set)
            with self.assertRaises(ValueError):
                utils.validate_samples(sample_set, "{bad_run_dir_names}")
        sample_set = [{"overlapping_vals": 1} for _ in range(3)]
        with self.assertRaises(ValueError):
            utils.validate_samples(sample_set, "{overlapping_vals}")
        with self.assertRaises(TypeError):
            utils.validate_samples(5)
        with self.assertRaises(ValueError):
            utils.validate_samples([], check_nonempty=True)

    def test_directory_manager(self):
        original_cwd = os.getcwd()
        test_dir = os.path.join(original_cwd, "directory_manager_test")
        os.mkdir(test_dir)
        dir_mgr = utils.DirectoryManager(test_dir)
        with dir_mgr:
            self.assertEqual(os.getcwd(), test_dir)
        self.assertEqual(os.getcwd(), original_cwd)
        try:
            with dir_mgr:
                self.assertEqual(os.getcwd(), test_dir)
                raise ValueError("Test!")
        except ValueError:
            pass
        self.assertEqual(os.getcwd(), original_cwd)
        os.rmdir(test_dir)

    def test_sequence_of_path_patterns(self):
        for file_sequence in (None, ()):
            self.assertEqual((), utils.sequence_of_path_patterns(file_sequence))
        self.assertEqual(
            (sys.executable,), utils.sequence_of_path_patterns(sys.executable)
        )

    def test_get_run_dir(self):
        for sample_val, sample in [
            (i, {"run_dir_test_one": i, "run_dir_test_two": 2 * i})
            for i in range(-5, 5)
        ]:
            run_dir = utils.get_run_dir(
                0,
                os.path.join(os.getcwd(), "{run_dir_test_one}", "{run_dir_test_two}"),
                sample,
            )
            self.assertEqual(
                run_dir, os.path.join(os.getcwd(), str(sample_val), str(2 * sample_val))
            )

    def test_read_csv(self):
        rows = list(utils.read_csv(os.path.join(SUPPORTING_FILES, "csv_example.csv")))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["message"], "hello world")
        self.assertEqual(rows[2]["language"], " french ")
        self.assertEqual(rows[3]["message"], " Buongiorno   MONDO   ")


if __name__ == "__main__":
    unittest.main()
