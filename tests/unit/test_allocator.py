from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import sys
import os
import shutil
import subprocess as sp

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

from themis import utils
from themis.resource import allocators


class ShellScriptTests(unittest.TestCase):
    def test_sleep(self):
        dir_to_create = "allocator_unit_tests"
        if os.path.exists(dir_to_create):
            shutil.rmtree(dir_to_create)
        target_path = os.path.abspath("allocator_unit_tests.sh")
        script = allocators.ShellScript(target_path)
        script.commands.extend(
            ["set -e", "mkdir -p allocator_unit_tests", "sleep 0.1 &"]
        )
        script.working_directory = os.getcwd()
        script.headers.append("#blah")
        script_path = script.write()
        self.assertEqual(script_path, target_path)
        self.assertTrue(os.path.isfile(script_path))
        # check that it's executable
        self.assertEqual(utils.which(script_path), script_path)
        proc = sp.Popen(script_path.split())
        proc.wait()
        self.assertEqual(proc.returncode, 0)
        self.assertTrue(os.path.exists(dir_to_create))


class AllocationTests(unittest.TestCase):
    def test_bad_input(self):
        resources = {"tasks": 1, "cores_per_task": 1, "gpus_per_task": 1}
        with self.assertRaises(ValueError):
            allocators.Allocation(-1, None, "", "", 5, resources)
        with self.assertRaises(ValueError):
            allocators.Allocation(-1, None, "", "", -9, resources)
        with self.assertRaises(TypeError):
            allocators.Allocation(-1, None, 687, "", 5, resources)
        with self.assertRaises(TypeError):
            allocators.Allocation(-1, None, "", 687, 5, resources)


class AllocatorTests(unittest.TestCase):
    """Tests for the ensemble.allocators.Allocator class"""

    @classmethod
    def setUpClass(cls):
        cls.allocation = allocators.Allocation(
            1, None, "", "", 5, {"tasks": 1, "cores_per_task": 1, "gpus_per_task": 1}
        )

    def test_sleep(self):
        allocator = allocators.InteractiveAllocator()
        allocator.start(self.allocation, ["sleep 0.01 &", "sleep 0.01 &"])
        self.assertFalse(allocator.done())
        allocator.wait()
        self.assertTrue(allocator.done())


if __name__ == "__main__":
    unittest.main()
