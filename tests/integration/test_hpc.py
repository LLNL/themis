from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import os
import sys
import shutil
from decimal import Decimal

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

import themis
from themis import database
from themis import resource
from themis.resource.allocators import Allocation
from themis import utils
from themis import laf

# global constants
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APPLICATIONS = os.path.join(THIS_DIR, "Applications")
INTERFACES = os.path.join(THIS_DIR, "Interfaces")


class ThemisHPCIntegrationTests(unittest.TestCase):
    """Test cases for the ensemble manager on less simple examples"""

    @classmethod
    def setUpClass(cls):
        rmgr = resource.identify_resource_manager()
        if isinstance(rmgr, resource.NoResourceManager):
            raise unittest.SkipTest("These tests require an HPC machine")
        if isinstance(rmgr, resource.LANLSlurm):
            cls.testing_partition = "pdebug"
        else:
            cls.testing_partition = "pdebug"
        cls.testing_dir = utils.CleanDirectory(cls.__name__)
        cls.testing_dir.go_to_new()

    @classmethod
    def tearDownClass(cls):
        cls.testing_dir.go_to_old()

    @utils.clean_directory_decorator()
    def test_mpi_app_native(self):
        self._test_mpi_app(False, None)

    @utils.clean_directory_decorator()
    @unittest.skipUnless(resource.Flux.available(), "test requires Flux")
    def test_mpi_app_flux(self):
        self._test_mpi_app(True, None)

    @utils.clean_directory_decorator()
    def test_mpi_app_batch_script_native(self):
        self._test_mpi_app(False, os.path.join(APPLICATIONS, "mpi_app_batch_script.sh"))

    @utils.clean_directory_decorator()
    @unittest.skipUnless(resource.Flux.available(), "test requires Flux")
    def test_mpi_app_batch_script_flux(self):
        self._test_mpi_app(True, os.path.join(APPLICATIONS, "mpi_app_batch_script.sh"), timeout=20)

    def _test_mpi_app(self, use_flux, batch_script, timeout=10):
        """Test an application whose output depends on the number of ranks."""
        run_ids = range(1, 16)
        tasks_list = [(i // 5) + 1 for i in run_ids]
        # create the samples
        runs = [
            themis.Run(
                args=os.path.join(APPLICATIONS, "mpi_app.py") + " " + str(run_id),
                tasks=tasks,
                sample={"python": sys.executable},
            )
            for run_id, tasks in zip(run_ids, tasks_list)
        ]
        # prepare the ensemble info
        mgr = themis.Themis.create(
            application=batch_script if batch_script is not None else sys.executable,
            runs=runs,
            app_interface=os.path.join(INTERFACES, "mpi_app_interface.py"),
            app_is_batch_script=True if batch_script is not None else False,
            use_flux=use_flux,
        )
        alloc = Allocation(nodes=1, partition=self.testing_partition)
        mgr._execute(alloc, blocking=True)
        run_results = mgr.runs(run_ids)
        # now test the results
        for run_id, tasks in zip(run_ids, tasks_list):
            self.assertEqual(
                (tasks * run_id, int(tasks * run_id)), run_results[run_id].result,
            )

    @utils.clean_directory_decorator()
    def test_batch_submitter(self):
        """Test the BatchSubmitter class on a proper batch script."""
        rmgr = resource.identify_resource_manager()
        if isinstance(rmgr, resource.Slurm):
            batch_script = os.path.join(APPLICATIONS, "slurm_laf_batch_script.sh")
        elif isinstance(rmgr, resource.Lsf):
            batch_script = os.path.join(APPLICATIONS, "lsf_laf_batch_script.sh")
        else:
            raise ValueError("Unrecognized resource manager")
        run_ids = range(2)
        mgr = laf.BatchSubmitter(
            batch_script=batch_script,
            resource_mgr=rmgr.identifier,
            samples=[
                {"sleep_time": 0, "partition": self.testing_partition} for _ in run_ids
            ],
        )
        job_ids = mgr.execute()
        self.assertEqual(len(job_ids), len(run_ids))


if __name__ == "__main__":
    unittest.main()
