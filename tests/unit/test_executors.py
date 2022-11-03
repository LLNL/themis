"""Tests for executors.py"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import sys
import os
import time
import signal

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

from themis import resource
from themis.resource import executors
from themis.utils import Step, which


def _sleep(executor, sleep_time=0):
    return executor.submit(0, Step([which("sleep"), str(sleep_time)]))


def wait_for_job(executor):
    result = executor.wait()
    while result is None:
        result = executor.wait()
    return result


class ProcessExecutorTests(unittest.TestCase):
    """Test the ProcessExecutor class--the only Launcher always available."""

    @classmethod
    def setUpClass(cls):
        cls.executor_factory = lambda _: resource.NoResourceManager.executor(
            "foo", ("foo", "bar")
        )

    def test_success(self):
        """Test submitting an application that succeeds."""
        executor = self.executor_factory()
        jobid = executor.submit(0, Step([which("true")]), )
        completed_job_id, returncode = wait_for_job(executor)
        self.assertEqual(completed_job_id, jobid)
        self.assertEqual(returncode, 0)

    def test_kill(self):
        """Test killing an application."""
        executor = self.executor_factory()
        jobid = _sleep(executor, 2)
        executor.kill(jobid)
        completed_job_id, returncode = wait_for_job(executor)
        self.assertEqual(completed_job_id, jobid)
        self.assertEqual(returncode, -signal.SIGTERM)

    def test_failure(self):
        """Test submitting an application that fails."""
        executor = self.executor_factory()
        jobid = executor.submit(0, Step([which("false")]), )
        completed_job_id, returncode = wait_for_job(executor)
        self.assertEqual(completed_job_id, jobid)
        self.assertEqual(returncode, 1)

    def test_multiple_submission(self):
        """Test submitting multiple jobs."""
        num_jobs = 5
        executor = self.executor_factory()
        jobids = [_sleep(executor, 0) for _ in range(num_jobs)]
        waited_jobids = []
        for _ in range(num_jobs):
            completed_job_id, returncode = wait_for_job(executor)
            self.assertEqual(returncode, 0)
            waited_jobids.append(completed_job_id)
        self.assertEqual(sorted(jobids), sorted(waited_jobids))


class FluxBindingsLauncherTests(ProcessExecutorTests):
    """Test the FluxBindingsExecutor class, if it is available."""

    @classmethod
    def setUpClass(cls):
        cls.executor_factory = lambda _: resource.Flux.executor("foo", ("foo", "bar"))
        if not executors.FLUX_BINDINGS_AVAIL:
            raise unittest.SkipTest("Flux bindings unavailable")

    def test_kill(self):
        """Test killing an application."""
        from concurrent.futures import CancelledError

        executor = self.executor_factory()
        jobid = _sleep(executor, 2)
        executor.kill(jobid)
        completed_job_id, returncode = wait_for_job(executor)
        # either the future was cancelled or the `flux.job.cancel` function will
        # set an exception on the future
        self.assertIsInstance(returncode, Exception)


if __name__ == "__main__":
    unittest.main()
