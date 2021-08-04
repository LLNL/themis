"""
Unit tests for worker/schedulers.py
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import sys
import os
import threading
import time
import logging
import traceback
import collections

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

from themis.backend.worker import schedulers
from themis.database.utils import EnsembleDatabase


ShamRunInfo = collections.namedtuple(
    "ShamRunInfo", ["run_id", "status", "restarts", "sample"]
)


def do_nothing(*args, **kwargs):
    pass


ShamDatabase = collections.namedtuple(
    "ShamDatabase", ["set_run_status", "enable_runs", "RUN_FAILURE"]
)


class ShamExecutor(object):

    returncode = 0

    def __init__(self):
        self.next_job_id = 0
        self.submitted_job_ids = collections.deque()

    def submit(self, *args, **kwargs):
        self.next_job_id += 1
        self.submitted_job_ids.append(self.next_job_id - 1)
        return self.next_job_id - 1

    def wait(self, timeout=None):
        try:
            return (self.submitted_job_ids.popleft(), self.returncode)
        except IndexError:
            return None


class AppRunTests(unittest.TestCase):
    def test_success(self):
        steps = [((), "") for i in range(10)]
        run = schedulers._AppRun(0, 0, 0, steps, (), ShamExecutor())
        for i in range(len(steps) - 1):
            self.assertEqual(run.state, i)
            self.assertFalse(run.run_complete(0))
        self.assertTrue(run.run_complete(0))
        self.assertEqual(run.completion_state, EnsembleDatabase.RUN_SUCCESS)

    def test_failure(self):
        tasks = 2
        steps = [((), "")]
        run = schedulers._AppRun(0, 0, 0, steps, (), ShamExecutor())
        self.assertTrue(run.run_complete(1))
        self.assertEqual(run.state, 0)
        self.assertEqual(run.completion_state, EnsembleDatabase.RUN_FAILURE)


class SchedulerTests(unittest.TestCase):
    @classmethod
    def step_creator(cls, num_steps):
        return [((), "") for _ in range(num_steps)]

    @classmethod
    def run_creator(cls, run_info, executor):
        return schedulers._AppRun(
            run_info.run_id,
            run_info.status,
            run_info.restarts,
            cls.step_creator(5),
            (),
            executor,
        )

    def setUp(self):
        ShamExecutor.returncode = 0

    def test_scheduler_completion(self):
        scheduler = schedulers.JobSubmissionThread(
            20, self.run_creator, ShamExecutor(), 10
        )
        scheduler.start()
        for _ in range(3):
            run_infos = [ShamRunInfo(i, 0, 0, {}) for i in range(10)]
            scheduler.submit(run_infos)
            while scheduler.is_alive() and not scheduler.done():
                time.sleep(0.1)
            self.assertTrue(scheduler.is_alive() and scheduler.done())
            self.assertEqual(scheduler.count_pending_runs(), 0)
        scheduler.shutdown(ShamDatabase(do_nothing, do_nothing, 0))


if __name__ == "__main__":
    unittest.main()
