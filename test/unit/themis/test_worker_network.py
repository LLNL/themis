"""
Unit tests for worker/network.py
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import sys
import ssl
import socket
import contextlib
import os
from collections import namedtuple

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

from themis.backend.network import CERT_NAME, create_ssl_certificate
from themis.backend.worker import network
from themis.versions import http_client

ShamRunInfo = namedtuple("ShamRunInfo", ["run_id", "sample", "resources"])


def do_nothing(*args, **kwargs):
    pass


ShamDatabase = namedtuple("ShamDatabase", ["set_run_status", "add_result"])


class WorkerServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        create_ssl_certificate(os.getcwd(), CERT_NAME)
        cls.app_spec = {
            "run_parse": ["one", "two", "three"],
            "reqd_files": ["a", "b"],
            "application": "/usr/bin/bash",
            "max_restarts": 5,
        }
        cls.server = network.WorkerServer(CERT_NAME, cls.app_spec)
        cls.num_samples = 20
        cls.resources = [{"cores": i} for i in range(cls.num_samples)]
        cls.samples = [
            {"sample_1": i, "sample_2": 2 * i} for i in range(cls.num_samples)
        ]
        cls.run_infos = [
            ShamRunInfo(i, cls.samples[i], cls.resources[i])
            for i in range(cls.num_samples)
        ]
        cls.server.submit(cls.run_infos)
        cls.client = network.WorkerClient(CERT_NAME, *cls.server.server_address)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(ShamDatabase(do_nothing, do_nothing))

    def test_get_worker_setup(self):
        for run_id in range(self.num_samples):
            app_spec, run_info = self.client.get_worker_setup(run_id)
            self.assertEqual(app_spec, self.app_spec)
            self.assertEqual(run_info.sample, self.samples[run_id])
            self.assertEqual(run_info.resources, self.resources[run_id])

    def test_send_results(self):
        for run_id in range(self.num_samples):
            result = list(range(20))
            self.client.add_result(run_id, result)
            self.assertTrue((run_id, result) in self.server.results)

    def test_set_active_runs(self):
        for run_id_to_remove in range(0, len(self.run_infos)):
            self.assertTrue(run_id_to_remove in self.server.run_infos.keys())
            self.server.remove_completed_runs([run_id_to_remove])
            self.assertFalse(run_id_to_remove in self.server.run_infos.keys())
        # reset server back to original state
        self.server.submit(self.run_infos)


if __name__ == "__main__":
    unittest.main()
