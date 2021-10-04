"""
Unit tests for the runtime module
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import pickle
import unittest
import collections

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

from themis import runtime
from themis.backend import managed_shutdown
from themis.backend.worker.network import WorkerServer
from themis.runtime import (
    URI_ENV_VAR,
    SETUPDIR_ENV_VAR,
    RUNID_ENV_VAR,
    URI_SPLITCHAR,
    EXECDIR_ENV_VAR,
)


ShamRunInfo = collections.namedtuple("ShamRunInfo", ("run_id", "run"))
ShamRun = collections.namedtuple("ShamRun", ("run_id", "sample"))
ShamDatabase = collections.namedtuple("ShamDatabase", ["set_run_status", "add_result"])


def do_nothing(*args, **kwargs):
    pass


def shutdown(server):
    return managed_shutdown(server, ShamDatabase(do_nothing, do_nothing))


def _clear_env():
    for env_var in (URI_ENV_VAR, SETUPDIR_ENV_VAR, RUNID_ENV_VAR, EXECDIR_ENV_VAR):
        if env_var in os.environ:
            del os.environ[env_var]


class RuntimeTests(unittest.TestCase):
    """Tests for themis.runtime module"""

    def setUp(self):
        _clear_env()

    def tearDown(self):
        _clear_env()

    def test_environment(self):
        for func in (
            runtime.fetch_run,
            runtime.fetch_run_id,
            runtime.themis_handle,
            lambda: runtime.set_result(None),
            lambda: runtime.parse_file(None, None),
        ):
            with self.assertRaisesRegexp(RuntimeError, runtime._RUNTIME_ERRMSG):
                func()

    def test_run_id(self):
        for run_id in range(5):
            os.environ[RUNID_ENV_VAR] = str(run_id)
            self.assertEqual(run_id, runtime.fetch_run_id())

    def test_run(self):
        with shutdown(WorkerServer("not_a_cert", {})) as server:
            sample = {"viscocity": 45.8, "hydrostatics": 1740}
            run_id = 17
            sham_run = ShamRun(run_id, sample)
            runs = [ShamRunInfo(run_id, sham_run)]
            server.submit(runs)
            expected_file = "test_runtime_parse.txt"
            os.environ[RUNID_ENV_VAR] = str(run_id)
            os.environ[URI_ENV_VAR] = URI_SPLITCHAR.join(server.server_address)
            os.environ[SETUPDIR_ENV_VAR] = "foobar"
            os.environ[EXECDIR_ENV_VAR] = "foobar"
            self.assertEqual(runtime.fetch_run(), sham_run)

    def test_set_result(self):
        with shutdown(WorkerServer("not_a_cert", {})) as server:
            self.assertFalse(server.results)
            run_id = 17
            result = {"foobar": 24898, "bar": "something"}  # junk
            os.environ[RUNID_ENV_VAR] = str(run_id)
            os.environ[URI_ENV_VAR] = URI_SPLITCHAR.join(server.server_address)
            os.environ[SETUPDIR_ENV_VAR] = "foobar"
            os.environ[EXECDIR_ENV_VAR] = "foobar"
            runtime.set_result(result)
            received_runid, received_result = server.results.popleft()
            self.assertEqual(received_runid, run_id)
            self.assertEqual(result, received_result)


if __name__ == "__main__":
    unittest.main()
