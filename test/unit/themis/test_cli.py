"""
Unit tests for ensemble/__main__.py
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import argparse
import sys
import os
import pickle
import collections

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

from themis import __main__ as cli
from themis.backend.worker.network import WorkerServer
from themis.backend import managed_shutdown
import themis.runtime


SUPPORTING_FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support")

ShamRunInfo = collections.namedtuple("ShamRunInfo", ("run_id", "run"))
ShamRun = collections.namedtuple("ShamRun", ("run_id", "sample"))
ShamDatabase = collections.namedtuple("ShamDatabase", ["set_run_status", "add_result"])


def do_nothing(*args, **kwargs):
    pass


def error_handler(self, message):
    raise SystemExit(message)


def shutdown(server):
    return managed_shutdown(server, ShamDatabase(do_nothing, do_nothing))


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_err = argparse.ArgumentParser.error
        argparse.ArgumentParser.error = error_handler
        cls.parser = cli.setup_parsers()

    @classmethod
    def tearDownClass(cls):
        argparse.ArgumentParser.error = cls._orig_err

    def test_status(self):
        for status in ("successful", "failed", "queued", "killed"):
            for count in ("-c ", "--count ", ""):
                args = self.parser.parse_args(("status " + count + status).split())
                self.assertEqual(args.status_names, [status])
                self.assertEqual(args.count, bool(count))
                self.assertEqual(args.handler, cli.handle_status_subcommand)
        for count in ("-c ", "--count ", ""):
            args = self.parser.parse_args(
                ("status successful failed queued killed " + count).split()
            )
            self.assertEqual(
                args.status_names, ["successful", "failed", "queued", "killed"]
            )
            self.assertEqual(args.count, bool(count))
            self.assertEqual(args.handler, cli.handle_status_subcommand)

    def test_kill(self):
        args = self.parser.parse_args("kill 1 2 3".split())
        self.assertEqual(args.run_ids, [1, 2, 3])
        self.assertEqual(args.handler, cli.handle_kill_subcommand)
        with self.assertRaises(SystemExit):
            self.parser.parse_args("kill".split())

    def test_restart(self):
        args = self.parser.parse_args("restart 1 2 3".split())
        self.assertEqual(args.run_ids, [1, 2, 3])
        self.assertEqual(args.handler, cli.handle_requeue_subcommand)
        with self.assertRaises(SystemExit):
            self.parser.parse_args("restart".split())

    def test_display(self):
        args = self.parser.parse_args("display 0 5".split())
        self.assertEqual(args.lowerbound, 0)
        self.assertEqual(args.upperbound, 5)
        self.assertEqual(args.handler, cli.handle_display_subcommand)
        with self.assertRaises(SystemExit):
            self.parser.parse_args("display".split())

    def test_progress(self):
        args = self.parser.parse_args("progress".split())
        self.assertEqual(args.handler, cli.handle_progress_subcommand)

    def test_runtime_parse(self):
        with shutdown(WorkerServer("not_a_cert", {})) as server:
            sample = {"viscocity": 45.8, "hydrostatics": 1740}
            run_id = 17
            runs = [ShamRunInfo(run_id, ShamRun(run_id, sample))]
            server.submit(runs)
            expected_file = "test_runtime_parse.txt"
            args = self.parser.parse_args(
                "runtime parse {} {}".format(
                    os.path.join(SUPPORTING_FILES, "docs_input_deck.txt"), expected_file
                ).split()
            )
            os.environ[themis.runtime.RUNID_ENV_VAR] = str(run_id)
            os.environ[themis.runtime.URI_ENV_VAR] = themis.runtime.URI_SPLITCHAR.join(
                server.server_address
            )
            os.environ[themis.runtime.SETUPDIR_ENV_VAR] = "foobar"
            os.environ[themis.runtime.EXECDIR_ENV_VAR] = "foobar"
            args.handler(args)
            with open(expected_file) as file_handle:
                received_lines = file_handle.readlines()
            with open(
                os.path.join(SUPPORTING_FILES, "docs_input_deck_reference.txt")
            ) as file_handle:
                expected_lines = file_handle.readlines()
            self.assertEqual(received_lines, expected_lines)

    def test_runtime_collect(self):
        with shutdown(WorkerServer("not_a_cert", {})) as server:
            self.assertFalse(server.results)
            run_id = 17
            result = {"foobar": 24898, "bar": "something"}  # junk
            collect_file = "test_runtime_collect.pkl"
            with open(collect_file, "wb") as file_handle:
                pickle.dump(result, file_handle)
            args = self.parser.parse_args(["runtime", "collect", collect_file])
            os.environ[themis.runtime.RUNID_ENV_VAR] = str(run_id)
            os.environ[themis.runtime.URI_ENV_VAR] = themis.runtime.URI_SPLITCHAR.join(
                server.server_address
            )
            os.environ[themis.runtime.SETUPDIR_ENV_VAR] = "foobar"
            os.environ[themis.runtime.EXECDIR_ENV_VAR] = "foobar"
            args.handler(args)
            received_runid, received_result = server.results.popleft()
            self.assertEqual(received_runid, run_id)
            self.assertEqual(result, pickle.loads(received_result))


if __name__ == "__main__":
    unittest.main()
