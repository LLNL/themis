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
from themis.backend import network
from themis.versions import http_client
from themis.backend import RunFeeder


ShamRunInfo = namedtuple("ShamRunInfo", ["run_id", "sample", "resources"])


class ShamDatabase:

    run = ShamRunInfo(13, {}, None)

    def __getattr__(self, attr):
        def do_nothing(*args, **kwargs):
            pass

        return do_nothing

    def new_runs(self, limit=0):
        return [self.run for _ in range(limit)]

    def runs_to_kill(self):
        return []


def get_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    return sock


class BaseServerTests(unittest.TestCase):
    @classmethod
    def create_cert(cls):
        if os.path.exists(CERT_NAME):
            os.remove(CERT_NAME)
        return "there_is_no_cert"

    @classmethod
    def setUpClass(cls):
        cert_path = cls.create_cert()
        cls.server = network.BaseServer(network.BaseRequestHandler, cert_path)
        cls.client = network.BaseClient(cert_path, *cls.server.server_address)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(ShamDatabase())

    def test_alive(self):
        self.assertTrue(self.server.is_alive())

    def test_bad_topic(self):
        for topic in ("not_a_topic", "another bad topic"):
            with self.assertRaises(IOError):
                self.client._get_response("GET", topic)

    @unittest.skipIf(
        sys.version_info.major < 3, "Library behaves differently in Py2 with bad method"
    )
    def test_bad_method(self):
        for method in ("not_a_method", "another bad method"):
            with self.assertRaises(IOError):
                self.client._get_response(method, "ping")

    def test_connection_type(self):
        self.assertTrue(
            isinstance(self.client._make_connection(), http_client.HTTPConnection)
        )
        self.assertTrue(isinstance(self.server.server.socket, socket.socket))

    def test_ping(self):
        self.client.ping(timeout=0.05)
        sock = get_socket()
        sock.listen(0)
        with contextlib.closing(sock):
            bad_port = sock.getsockname()[1]  # the socket isn't accepting connections
            client = network.BackendClient("notafile", "localhost", bad_port)
            with self.assertRaises((socket.timeout, ssl.SSLError)):
                client.ping(timeout=0.05)


class SSLBaseServerTests(BaseServerTests):
    """Same as Http tests, but create and use an SSL cert"""

    @classmethod
    def create_cert(cls):
        create_ssl_certificate(os.getcwd(), CERT_NAME)
        return CERT_NAME

    def test_connection_type(self):
        self.assertTrue(
            isinstance(self.client._make_connection(), http_client.HTTPSConnection)
        )
        self.assertTrue(isinstance(self.server.server.socket, ssl.SSLSocket))


class BackendServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        create_ssl_certificate(os.getcwd(), CERT_NAME)
        cls.db = ShamDatabase()
        cls.server = network.BackendServer(CERT_NAME, RunFeeder(cls.db, 1000))
        cls.client = network.BackendClient(CERT_NAME, *cls.server.server_address)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(ShamDatabase())

    def test_enable_runs(self):
        self.assertEqual(len(self.server.enabled_runs), 0)
        for run_id_set in (range(5), range(5, 10)):
            self.client.enable_runs(run_id_set)
            for run_id in run_id_set:
                self.assertIn(run_id, self.server.enabled_runs)
        self.server.enabled_runs.clear()

    def test_runs_to_kill(self):
        for runs_to_kill in (range(8), range(17, 25)):
            self.server.runs_to_kill.clear()
            self.assertEqual(len(self.server.runs_to_kill), 0)
            self.server.runs_to_kill.extend(runs_to_kill)
            self.assertEqual(list(runs_to_kill), list(self.client.runs_to_kill()))

    def test_add_result(self):
        for result_set in ([None, "my_result", (1, 2, 3), 7.8],):
            self.client.add_result(range(len(result_set)), result_set)
            for pair in enumerate(result_set):
                self.assertIn(pair, self.server.run_results)

    def set_run_status(self):
        for status_set in ([1, (1, 2), "status_foo"],):
            self.client.set_run_status(len(status_set), status_set)
            for pair in enumerate(status_set):
                self.assertIn(pair, self.server.run_statuses)

    def test_new_runs(self):
        for new_run_count in range(5, 26, 5):
            new_runs = self.client.new_runs(new_run_count)
            self.assertTrue(len(new_runs) > 0)
            self.assertTrue(len(new_runs) <= new_run_count)
            self.assertEqual(list(new_runs), [self.db.run for _ in new_runs])


if __name__ == "__main__":
    unittest.main()
