"""Classes and functions for HTTP networking across Themis' backend.

Includes base classes, and derived classes for networking between
the backend and workers.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import ssl
import collections
import traceback
import platform
import subprocess as sp
import contextlib
import threading
import socket
import logging

from themis.database.utils import EnsembleDatabase
from themis import backend
from themis.versions import pickle, http_server, http_client, urlencode, parse_qs


CERT_NAME = "ssl_certificate.pem"
_DEFAULT_TIMEOUT = socket.getdefaulttimeout()


def create_ssl_certificate(cert_dir=os.curdir, cert_name=CERT_NAME):
    """Create an SSL certificate"""
    cert_path = os.path.join(cert_dir, cert_name)
    if not os.path.exists(cert_path):
        command = (
            (
                "openssl req -new -x509 -nodes -out {cert} -keyout {cert} "
                "-subj /C=US/ST=CA/L=Livermore/O=LLNL/OU=LLNL/CN=llnl.gov"
            )
            .format(cert=cert_path)
            .split()
        )
        subproc = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT)
        stdout, _ = subproc.communicate()
        if subproc.returncode != 0:
            raise OSError(
                "SSL certificate couldn't be generated; gave error:\n{}".format(stdout)
            )
    return os.path.abspath(cert_path)


def copy_and_empty(deq):
    """Empty a collections.deque into a list and return the list."""
    while deq:
        yield deq.popleft()


def copy_and_empty2(deq):
    """Empty a deque of 2-tuples into two lists and return them."""
    copied1 = []
    copied2 = []
    while deq:
        first, second = deq.popleft()
        copied1.append(first)
        copied2.append(second)
    return (copied1, copied2)


class BaseRequestHandler(http_server.BaseHTTPRequestHandler, object):
    """Handles requests to the server. Each request instantiates an instance."""

    def log_message(self, format_str, *args):  # pylint: disable=arguments-differ
        """Log a message."""
        logging.getLogger(__name__).info(format_str, *args)

    def get_content(self):
        """Get the content from a POST request.

        Raises ValueError if Content-Length header is improper
        """
        if "Content-Length" not in self.headers:
            return None
        length = int(self.headers["Content-Length"])
        return self.rfile.read(length)

    def call_handler(self):
        """Call a handler method based on the `topic` value in the query string."""
        parsed_request = parse_qs(self.path)
        try:
            content = self.get_content()
        except ValueError:
            self.send_response(400, message="Invalid Content-Length header.")
            self.end_headers()
            # pylint: disable=attribute-defined-outside-init
            self.close_connection = True
            return
        handler = getattr(self, "handle_" + parsed_request.get("topic", ("",))[0], None)
        try:
            content_to_write = handler(parsed_request, content)
        except Exception as unknown_exc:  # pylint: disable=broad-except
            self.log_message("Invalid request:\n" + traceback.format_exc())
            self.send_response(400, message=str(unknown_exc))
            content_to_write = None
        else:
            self.send_response(200)
        self.end_headers()
        if content_to_write is not None:
            self.wfile.write(content_to_write)
        # pylint: disable=attribute-defined-outside-init
        self.close_connection = True

    def handle_ping(self, request, content):
        """For topic=ping, do nothing, just acknowledge the request."""

    def do_GET(self):  # pylint: disable=invalid-name
        """Handle a GET request."""
        self.call_handler()

    def do_POST(self):  # pylint: disable=invalid-name
        """Handle a POST request."""
        self.call_handler()


class BaseServer(object):
    """Base class for HTTP network servers.

    On construction, starts the server in a separate thread.
    """

    def __init__(self, request_handler, cert_path):
        if platform.system().lower() in ["darwin", "windows"]:
            address = "127.0.0.1"  # LLNL-issued laptops have trouble running servers
        else:
            address = socket.gethostname()
        self.server = http_server.HTTPServer((address, 0), request_handler)
        if os.path.isfile(cert_path):
            context = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(cert_path)
            self.server.socket = context.wrap_socket(
                self.server.socket, server_side=True
            )
        self.profiler = backend.get_profiler()
        self.server_thread = threading.Thread(
            target=backend.profile,
            name="server_thread",
            args=(self.profiler, self.server.serve_forever),
        )
        self.server_thread.daemon = True
        self.server_thread.start()

    def __repr__(self):
        """Return string representation of constructor call"""
        return "{}()".format(type(self).__name__)

    @property
    def server_address(self):
        """Return an iterable of information about the server.

        This information should be sufficient for a worker to connect.
        """
        return tuple(str(entry) for entry in self.server.server_address)

    def shutdown(self, ensemble_db):
        """Shut down the server; stop sending and receiving information.

        This should only be called once there are no more workers/workers active.
        """
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()
        self.save_progress(ensemble_db)
        return self.profiler

    def save_progress(self, ensemble_db):
        """Save any temporary state to `ensemble_db`."""

    def is_alive(self):
        """Return True if the server is still alive and working, False otherwise."""
        return self.server_thread.is_alive()


class BaseClient(EnsembleDatabase):
    """Base class for client connections to `BaseServer` servers."""

    def __init__(self, cert_path, host, port):
        """Create a new WorkerClient instance connecting to a server at (host, port)."""
        self.host = host
        self.port = int(port)
        if os.path.isfile(cert_path):
            self.context = ssl.create_default_context(
                purpose=ssl.Purpose.SERVER_AUTH, cafile=cert_path
            )
            self.context.check_hostname = False
        else:
            self.context = None

    def __repr__(self):
        """Return string representation of constructor call"""
        return "{}({!r}, {!r}, {!r})".format(
            type(self).__name__, self.context, self.host, self.port
        )

    def ping(self, timeout=_DEFAULT_TIMEOUT):
        """Ping the server at the given host and port."""
        return self._get_response("GET", "ping", timeout=timeout)

    def _make_connection(self, timeout=_DEFAULT_TIMEOUT):
        """Make a HTTP connection to the server."""
        if self.context is not None:
            return http_client.HTTPSConnection(
                self.host, self.port, context=self.context, timeout=timeout
            )
        return http_client.HTTPConnection(self.host, self.port, timeout=timeout)

    # pylint: disable=too-many-arguments
    def _get_response(
        self, method, topic, query=None, body=None, timeout=_DEFAULT_TIMEOUT
    ):
        """Get the content of the server's response to a given query.

        :raise IOError: if the connection is unsuccessful.
        """
        query = {} if query is None else query
        query["topic"] = topic
        with contextlib.closing(self._make_connection(timeout)) as conn:
            # pylint: disable=no-member
            conn.request(method, urlencode(query), body=body)
            response = conn.getresponse()
            content = response.read()
        if response.status != 200:
            raise IOError(
                "Connection unsuccessful; error code: {}, reason: {}".format(
                    response.status, response.reason
                )
            )
        return content

    def _get_pickled_data(self, topic, query=None):
        """Unpickle and return data fetched from the server."""
        return pickle.loads(self._get_response("GET", topic, query))

    def _send_data_pickle(self, topic, query=None, data=None):
        """Pickle data and send it to the server."""
        return self._get_response(
            "POST", topic, query, pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL),
        )


class BackendRequestHandler(BaseRequestHandler):
    """Handles requests to the server. Each request instantiates an instance."""

    def handle_enable_runs(self, request, content):  # pylint: disable=unused-argument
        """Add enabled run_ids to some in-memory copy."""
        self.server.enabled_runs.extend(pickle.loads(content))

    def handle_runs_to_kill(self, request, content):  # pylint: disable=unused-argument
        """Return a pickled copy of the in-memory copy of runs to kill."""
        return pickle.dumps(self.server.runs_to_kill, protocol=pickle.HIGHEST_PROTOCOL)

    def handle_add_result(self, request, content):  # pylint: disable=unused-argument
        """Add status to some in-memory copy and wait for it to be written."""
        run_ids, results = pickle.loads(content)
        self.server.run_results.extend(zip(run_ids, results))

    def handle_set_run_status(
        self, request, content
    ):  # pylint: disable=unused-argument
        """Add status to some in-memory copy and wait for it to be written."""
        run_ids, statuses = pickle.loads(content)
        self.server.run_statuses.extend(zip(run_ids, statuses))

    def handle_new_runs(self, request, content):  # pylint: disable=unused-argument
        """Handle a worker's args POST request"""
        num_requested_new_runs = pickle.loads(content)
        return pickle.dumps(
            self.server.run_feeder.new_runs(num_requested_new_runs),
            protocol=pickle.HIGHEST_PROTOCOL,
        )


class BackendServer(BaseServer):
    """Server for communicating between the backend process and worker processes.

    Provides the subset of the themis.database.EnsembleDatabase functionality that
    workers need.
    """

    def __init__(self, cert_path, run_feeder):
        super(BackendServer, self).__init__(BackendRequestHandler, cert_path)
        self.enabled_runs = collections.deque()
        self.runs_to_kill = collections.deque()
        self.run_statuses = collections.deque()
        self.run_results = collections.deque()
        self._export_attributes_to_server(run_feeder)

    def __repr__(self):
        """Return string representation of constructor call"""
        return "{}()".format(type(self).__name__)

    def _export_attributes_to_server(self, run_feeder):
        self.server.run_feeder = run_feeder
        self.server.enabled_runs = self.enabled_runs
        self.server.runs_to_kill = self.runs_to_kill
        self.server.run_results = self.run_results
        self.server.run_statuses = self.run_statuses

    def save_progress(self, ensemble_db):
        """Write the server's caches to the database."""
        ensemble_db.set_run_status(*copy_and_empty2(self.run_statuses))
        ensemble_db.enable_runs(copy_and_empty(self.enabled_runs))
        ensemble_db.add_result(*copy_and_empty2(self.run_results))
        self.runs_to_kill.clear()
        self.runs_to_kill.extend(ensemble_db.runs_to_kill())

    def is_alive(self):
        """Return True if the server is still alive and working, False otherwise."""
        return self.server_thread.is_alive()


class BackendClient(BaseClient):
    """Represents a connection to a BackendServer instance."""

    def runs_to_kill(self):
        """Mirrors EnsembleDatabase method."""
        return self._get_pickled_data("runs_to_kill")

    def new_runs(self, limit):
        """Mirrors EnsembleDatabase method."""
        return pickle.loads(self._send_data_pickle("new_runs", data=limit))

    def enable_runs(self, run_ids):
        """Mirrors EnsembleDatabase method."""
        self._send_data_pickle("enable_runs", data=run_ids)

    def add_result(self, run_ids, results):
        """Mirrors EnsembleDatabase method."""
        self._send_data_pickle("add_result", data=(run_ids, results))

    def set_run_status(self, run_ids, statuses):
        """Mirrors EnsembleDatabase method."""
        self._send_data_pickle("set_run_status", data=(run_ids, statuses))
