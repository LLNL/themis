"""Networking classes for communicating between workers and their sub-scripts."""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import collections
import threading

from themis.backend import network
from themis.versions import pickle, UserDict


class ThreadSafeDict(UserDict, object):  # pylint: disable=too-many-ancestors
    """Dictionary that protects certain operations with an RLock object."""

    def __init__(self, *args, **kwargs):
        self.lock = threading.RLock()
        super(ThreadSafeDict, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        with self.lock:
            return super(ThreadSafeDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        with self.lock:
            return super(ThreadSafeDict, self).__getitem__(key)

    def __delitem__(self, key):
        with self.lock:
            return super(ThreadSafeDict, self).__delitem__(key)

    def items(self):
        with self.lock:
            return tuple(super(ThreadSafeDict, self).items())

    def keys(self):
        with self.lock:
            return tuple(super(ThreadSafeDict, self).keys())


class WorkerServerRequestHandler(network.BaseRequestHandler):
    """Handles requests to the server. Each request instantiates an instance."""

    def handle_results(self, request, content):
        """Handle a worker's results POST request"""
        self.server.themis_results.append(
            (int(request["runid"][0]), pickle.loads(content))
        )

    def handle_setup(self, request, content):  # pylint: disable=unused-argument
        """Handle a worker's setup GET request"""
        return self.server.themis_run_infos[int(request["runid"][0])]


class WorkerServer(network.BaseServer):
    """Serve database requests needed by a workers' subscripts."""

    def __init__(self, cert_path, app_spec):
        super(WorkerServer, self).__init__(WorkerServerRequestHandler, cert_path)
        self.app_spec = app_spec
        self.results = collections.deque()
        self.run_infos = ThreadSafeDict()
        self.server.themis_run_infos = self.run_infos
        self.server.themis_results = self.results

    def submit(self, run_info_iterable):
        """Add the RunInfo objects to the cache for distribution."""
        for run_info in run_info_iterable:
            self.run_infos[run_info.run_id] = pickle.dumps(
                {"run_info": run_info, "app_spec": self.app_spec},
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def remove_completed_runs(self, run_ids_to_delete):
        """Remove completed runs from the cache.

        Allows the server to free up memory; otherwise the server's memory
        would continue growing the longer it runs.
        """
        for run_id in run_ids_to_delete:
            del self.run_infos[run_id]

    def save_progress(self, ensemble_db):
        """Write the server's cache of results to the database.

        Since the results are not used anywhere else, the
        results can be deleted as soon as they are written.
        """
        results_as_list = []
        while self.results:
            results_as_list.append(self.results.popleft())
        if results_as_list:
            ensemble_db.add_result(
                tuple(pair[0] for pair in results_as_list),
                tuple(pair[1] for pair in results_as_list),
            )


class WorkerClient(network.BaseClient):
    """Class for making calls to a WorkerServer instance."""

    def get_worker_setup(self, run_id):
        """Get the setup for a run from the server."""
        payload = self._get_pickled_data("setup", {"runid": run_id})
        return (payload["app_spec"], payload["run_info"])

    def add_result(self, run_id, result):
        """Send the result to the server."""
        self._send_data_pickle("results", {"runid": run_id}, result)
