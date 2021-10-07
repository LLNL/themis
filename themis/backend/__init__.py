"""Common functions and classes across the ensemble.backend package."""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import logging.handlers
import os
import sys
import ssl
import cProfile
import pstats
import copy
import subprocess
import collections
import signal
import contextlib

from themis.utils import DirectoryManager
from themis import user_utils
from themis.versions import reprlib


PROFILE_ENV_VAR = "THEMIS_PROFILE"
LOGGER = logging.getLogger(__name__)


class SignalCaught(Exception):
    """Exception raised on catching a signal"""

    def __init__(self, message, signum):
        super(SignalCaught, self).__init__(message)
        self.signum = signum


def signal_handler_exit(signum, _):
    """Signal handler for `signal` module."""
    LOGGER.info("Signal handler called with signal %s", signum)
    raise SignalCaught("Signal handler called with signal {}".format(signum), signum)


def handle_signals():
    """Attach a signal handler that raises an exception on SIGALRM and SIGTERM."""
    for signum in (signal.SIGALRM, signal.SIGTERM):
        signal.signal(signum, signal_handler_exit)


@contextlib.contextmanager
def managed_shutdown(thing, *args, **kwargs):
    """Invoke `shutdown` on `thing` upon completion of a block of code."""
    try:
        yield thing
    finally:
        thing.shutdown(*args, **kwargs)


def user_prep_ensemble(prep_ensemble, root_dir):
    """Call user's prep_ensemble function and return None.

    Any return value from the prep_ensemble function is ignored.

    :param prep_ensemble: the user's prep_ensemble function, a callable
    """
    with DirectoryManager(root_dir):
        try:
            prep_ensemble()
        except:  # pylint: disable=bare-except
            LOGGER.exception(
                "The following exception was generated "
                "by the prep_ensemble function:"
            )


def user_post_ensemble(post_ensemble, root_dir):
    """Call the user's post_ensemble. Return True if new points were added.

    :param post_ensemble: the user's post_ensemble function, a callable
    """
    # make sure we stay in the right directory
    LOGGER.info("Calling user's post_ensemble")
    with DirectoryManager(root_dir):
        try:
            post_ensemble()
        except:  # pylint: disable=bare-except
            LOGGER.exception(
                "The following exception was generated "
                "by the post_ensemble function:"
            )


def invoke_completion_script(on_completion_dict):
    """Invoke a user's completion script."""
    stdout = None
    stderr = None
    LOGGER.info("Launching completion script")
    try:
        if on_completion_dict["stdout"] is not None:
            stdout = open(on_completion_dict["stdout"], "a")
        if on_completion_dict["stderr"] is not None:
            stderr = open(on_completion_dict["stderr"], "a")
        proc = subprocess.Popen(
            on_completion_dict["args"],
            stdout=stdout,
            stderr=stderr,
            cwd=on_completion_dict["cwd"],
            universal_newlines=True,
        )
    except (OSError, FileNotFoundError):
        LOGGER.exception("Couldn't launch on_completion application:")
    else:
        # check to ensure process exited cleanly
        if proc.wait() != 0:
            LOGGER.error(
                "on_completion script exited with returncode %i", proc.returncode
            )
    finally:
        # close the IO streams
        if stdout is not None:
            stdout.close()
        if stderr is not None:
            stderr.close()


def profile(profiler, func, *args, **kwargs):
    """Profile a given function if PROFILE_ENV_VAR is set."""
    if os.getenv(PROFILE_ENV_VAR, ""):
        profiler.enable()
        return_val = func(*args, **kwargs)
        profiler.disable()
        return return_val
    return func(*args, **kwargs)


def get_profiler():
    """Return a Profile instance or None."""
    if os.getenv(PROFILE_ENV_VAR, ""):
        return cProfile.Profile()
    return None


def profile_from_new(func, *args, **kwargs):
    """Profile a function with a new profile """
    profiler = None
    if os.getenv(PROFILE_ENV_VAR, ""):
        profiler = cProfile.Profile()
    return (profiler, profile(profiler, func, *args, **kwargs))


def dump_profile_data(path, *profilers):
    """Dump profile data accumulated by `profilers` to a file at `path`."""
    if os.getenv(PROFILE_ENV_VAR, "") and profilers:
        profile_stats = pstats.Stats(profilers[0])
        for additional_profile in profilers[1:]:
            if additional_profile is not None:
                profile_stats.add(additional_profile)
            profile_stats.dump_stats(path)


def clear_user_utils():
    """Clear all information exported to the user_utils module."""
    # pylint: disable=protected-access
    user_utils._RUN_ID = None
    user_utils._STATUS_DB = None
    user_utils._SETUP_DIR = None
    user_utils._RESOURCES = None


def export_to_user_utils(app_spec, status_db, run_id=None, run=None):
    """Export the given information about this run to the user_utils module."""
    # pylint: disable=protected-access
    user_utils._RUN_ID = run_id
    user_utils._STATUS_DB = status_db
    user_utils._SETUP_DIR = app_spec["setup_dir"]
    user_utils._RUN = copy.deepcopy(run)


def _make_stream_handlers(streams):
    """Make logging handlers for writing to a stream."""
    formatter = logging.Formatter("THEMIS - %(levelname)s - %(asctime)s - %(message)s")
    handlers = []
    for stream in streams:
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)
    return handlers


def _make_queue_handlers(streams):
    """Make a QueueHandler for writing log messages asynchronously.

    Only available in Python 3.
    """
    import queue

    que = queue.Queue()
    queue_handler = logging.handlers.QueueHandler(que)
    stream_handlers = _make_stream_handlers(streams)
    listener = logging.handlers.QueueListener(
        que, *stream_handlers, respect_handler_level=True
    )
    listener.start()
    return (listener, [queue_handler])


def _make_handlers(streams):
    """Return a log handler."""
    if sys.version_info.major >= 3:
        return _make_queue_handlers(streams)
    return (None, _make_stream_handlers(streams))


def _setup_logger(streams):
    """Redirect stderr, stdout, and the root logger to log_file."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    listener, handlers = _make_handlers(streams)
    for handler in handlers:
        root_logger.addHandler(handler)
    return listener


@contextlib.contextmanager
def managed_logging(*streams):
    """Manage logging and shut down on exit."""
    listener = _setup_logger(streams)
    try:
        yield
    finally:
        if listener is not None:
            listener.stop()


class RunFeeder(object):
    """Instances of this class produce RunInfo objects representing new runs.

    :param ensemble_db: the EnsembleDatabase object
        to use for fetching new runs.
    :param cache_size: the size of the cache for RunInfo objects.
        A larger cache means less-frequent but larger trips to
        the database, but a smaller cache makes for better
        load-balancing among multiple RunFeeder instances.
    """

    def __init__(self, ensemble_db, cache_size, enable_runs=False):
        """Initialize a new instance."""
        self._ensemble_db = ensemble_db
        self._run_cache = collections.deque()
        self._disabled_run_ids = set()
        self._enable_runs = enable_runs
        self._cache_max_size = cache_size
        self._last_fill_attempt_succeeded = True
        self.fill_cache()

    def __repr__(self):
        """Return str representation of constructor call"""
        return "{}({}, {})".format(
            type(self).__name__, self._ensemble_db, self._cache_max_size
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    def new_runs(self, limit):
        """Return new runs and their statuses.

        Returns a two-tuple; the first entry (index 0) contains a sequence of
        run IDs identifying the new runs, the second contains the statuses
        corresponding to those IDs.

        :param limit: the max number of runs to return.
        """
        if limit <= 0:
            # return an empty tuple
            return ()
        return [
            self._run_cache.popleft() for _ in range(min(limit, len(self._run_cache)))
        ]

    def poll_cache(self, fill_factor=0.25):
        """Fill the cache if it is less than `fill_factor` full.

        If the last attempt at fetching runs from the database failed,
        don't fetch any runs.
        """
        if self._last_fill_attempt_succeeded and len(self._run_cache) < (
            fill_factor * self._cache_max_size
        ):
            self.fill_cache()

    def fill_cache(self):
        """Fill up the cache of RunInfo objects.

        Return True if the fill attempt succeeds, False other if it fails
        (because there are no eligible runs in the database).
        """
        runs_to_add = self._cache_max_size - len(self._run_cache)
        if runs_to_add < 1:
            return True
        new_runs = self._ensemble_db.new_runs(limit=runs_to_add)
        self._run_cache.extend(new_runs)
        if self._enable_runs:
            for run in new_runs:
                self._disabled_run_ids.add(run.run_id)
        if not new_runs:
            self._last_fill_attempt_succeeded = False
        else:
            self._last_fill_attempt_succeeded = True
        return self._last_fill_attempt_succeeded

    def is_empty(self):
        """Return True if all new runs have been exhausted, False otherwise."""
        return len(self._run_cache) <= 0

    def shutdown(self):
        """Enable all runs still held in the cache."""
        if self._disabled_run_ids:
            LOGGER.debug("Enabling runs %s", reprlib.repr(self._disabled_run_ids))
            self._ensemble_db.enable_runs(self._disabled_run_ids)
