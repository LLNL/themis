"""The ``themis.runtime`` module provides functions and operations for
applications launched by Themis.
"""

import os

from themis.backend.network import CERT_NAME
from themis.backend.worker import prepper, network
from themis.utils import (
    URI_ENV_VAR,
    RUNID_ENV_VAR,
    SETUPDIR_ENV_VAR,
    URI_SPLITCHAR,
    EXECDIR_ENV_VAR,
)
import themis


_RUNTIME_ERRMSG = "This function must be called from an application launched by Themis"


def _verify_themis_runtime():
    """Check environment variables and return a (run ID, themis connection) pair."""
    for env_var in (URI_ENV_VAR, RUNID_ENV_VAR, SETUPDIR_ENV_VAR, EXECDIR_ENV_VAR):
        if env_var not in os.environ:
            raise RuntimeError(_RUNTIME_ERRMSG)
    run_id = os.environ[RUNID_ENV_VAR]
    host, _, port = os.environ[URI_ENV_VAR].rpartition(URI_SPLITCHAR)
    cert_path = os.path.join(os.environ[EXECDIR_ENV_VAR], CERT_NAME)
    client = network.WorkerClient(cert_path, host, port)
    return (run_id, client)


def fetch_run_id():
    """Return the integer ID of the current run.

    :raises RuntimeError: if called outside of the Themis runtime environment.
    """
    if RUNID_ENV_VAR not in os.environ:
        raise RuntimeError(_RUNTIME_ERRMSG)
    return int(os.environ[RUNID_ENV_VAR])


def fetch_run():
    """Return the ``themis.CompositeRun`` object for the current run.

    :raises RuntimeError: if called outside of the Themis runtime environment.
    """
    run_id, client = _verify_themis_runtime()
    _, run_info = client.get_worker_setup(run_id)
    return run_info.run


def themis_handle():
    """Return a ``themis.Themis`` handle for the current ensemble.

    :raises RuntimeError: if called outside of the Themis runtime environment.
    """
    if SETUPDIR_ENV_VAR not in os.environ:
        raise RuntimeError(_RUNTIME_ERRMSG)
    return themis.Themis(setup_dir=os.environ[SETUPDIR_ENV_VAR])


def set_result(result):
    """Set the result for the current run.

    ``result`` should be an arbitrary Python object consisting only
    of built-in types, rather than custom classes.

    :raises RuntimeError: if called outside of the Themis runtime environment.
    """
    run_id, client = _verify_themis_runtime()
    client.add_result(run_id, result)


def parse_file(source, destination, sample=None):
    """Parse and token-replace a text file based on the key-value pairs in ``sample``.

    :param source: the path to the text file to parse.
    :param destination: the destination for the resulting parsed file
    :param sample: the key-value pairs to use for the token-replacement operation.
        If ``None``, fetch the sample by calling ``themis.runtime.run().sample``.
    """
    if sample is None:
        sample = fetch_run().sample
    prepper.parse_input_deck(source, destination, sample)
