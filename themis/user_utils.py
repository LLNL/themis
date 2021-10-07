"""
The :code:`ensemble.user_utils` module exports certain functions that
are designed to be called from an application interface.

Since the application interface functions (``prep_run``, etc.) take no arguments,
they do not automatically have access to any information about the ensemble.
For this reason,
certain information is exported, so that, for instance, ``prep_run`` can find out
what run it is, what the sample point is for that run, and how many total runs
there are. This information is exported to the :code:`ensemble.user_utils` module;
to access it, just add :code:`from themis import user_utils` to the top
of your application interface.

If any of these functions are called outside an application interface,
their behavior is undefined.

Many of the exported functions are designed to be used by specific
application interface functions, and their behavior may change depending
on the caller (e.g. a function only meant to be used by ``prep_ensemble``
might raise an error when called by ``post_run``);
consult the documentation of each function for more information.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import themis

_RUN_ID = None  # run ID of current run
_STATUS_DB = None  # status DB handle
_SETUP_DIR = None  # setup dir for ensemble
_RUN = None  # Run object for current run


def run_id():
    """Return the integer ID of the current run.

    For use by ``prep_run`` or ``post_run``. Returns ``None`` if called
    by ``prep_ensemble`` or ``post_ensemble``.
    """
    return _RUN_ID


def run():
    """Return the ``themis.CompositeRun`` object for the current run.

    This can be used to obtain the sample, arguments, and resource requirements
    for the current run.

    For use by ``prep_run`` or ``post_run``. Returns ``None`` if called
    by ``prep_ensemble`` or ``post_ensemble``.
    """
    return _RUN


def themis_handle():
    """Return a ``themis.Themis`` for this ensemble."""
    if _SETUP_DIR is not None:
        return themis.Themis(setup_dir=_SETUP_DIR)
    return None


def stop_ensemble():
    """Stop the ensemble from initiating any new runs.

    Existing runs will complete, and then the ensemble will exit
    after calling the user's ``post_ensemble`` (if it exists).
    """
    if _RUN_ID is not None and _STATUS_DB is not None:
        _STATUS_DB.set(stop=True)
