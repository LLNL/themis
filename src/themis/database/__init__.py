"""
This packagew defines the EnsembleDatabase class and supporting functions.

The EnsembleDatabase class is used to connect and interact with
the ensemble's backend. This backend is used all over this package--
by both workers and by the user-facing API.

The backend (whatever it is) is used both as a simple store of information
and as concurrency control. Most of the operations `must` be atomic.

The reference EnsembleDatabase implementation is the SQLiteDatabase class,
defined in this module.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import json

from themis.database.statusstore import SQLiteStatusStore
from themis.database import sqlitedb


APP_SPEC_NAME = "ensemble_spec.json"

STATUS_STORE_NAME = "ensemble_status.db"


def get_status_store(where=os.curdir):
    """Return the StatusStore instance to be used for the ensemble."""
    return SQLiteStatusStore(os.path.abspath(os.path.join(where, STATUS_STORE_NAME)))


def default_database(app_spec=None):
    """Return the EnsembleDatabase instance to be used for the ensemble.

    :param app_spec: the app_spec dictionary, which holds information
        about the type and location of the database. If None,
        it will be collected via the get_app_spec function.
    """
    if app_spec is None:
        app_spec = get_app_spec()
    if app_spec["database"]["type"] == "sqlite":
        return sqlitedb.SQLiteDatabase(app_spec["database"]["path"])
    raise ValueError("Database type not recognized: {}".format(app_spec["database"]))


def write_app_spec(app_spec, where=os.curdir):
    """Pickle the app_spec dictionary and dump to a file"""
    with open(os.path.join(where, APP_SPEC_NAME), "w") as file_handle:
        json.dump(app_spec, file_handle, indent=4)


def get_app_spec(where=os.curdir):
    """Collect the app_spec dictionary from the file system.

    :param where: the directory to seach for the json'd app_spec
        dictionary. If None, search in the cwd.
    """
    with open(os.path.join(where, APP_SPEC_NAME), "r") as file_handle:
        return json.load(file_handle)
