"""Common functions and utility functions for `themis.database`."""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sqlite3 as sql
import pickle
import contextlib


def unpickle_or_none(value):
    """Unpickle a non-None value."""
    return pickle.loads(value) if value is not None else None


class EnsembleDatabase(object):  # pylint: disable=too-few-public-methods
    """Base class for all ensemble backends."""

    # A sort of enum for run completion statuses
    RUN_QUEUED = 0
    RUN_SUCCESS = 1
    RUN_FAILURE = 2
    RUN_KILLED = 3
    RUN_ABORTED = 4


@contextlib.contextmanager
def cursorcloser(conn):
    """A context manager for closing cursor objects.

    :param conn: a Python DB API 2.0 Connection object.
    """
    # Code to acquire resource, e.g.:
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        conn.commit()
        cursor.close()


@contextlib.contextmanager
def sql_transaction(cursor, immediate=False):
    """A context manager for managing SQL transactions.

    Issues a "BEGIN" statement when entered, and then
    a "COMMIT" if no errors were raised within the managed block,
    or a "ROLLBACK" otherwise.

    :param cursor: a Python DB API 2.0 Cursor object for a SQL database.
    :param immediate: used in sqlite to begin a write transaction
        immediately, not waiting for the first write statement.
    """
    if immediate:
        cursor.execute("BEGIN IMMEDIATE")
    else:
        cursor.execute("BEGIN")
    try:
        yield
    except:
        cursor.execute("ROLLBACK")
        raise
    else:
        cursor.execute("COMMIT")


class SQLiteMixin(object):
    """Defines a handful of useful methods for objects encapsulating SQLite DBs.

    A connection to the DB is opened on construction and closed
    on garbage collection.
    """

    def __init__(self, db_path, timeout=60):
        """Constructor. Opens a connection to the database.

        The connection is only closed on object destruction.
        """
        self._db_path = db_path
        self._db_handle = sql.connect(
            db_path, timeout=timeout, isolation_level=None, check_same_thread=False
        )
        self._db_handle.text_factory = str  # needed only for Python 2
        self._table_names = []

    def __del__(self):
        """Destructor. Closes the connection to the database."""
        if hasattr(self, "_db_handle"):
            # this check is necessary if the constructor raises an exception
            self._db_handle.close()

    def __repr__(self):
        """Return str representation of constructor call"""
        return "{}({!r})".format(type(self).__name__, self._db_path)

    def delete(self):
        """Empty the database of all data. Deletes all tables."""
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            for table in self._table_names:
                cursor.execute("DROP TABLE IF EXISTS {}".format(table))

    def _execute_shorthand(self, *query_param_pairs):
        """Helper method for executing a series of queries."""
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            for query, params in query_param_pairs:
                if params is not None:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
            return cursor.fetchone()
