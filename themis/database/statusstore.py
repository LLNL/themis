"""This module contains the SQLiteStatusStore class."""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sqlite3

from themis.database import utils
from themis.database.utils import cursorcloser, sql_transaction
from themis.resource.allocators import Allocation


class SQLiteStatusStore(utils.SQLiteMixin):
    """SQLite database holding global information about an ensemble."""

    _SELECT_ALLOC = (
        "SELECT nodes, partition, bank, name, timeout, "
        "repeats FROM Allocations WHERE id = ?"
    )

    def __init__(self, db_path):
        """Initialize a new instance."""
        super(SQLiteStatusStore, self).__init__(db_path)
        self._table_names.extend(("EnsembleStatus", "Allocations"))
        self._db_handle.row_factory = sqlite3.Row

    def create(self):
        """Create the EnsembleStatus table.

        This table contains all the ensemble status information, which is described in
        the `get_ensemble_status` method.
        """
        self._execute_shorthand(
            (
                "CREATE TABLE EnsembleStatus (id SMALLINT PRIMARY KEY, "
                "prepped SMALLINT, stop SMALLINT, exit SMALLINT, "
                "failed_runs INTEGER)",
                None,
            ),
            ("INSERT INTO EnsembleStatus VALUES (0, 0, 0, 0, 0)", None),
            (
                "CREATE TABLE Allocations (id INTEGER PRIMARY KEY, nodes INTEGER, "
                "partition TEXT, bank TEXT, name TEXT, timeout REAL, "
                "repeats INTEGER)",
                None,
            ),
        )

    def fetch(self):
        """Return a mapping containing all the ensemble status information.

        This information consists of the following value;
        * A boolean indicating whether prep_ensemble has been called
        * A boolean indicating whether the ensemble should be stopped
        * A boolean indicating whether the ensemble may exit
        * An integer number of ensemble-wide failed runs
        * An integer number of allocation repeats remaining for the ensemble
        """
        return self._execute_shorthand(
            ("SELECT * FROM EnsembleStatus WHERE id = 0", None)
        )

    def set(self, **kwargs):
        """Update the ensemble status, setting `field = value`."""
        with utils.cursorcloser(self._db_handle) as cursor:
            with utils.sql_transaction(cursor):
                for key, value in kwargs.items():
                    cursor.execute(
                        "UPDATE EnsembleStatus SET {key} = (?) WHERE id = 0".format(
                            key=key
                        ),
                        (value,),
                    )

    def increment_failed_runs(self, increment):
        """Increment the number of ensemble-wide failed runs by `increment`.

        Return the new value.
        """
        return self._execute_shorthand(
            (
                "UPDATE EnsembleStatus SET failed_runs = failed_runs + ? WHERE id = 0",
                (increment,),
            ),
            ("SELECT failed_runs FROM EnsembleStatus WHERE id = 0", None),
        )[0]

    def add_allocation(self, alloc):
        """Add an allocation to the database and return its ID."""
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            cursor.execute(
                "INSERT INTO Allocations(nodes, partition, bank, name, timeout, "
                "repeats) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    alloc.nodes,
                    alloc.partition,
                    alloc.bank,
                    alloc.name,
                    alloc.timeout,
                    alloc.repeats,
                ),
            )
            return cursor.lastrowid

    def get_allocation(self, alloc_id):
        """Get the allocation specified by the id."""
        row = self._execute_shorthand((self._SELECT_ALLOC, (alloc_id,)))
        return Allocation(**row)

    def decrement_repeats(self, alloc_id):
        """Decrement the number of allocation repeats left for an allocation.

        Return the new value.
        """
        return Allocation(
            **self._execute_shorthand(
                (
                    "UPDATE Allocations SET repeats = repeats - 1 WHERE id = ?",
                    (alloc_id,),
                ),
                (self._SELECT_ALLOC, (alloc_id,)),
            )
        )
