"""
This module defines the EnsembleDatabase class and supporting functions.

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

import pickle

from themis.database.utils import (
    EnsembleDatabase,
    SQLiteMixin,
    cursorcloser,
    sql_transaction,
    unpickle_or_none,
)
from themis.utils import CompositeRun, AugmentedRun


class SqlRunInfo(object):  # pylint: disable=too-few-public-methods
    """Represents and stores the information needed by workers to execute a run."""

    __slots__ = ("_row", "run_id", "restarts", "status", "steps", "sample")

    def __init__(self, row):
        self._row = row
        self.run_id = row[0]
        self.restarts = row[1]
        self.status = row[2]
        self.steps = pickle.loads(row[3])
        self.sample = pickle.loads(row[4])

    def __repr__(self):
        """Return str representation of constructor call"""
        return "{}({})".format(type(self).__name__, self._row)

    @property
    def run(self):
        """Return a ``CompositeRun`` object."""
        return CompositeRun(self.sample, self.steps)


class SQLiteDatabase(SQLiteMixin, EnsembleDatabase):
    """SQLite backend for the ensemble, using the sqlite3 standard library module.

    This is the reference EnsembleDatabase implementation.

    The database is stored as a local file. Only one connection can execute a
    write command at a time. See the SQLite documentation for more information.

    Note that SQLite connections cannot be safely be shared between threads.
    For multithreaded environments, each thread should have its own copy
    of an instance; multiple threads should not have references to the same instance.

    This database has one table, RunInfo. This table holds all the information about
    the runs in the ensemble. Each row is one run. The columns are as follows:
        runID: a unique nonnegative integer identifying the run. The primary key of the
            table. New rows have their ID generated automatically by SQLite.
        restarts: the number of times the run has been restarted due to a failure.
        step_num: the current step that the run is on. Each run consists of one or more
            steps.
        taken_bool: a boolean indicating whether the run is available to be grabbed
            for execution. 1 if the run is unavailable, 0 else.
        completion_status: an integer indicating the status of the run.
            One of the RUN_* enums.
        arg: the arguments for the run
        tasks: the number of tasks for the run
        cores_per_task: the cores per task for the run
        gpus_per_task: the gpus per task for the run
        timeout: the timeout for the run
        sample: the sample for the run, pickled
        result: the result for the run, pickled


    :param db_path: the path to the database. If the database does not exist, it
        will be created.
    """

    def __init__(self, db_path):
        """Constructor. Opens a connection to the database.

        The connection is only closed on object destruction.
        """
        super(SQLiteDatabase, self).__init__(db_path, timeout=200)
        self._table_names.append("RunInfo")

    def create(self):
        """Create and populate the RunInfo table.

        The RunInfo table makes up the bulk of the database; it stores all of
        the run-specific information. That includes the status of the current run,
        its command-line arguments, its resources, sample, and result, and more.

        Most of these are stored as integers. However, the samples and results
        are stored in pickled form as the BLOB type.
        """
        create_query = (
            "CREATE TABLE RunInfo (runID INTEGER PRIMARY KEY, "
            "restarts SMALLINT DEFAULT 0, step_num SMALLINT DEFAULT 0, "
            "taken_bool SMALLINT DEFAULT 0, "
            "completion_status SMALLINT DEFAULT {}, "
            "sample BLOB, steps BLOB, result BLOB)"
        ).format(int(self.RUN_QUEUED))
        with cursorcloser(self._db_handle) as cursor:
            with sql_transaction(cursor):
                cursor.execute(create_query)
                # create indices to improve performance on big DBs
                cursor.execute(
                    "CREATE INDEX new_run_index ON "
                    "RunInfo(taken_bool, completion_status)"
                )
                cursor.execute(
                    "CREATE INDEX completion_index ON RunInfo(completion_status)"
                )

    def add_result(self, run_ids, results):
        """Add a result or sequence of results to the database.

        :param run_ids: an int run_id or iterable of int run_ids,
            indicating which runs the results belong to
        :param results: a single result or iterable of results,
            matching in this respect (and in the length of the iterable)
            with run_ids
        """
        if isinstance(run_ids, int):
            run_ids = (run_ids,)
            results = (results,)
        results = [
            pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL) for result in results
        ]
        add_result_query = "UPDATE RunInfo SET result = ? WHERE runID = ?"
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            cursor.executemany(add_result_query, zip(results, run_ids))

    def get_result(self, run_ids):
        """Return the result values for a run or sequence of runs.

        :param run_ids: the run_ids to fetch results for
        :return: a mapping from run_ids to results
        """
        if isinstance(run_ids, int):
            run_ids = (run_ids,)
        pickled_results = []
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            for run_id in run_ids:
                cursor.execute("SELECT result FROM RunInfo WHERE runID = ?", (run_id,))
                pickled_results.append(cursor.fetchone())
        return {
            run_id: pickle.loads(result[0])
            if result is not None and result[0] is not None
            else None
            for run_id, result in zip(run_ids, pickled_results)
        }

    def _fetch_results(self, include_none):
        query = "SELECT runID, sample, steps, result, completion_status FROM RunInfo"
        if not include_none:
            query += " WHERE result IS NOT NULL"
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            cursor.execute(query)
            return cursor.fetchall()

    def as_mappings(self, include_none):
        """Return info about all runs as a list of mappings"""
        result_rows = self._fetch_results(include_none)
        return [
            {
                "run_id": row[0],
                "sample": pickle.loads(row[1]),
                "steps": pickle.loads(row[2]),
                "result": unpickle_or_none(row[3]),
                "status": row[4],
            }
            for row in result_rows
        ]

    def add_runs(self, runs):
        """Add an iterable of new sample points to the database.

        The run IDs of the new points is determined when they are added.
        """
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            cursor.executemany(
                "INSERT INTO RunInfo(sample, steps) VALUES (?, ?)",
                (
                    (
                        pickle.dumps(run.sample, protocol=pickle.HIGHEST_PROTOCOL),
                        pickle.dumps(run.steps, protocol=pickle.HIGHEST_PROTOCOL),
                    )
                    for run in runs
                ),
            )

    def get_run_info(self, run_id):
        """Return the RunInfo object for the run specified by run_id."""
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            cursor.execute(
                "SELECT runID, restarts, step_num, steps, "
                "sample FROM RunInfo WHERE runID = ?",
                (run_id,),
            )
            result = cursor.fetchone()
        if result is None:
            return None
        return SqlRunInfo(result)

    def user_facing_run_info(self, run_ids):
        """Return a mapping from run IDs to AugmentedRun objects for `run_ids`."""
        rows = []
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            for run_id in run_ids:
                cursor.execute(
                    "SELECT runID, completion_status, result, sample, "
                    "steps FROM RunInfo WHERE runID = ?",
                    (run_id,),
                )
                row = cursor.fetchone()
                if row is not None:
                    rows.append(row)
        return {
            row[0]: AugmentedRun(
                sample=pickle.loads(row[3]),
                status=row[1],
                result=pickle.loads(row[2]) if row[2] is not None else None,
                steps=pickle.loads(row[4]),
            )
            for row in rows
        }

    def set_run_status(self, run_ids, status_tuples):
        """Update the statuses of runs.

        The length of run_ids and status_tuples should be the same.

        :param run_ids: an iterable of int run_ids
        :param status_tuples: an iterable of 4-tuples with the following format:
            (step, completion code, active bool, number of restarts)
        """
        if not run_ids:
            return
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            cursor.executemany(
                "UPDATE RunInfo SET step_num = ?, completion_status = ?, "
                "restarts = ? WHERE runID = ?",
                (
                    status_tup + (run_id,)
                    for run_id, status_tup in zip(run_ids, status_tuples)
                ),
            )

    def disable_runs(self, run_ids):
        """Mark the runs indicated by run_ids as disabled.

        Disabled runs can never be collected by the new_runs method,
        unless they are later re-enabled by a call to enable_runs.
        """
        self._set_run_taken_bool(run_ids, 1)

    def enable_runs(self, run_ids):
        """Mark the runs indicated by run_ids as enabled.

        Unless the runs were disabled, this method has no effect.
        """
        self._set_run_taken_bool(run_ids, 0)

    def _set_run_taken_bool(self, run_ids, new_status):
        """Set the taken_bool of run_ids to new_status."""
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            cursor.executemany(
                "UPDATE RunInfo SET taken_bool = ? WHERE runID = ?",
                ((new_status, run_id) for run_id in run_ids),
            )

    def enable_all(self):
        """Mark all runs as enabled."""
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            cursor.execute(
                "UPDATE RunInfo SET taken_bool = 0 WHERE "
                "taken_bool = 1 AND completion_status = ?",
                (self.RUN_QUEUED,),
            )

    def new_runs(self, limit):
        """Return a sequence (with length <= limit) of new RunInfo objects.

        These RunInfo objects are collected atomically, so that no other
        processes or threads calling this method will get the same objects.
        """
        if limit == 0:
            return ()
        with cursorcloser(self._db_handle) as cursor, sql_transaction(
            cursor, immediate=True
        ):
            # need to select runIDs and update those runs'
            # taken_bool in a single transaction
            # otherwise another DB connection could take them
            cursor.execute(
                "SELECT runID, restarts, step_num, steps, sample FROM RunInfo "
                "WHERE taken_bool == 0 AND completion_status <= ? LIMIT ?",
                (self.RUN_QUEUED, limit),
            )
            results = cursor.fetchall()
            # now change the runs with intermediate status to their final status
            cursor.executemany(
                "UPDATE RunInfo SET taken_bool = 1 WHERE runID = ?",
                [(row[0],) for row in results],
            )
        return tuple(SqlRunInfo(result) for result in results)

    def _runs_with_status(self, statuses, count=False):
        """Return a tuple containing all the run ID's of runs that have a
        certain completion status.

        Alternatively, if `count` is True, return the number of runs with one of the
        given statuses.

        :param statuses: an iterable of EnsembleDatabase.RUN_* status objects.
        """
        select_run_id_status = "SELECT {} FROM RunInfo"
        if count:
            select_run_id_status = select_run_id_status.format("COUNT(runID)")
        else:
            select_run_id_status = select_run_id_status.format("runID")
        if statuses:
            select_run_id_status += (
                " WHERE completion_status IN (" + ("?, " * (len(statuses) - 1)) + "?)"
            )
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            cursor.execute(select_run_id_status, statuses)
            result = cursor.fetchall()
        if count:
            return result[0][0]
        # the run_ids are returned as 1-tuples, so we must unpack them
        return tuple(one_tuple[0] for one_tuple in result)

    def runs_with_completion_status(self, *statuses):
        """Return an iterable of run ids with one of the given completion statuses.

        :param statuses: zero or more EnsembleDatabase.RUN_* status objects.
            If status is empty, return all run ID's. Otherwise, return any run ID
            with a status contained in the statuses parameter.
        """
        return self._runs_with_status(statuses)

    def count_runs_by_completion(self, *statuses):
        """Return the number of runs with one of the given completion statuses.

        :param statuses: same meaning as in runs_with_completion_status.
        """
        return self._runs_with_status(statuses, True)

    def mark_runs_to_restart(self, run_ids, hard):
        """Mark the runs indicated by run_ids as eligible for restart."""
        if hard:
            set_step_num = "step_num = 0, "
        else:
            set_step_num = ""
        params = [(self.RUN_QUEUED, run_id, self.RUN_QUEUED) for run_id in run_ids]
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            cursor.executemany(
                "UPDATE RunInfo SET restarts = 0, taken_bool = 0, "
                + set_step_num
                + "completion_status = ? WHERE runID = ? AND "
                "completion_status > ?",
                params,
            )

    def mark_runs_to_kill(self, run_ids):
        """Mark that the runs indicated by run_ids should be killed."""
        with cursorcloser(self._db_handle) as cursor, sql_transaction(cursor):
            cursor.executemany(
                "UPDATE RunInfo SET completion_status = ?, taken_bool = 1 "
                "WHERE runID = ? AND completion_status <= ?",
                zip(
                    (self.RUN_KILLED for _ in run_ids),
                    run_ids,
                    (self.RUN_QUEUED for _ in run_ids),
                ),
            )

    def runs_to_kill(self):
        """Return a sequence of run IDs that should be killed."""
        return self.runs_with_completion_status(self.RUN_KILLED)
