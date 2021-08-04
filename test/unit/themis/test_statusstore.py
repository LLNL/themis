from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import unittest
import sys
import os
import sqlite3 as sql

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

from themis.database import statusstore
from themis.resource.allocators import Allocation
from themis import utils


class StatusStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = statusstore.SQLiteStatusStore("statusstore_tests.db")
        self.db.delete()
        self.db.create()

    def test_fetch_and_set(self):
        """Test getting and setting various aspects of ensemble status"""
        statuses_to_test = ("stop", "exit", "prepped")
        # test each parameter one at a time
        for status_str in statuses_to_test:
            self.assertFalse(self.db.fetch()[status_str])
            self.db.set(**{status_str: True})
            self.assertTrue(self.db.fetch()[status_str])
        # now test bulk edits
        self.db.set(**{status_str: False for status_str in statuses_to_test})
        updated_status = self.db.fetch()
        self.assertFalse(
            any(updated_status[status_str] for status_str in statuses_to_test)
        )

    def test_increment_failed_runs(self):
        """Test incrementing failed runs"""
        expected_failures = 0
        for _ in range(5):
            self.assertEqual(expected_failures, self.db.fetch()["failed_runs"])
            expected_failures += 5
            self.assertEqual(expected_failures, self.db.increment_failed_runs(5))

    def test_add_get_allocation(self):
        for alloc in (
            Allocation(nodes=5),
            Allocation(timeout=10),
            Allocation(partition="foobar"),
        ):
            alloc_id = self.db.add_allocation(alloc)
            db_alloc = self.db.get_allocation(alloc_id)
            for attr in ("nodes", "timeout", "name", "partition", "bank", "repeats"):
                self.assertEqual(getattr(db_alloc, attr), getattr(alloc, attr))

    def test_decrement_repeats(self):
        repeats = 5
        alloc_id = self.db.add_allocation(Allocation(repeats=repeats))
        self.assertEqual(self.db.get_allocation(alloc_id).repeats, repeats)
        for i in range(repeats - 1, -1, -1):
            self.assertEqual(i, self.db.decrement_repeats(alloc_id).repeats)


if __name__ == "__main__":
    unittest.main()
