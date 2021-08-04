"""Tests for ensemble/resource.py"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import sys
import os.path

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

from themis import resource
from themis.resource import Slurm, Lsf, NoResourceManager, Flux, LANLSlurm


class ResourceManagerClassTests(unittest.TestCase):
    def test_flux_topology(self):
        self.assertEqual([30], Flux._calculate_topology(30, max_split=31))
        self.assertEqual(
            [15, 2], Flux._calculate_topology(30, max_split=20, default_split=15)
        )
        self.assertEqual(
            [5, 12], Flux._calculate_topology(60, max_split=30, default_split=5)
        )
        self.assertEqual([5, 5, 12], Flux._calculate_topology(300, default_split=5))
        self.assertEqual([5, 5, 5, 12], Flux._calculate_topology(1500, default_split=5))
        self.assertEqual(
            [5, 7], Flux._calculate_topology(31, max_split=15, default_split=5)
        )

    def test_flux_stringify_topology(self):
        self.assertEqual("5x7", Flux._stringify_topology([5, 7]))
        self.assertEqual("5x5x5x12", Flux._stringify_topology([5, 5, 5, 12]))


class UtilityFunctionTests(unittest.TestCase):
    def test_list_resource(self):
        ids = resource.list_resource_mgr_identifiers()
        expected_ids = ["slurm", "lsf", "flux", "none"]
        self.assertTrue(all(expected_id in ids for expected_id in expected_ids))

    def test_identify_resource_mgr(self):
        rmgr_pairs = (
            ("slurm", Slurm),
            ("lsf", Lsf),
            ("none", NoResourceManager),
            ("flux", Flux),
            ("lanl-slurm", LANLSlurm),
        )
        for rmgr_id, rmgr_class in rmgr_pairs:
            self.assertEqual(
                type(resource.identify_resource_manager(rmgr_id)), rmgr_class
            )

    def test_valid_resource(self):
        for rmgr_id in resource.list_resource_mgr_identifiers():
            self.assertTrue(resource.valid_resource_mgr(rmgr_id))


if __name__ == "__main__":
    unittest.main()
