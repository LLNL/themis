"""
Unit tests for the prepper script.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import sys
import os
import collections
import logging
import shutil

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 4)[0])

from themis.backend.worker import prepper
from themis import user_utils
from themis import utils
from themis import resource
from themis.backend import clear_user_utils, export_to_user_utils
from themis.utils import Step

SUPPORTING_FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support")

_PREP_RUN_VAL = None

ShamRunInfo = collections.namedtuple("ShamRunInfo", ["sample", "steps"])


def prep_run_1():
    global _PREP_RUN_VAL
    _PREP_RUN_VAL = user_utils.run_id()


def prep_run_2():
    """Add an argument to the sleep command"""
    global _PREP_RUN_VAL
    _PREP_RUN_VAL = str(user_utils.run().sample["sleep_sec"])


def prep_run_3():
    raise ValueError()


class PrepperTests(unittest.TestCase):

    INPUT_DECKS = [
        os.path.join(SUPPORTING_FILES, input_deck_name)
        for input_deck_name in (
            "example_input_deck.txt",
            "alternate_input_deck_syntax.txt",
            "docs_input_deck.txt",
            "mini_mpi_app_input.txt",
        )
    ]

    @classmethod
    def setUpClass(cls):
        logging.getLogger().setLevel(logging.CRITICAL)
        cls.app_spec = {
            "run_parse": ["/blah/blah.csv"],
            "root_dir" : os.getcwd(),
            "setup_dir": None,
        }
        cls.run_ids = range(10)
        cls.ensemble_db = None
        cls.runs = [
            utils.Run({"sleep_sec": 2 * i}, cores_per_task=5, gpus_per_task=7)
            for i in cls.run_ids
        ]
        for run in cls.runs:
            run.application = "/foo/bar"

    @classmethod
    def tearDownClass(cls):
        logging.getLogger().setLevel(logging.WARNING)

    def tearDown(self):
        global _PREP_RUN_VAL
        _PREP_RUN_VAL = None
        clear_user_utils()
        # remove files in the cwd (especially symlinks) which can confuse other tests
        for file_path in self.INPUT_DECKS:
            file_name = os.path.basename(file_path)
            if os.path.exists(file_name):
                os.remove(file_name)

    def test_user_prep_run_1(self):
        export_to_user_utils(self.app_spec, self.ensemble_db, run_id=0)
        prepper.user_prep_run(".", prep_run_1)
        self.assertEqual(
            0, _PREP_RUN_VAL,
        )

    def test_user_prep_run_2(self):
        for run_id in self.run_ids:
            export_to_user_utils(
                {"root_dir": os.getcwd(), "setup_dir": None},
                None,
                run_id,
                self.runs[run_id],
            )
            prepper.user_prep_run(".", prep_run_2)
            self.assertEqual(str(self.runs[run_id].sample["sleep_sec"]), _PREP_RUN_VAL)
            clear_user_utils()

    def test_user_prep_run_3(self):
        with self.assertRaises(ValueError):
            prepper.user_prep_run(".", prep_run_3)

    def test_parse_input_deck_base(self):
        input_deck_name = "example_input_deck.txt"
        source = os.path.join(SUPPORTING_FILES, input_deck_name)
        target = os.path.abspath(input_deck_name)
        self._test_parse_input_deck_base(source, target)

    def test_parse_input_deck_samefile(self):
        """Test that the same file can be both source and target."""
        input_deck_name = "example_input_deck.txt"
        source = os.path.abspath(input_deck_name)
        orig_source = os.path.join(SUPPORTING_FILES, input_deck_name)
        shutil.copyfile(orig_source, source)
        self._test_parse_input_deck_base(source, source)

    def _test_parse_input_deck_base(self, source, target):
        prepper.parse_input_deck_base(
            source, target, {"sample_1": "hello", "sample_2": 5, "sample_3": 3.9},
        )
        with open(target) as file_handle:
            lines = file_handle.readlines()
        self.assertEqual(lines[0], "sample1 = hello\n")
        self.assertEqual(lines[1], "sample2 = 5\n")
        self.assertEqual(lines[2], "sample3 = 3.9\n")

    def test_parse_input_deck(self):
        input_deck_name = "mini_mpi_app_input.txt"
        template_path = os.path.join(SUPPORTING_FILES, input_deck_name)
        prepper.parse_input_deck(template_path, input_deck_name, {"sleep_hello": 27})
        with open(input_deck_name) as file_handle:
            lines = file_handle.readlines()
        self.assertEqual(int(lines[0]), 27)

    def test_parse_input_deck_jinja(self):
        input_deck_name = "jinja_input_deck.html"
        source = os.path.join(SUPPORTING_FILES, input_deck_name)
        target = os.path.abspath(input_deck_name)
        self._test_parse_input_deck_jinja(source, target)

    def test_parse_input_deck_jinja_samefile(self):
        """Test that the same file can be both source and target."""
        input_deck_name = "jinja_input_deck.html"
        source = os.path.abspath(input_deck_name)
        orig_source = os.path.join(SUPPORTING_FILES, input_deck_name)
        shutil.copyfile(orig_source, source)
        self._test_parse_input_deck_jinja(source, source)

    def _test_parse_input_deck_jinja(self, source, target):
        try:
            import jinja2
        except ImportError:
            self.skipTest("jinja not available")
        prepper.parse_input_deck(
            source, target, {"myint": 27, "mylist": [1000], "mymapping": {"key": 50}},
        )
        with open(target) as file_handle:
            lines = file_handle.readlines()
        self.assertEqual(int(lines[0]), 40)  # add 13 to myint
        self.assertEqual(int(lines[2]), 1000)  # get first entry of mylist
        self.assertEqual(int(lines[3]), 50)  # get 'key' entry of mymapping

    def test_parse_input_deck_documentation(self):
        """Test the input deck example given in the documentation"""
        input_deck_name = "docs_input_deck.txt"
        template_path = os.path.join(SUPPORTING_FILES, input_deck_name)
        prepper.parse_input_deck(
            template_path, input_deck_name, {"viscocity": 45.8, "hydrostatics": 1740}
        )
        with open(input_deck_name) as file_handle:
            received_lines = file_handle.readlines()
        with open(
            os.path.join(SUPPORTING_FILES, "docs_input_deck_reference.txt")
        ) as file_handle:
            expected_lines = file_handle.readlines()
        self.assertEqual(received_lines, expected_lines)

    def test_parse_input_deck_alternate_syntax(self):
        """Test the input deck example given in the documentation"""
        input_deck_name = "alternate_input_deck_syntax.txt"
        template_path = os.path.join(SUPPORTING_FILES, input_deck_name)
        prepper.parse_input_deck_base(
            template_path,
            input_deck_name,
            {"hardness": 45.8, "density": 1740, "material": "phosphate"},
        )
        with open(input_deck_name) as file_handle:
            received_lines = file_handle.readlines()
        with open(
            os.path.join(SUPPORTING_FILES, "alternate_input_deck_syntax_reference.txt")
        ) as file_handle:
            expected_lines = file_handle.readlines()
        self.assertEqual(received_lines, expected_lines)

    @unittest.skip("tests an incomplete implmentation of Themis batch implementation") 
    def test_parse_input_deck_themis_launch(self):
        """Test themis_launch parsing"""
        input_deck_name = "parse_themis_launch.txt"
        template_path = os.path.join(SUPPORTING_FILES, input_deck_name)
        for resource_mgr_id in resource.list_resource_mgr_identifiers():
            resource_mgr = resource.identify_resource_manager(resource_mgr_id)
            run_info = ShamRunInfo({"foo": 17, "bar": "foobar"}, [Step(template_path)])
            app_spec = {
                "run_parse"   : (),
                "run_symlink" : (),
                "run_copy"    : (),
                "resource_mgr": resource_mgr_id,
            }
            prepper.preparation(
                None, os.getcwd(), app_spec, run_info.sample, run_info.steps
            )
            with open(input_deck_name) as file_handle:
                received_lines = file_handle.readlines()
            self.assertEqual(
                received_lines[0][:-1],
                " ".join(
                    resource_mgr.build_cmd(Step(template_path, batch_script=False))
                ),
            )

    def test_populate_run_dir_symlink(self):
        """WARNING: be careful with this test! It can wreck the other tests!"""
        prepper.populate_run_dir(self.INPUT_DECKS, (), (), {}, os.getcwd())
        for file_path in self.INPUT_DECKS:
            file_name = os.path.basename(file_path)
            self.assertTrue(os.path.islink(file_name))
            self.assertTrue(os.path.samefile(file_path, os.path.realpath(file_name)))

    def test_populate_run_dir_copy(self):
        prepper.populate_run_dir((), self.INPUT_DECKS, (), {}, os.getcwd())
        for file_path in self.INPUT_DECKS:
            file_name = os.path.basename(file_path)
            self.assertTrue(os.path.isfile(file_name))
            self.assertFalse(os.path.islink(file_name))

    def test_populate_run_dir_parse(self):
        prepper.populate_run_dir((), (), self.INPUT_DECKS, {}, os.getcwd())
        for file_path in self.INPUT_DECKS:
            file_name = os.path.basename(file_path)
            self.assertTrue(os.path.isfile(file_name))
            self.assertFalse(os.path.islink(file_name))


if __name__ == "__main__":
    unittest.main()
