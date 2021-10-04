"""Common functions and classes used by testing infrastructure."""

import os
import sys
import json
import unittest
import argparse
import functools

RESULT_FILE = "{}_test_results_py{}.json"

load_tests = None

class TextTestResultWithSuccesses(unittest.TextTestResult):
    """Variation of TextTestResult for saving successful tests."""

    def __init__(self, *args, **kwargs):
        super(TextTestResultWithSuccesses, self).__init__(*args, **kwargs)
        self.successes = []

    def addSuccess(self, test):
        """Register a successful test."""
        super(TextTestResultWithSuccesses, self).addSuccess(test)
        self.successes.append(test)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--testdir", action="append", default=[])
    return parser


def load_tests_func(test_dirs, loader, standard_tests, pattern):
    """Special-name function called by unittest to discover tests."""
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 2)[0])
    pattern = "test*.py" if pattern is None else pattern
    for test_dir in test_dirs:
        test_suite = loader.discover(
            start_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), test_dir),
            pattern=pattern,
            top_level_dir=os.path.dirname(os.path.abspath(__file__))
        )
        standard_tests.addTests(test_suite)
    return standard_tests


def write_results(test_results, prefix):
    """Write a summary of a `unittest.TestResult` to a JSON file.

    `prefix` is used to format `RESULT_FILE`, yielding the JSON file's name.
    """
    test_result_dict = {
        "errors": convert_test_case_tuples(test_results.errors),
        "failures": convert_test_case_tuples(test_results.failures),
        "skipped": convert_test_case_tuples(test_results.skipped),
        "expectedFailures": convert_test_case_tuples(test_results.expectedFailures),
        "unexpectedSuccesses": [
            (case.id(), "", sys.version_info.major)
            for case in test_results.unexpectedSuccesses
        ],
        "successes": [
            (case.id(), "", sys.version_info.major)
            for case in test_results.successes
        ],
        "testsRun": [test_results.testsRun],
    }
    file_name = RESULT_FILE.format(prefix, sys.version_info.major)
    with open(file_name, "w") as result_file:
        json.dump(test_result_dict, result_file, indent=2)


def main(suite=None):
    """Run the unittest command-line program and save results."""
    global load_tests

    args, unknown_args = setup_parser().parse_known_args()
    load_tests = functools.partial(load_tests_func, args.testdir)
    test_result = unittest.main(
        exit=False,
        testRunner=unittest.runner.TextTestRunner(
            resultclass=TextTestResultWithSuccesses
        ),
        argv=sys.argv[:1] + unknown_args
    ).result
    write_results(test_result, suite)
    sys.exit(int(not test_result.wasSuccessful()))


def convert_test_case_tuples(iterable_of_case_tuples):
    """Return only the ID of TestCase objects in 2-tuples."""
    return [
        (case.id(), explanation, sys.version_info.major)
        for case, explanation in iterable_of_case_tuples
    ]

if __name__ == "__main__":
    main()
