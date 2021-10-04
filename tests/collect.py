"""Command-line program for combining multiple test results."""

import glob
import json
import argparse
import os

import utils


def setup_parser():
    """Set up the command line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Combine test suite results, "
            "then print them and write them to a file."
        ),
    )
    parser.add_argument(
        "suites",
        help="the test suites to collect from",
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "-d", "--directories",
        help="the directories to search for test results",
        type=str,
        nargs="+",
        default=["."],
    )
    parser.add_argument(
        "-o", "--outfile",
        help="file path to write results to",
        type=str,
        default="combined_results.json"
    )
    return parser


def print_failures(results, msg, newline=False):
    """Print an iterable of 3-tuple test results."""
    for test_method, explanation, version in sorted(results):
        print("=" * 70)
        print("Python {} {}: {}".format(version, msg, test_method))
        print("=" * 70)
        print(explanation)
        if newline:
            print()


def combine(result_iterable):
    """Combine multiple dictionaries into one."""
    to_return = {key: [] for key in result_iterable[0].keys()}
    for result_dict in result_iterable:
        for key in result_dict.keys():
            to_return[key].extend(result_dict[key])
    return to_return


def main():
    args = setup_parser().parse_args()
    test_files = []
    for suite in args.suites:
        for directory in args.directories:
            test_files.extend(glob.glob(
                os.path.join(directory, utils.RESULT_FILE.format(suite, "*"))
            ))
    results = []
    for file_name in test_files:
        with open(file_name, "r") as file_handle:
            results.append(json.load(file_handle))
    if not results:
        raise ValueError("No results found")
    complete_results = combine(results)
    print_failures(complete_results["failures"], "FAIL")
    print_failures(complete_results["errors"], "ERROR")
    print_failures(complete_results["unexpectedSuccesses"], "UNEXPECTED SUCCESS")
    print()
    print("Tests run: {}, skips: {}, expected failures: {}".format(
            sum(complete_results["testsRun"]),
            len(complete_results["skipped"]),
            len(complete_results["expectedFailures"])
        )
    )
    print("Total failures: {}, errors: {}, unexpected successes: {}".format(
            len(complete_results["failures"]),
            len(complete_results["errors"]),
            len(complete_results["unexpectedSuccesses"])
        )
    )
    with open(args.outfile, "w") as file_handle:
        json.dump(complete_results, file_handle, indent=2)


if __name__ == '__main__':
    main()
