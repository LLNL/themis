#!/usr/bin/env python

import os
import argparse
import numpy as np
import csv
import math
import json

# import pdb
import pickle

print("Parse Arguments")
# Get supplied Bamboo Build Number and location of history file.
parser = argparse.ArgumentParser(description="Retrieve ATS Test Results")
parser.add_argument("Json Results File", help="Json Results File")
parser.add_argument("J-Unit File", help="J-Unit File")
parser.add_argument("Test Suite", help="Test Suite Name")

args = vars(parser.parse_args())

print("Store Arguments")
# Name of file with test runtime data
resultsFileName = args["Json Results File"]
# Name of J-Unit xml file
junitFileName = args["J-Unit File"]
# Name of Test Suite
testSuiteName = args["Test Suite"]


class TestResult:
    def __init__(self, name, status, testResult):
        self.name = name
        self.status = status
        self.result = testResult.replace("<", "").replace(">", "")
        self.classname = "UQ"

    def getStatus(self):
        return self.status

    def getName(self):
        return self.name

    def getClassname(self):
        return self.classname

    def getResultDesc(self):
        return self.result

    def getElapsedTime(self):
        return 0


def convertToSeconds(str):
    answer = 0
    if len(str) != 0:
        HrMnSc = str.split(":")
        if len(HrMnSc) == 3:
            hours = float(HrMnSc[0])
            minutes = float(HrMnSc[1])
            seconds = float(HrMnSc[2])
            answer = 3600 * hours + 60 * minutes + seconds
        else:
            answer = float(str)
    return answer


def processResults(results_dict, tests_dict):
    for key, value in results_dict.items():
        parseTestCases(key, value)


def parseTestCases(state, results):
    if state == "successes":
        testStatus = "PASSED"
    elif state == "expectedFailures":
        testStatus = "IGNORED"
    elif state == "errors":
        testStatus = "NOT_PASSED"
    elif state == "unexpectedSuccesses":
        testStatus = "PASSED"
    elif state == "failures":
        testStatus = "NOT_PASSED"
    elif state == "skipped":
        testStatus = "SKIPPED"
    else:
        return
    for testcase in results:
        testNum = 0
        testDuration = 0
        testDurationSec = 0
        testName = testcase[0]
        testResultDesc = testcase[1]
        testResult = TestResult(testName, testStatus, testResultDesc)
        # testResult.classname = name
        if testStatus == "PASSED":
            tests["PASSED"].append(testResult)
        elif testStatus == "NOT_PASSED":
            tests["NOT_PASSED"].append(testResult)
        elif testStatus == "SKIPPED":
            tests["SKIPPED"].append(testResult)
        elif testStatus == "IGNORED":
            tests["IGNORED"].append(testResult)
        else:
            pass


# Dict of lists containing test results
tests = dict()
tests["PASSED"] = list()
tests["NOT_PASSED"] = list()
tests["SKIPPED"] = list()
tests["IGNORED"] = list()

try:
    results = open(resultsFileName, "rb")
except (OSError, IOError) as e:
    print("Failed to open results file: ", resultsFileName)
    exit(1)
else:
    test_dict = json.load(results)
    processResults(test_dict, tests)
    results.close()
    print("Num passed tests", len(tests["PASSED"]))
    print("Num unpassed tests", len(tests["NOT_PASSED"]))
    print("Num ignored tests", len(tests["IGNORED"]))


print("open J-Unit file")

# Open J-Unit file if it exists.
try:
    fil = open(junitFileName, "r+")
    lines = fil.readlines()
    fil.seek(0)
    for line in lines:
        if line == "</testsuites>\n":
            break
        #    print line
        fil.write(line)

except (OSError, IOError) as e:
    # File doesn't exist. Create J-Unit file.
    fil = open(junitFileName, "w")
    fil.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    fil.write("<testsuites>\n")

finally:
    # File exists.  Read historical data and append new data.

    fil.write('  <testsuite name="' + testSuiteName + '">\n')
    for testresult in tests["PASSED"]:
        fil.write(
            '    <testcase status="run" time="%.3f" classname="%s" name="%s"/>\n'
            % (
                testresult.getElapsedTime(),
                testresult.getClassname(),
                testresult.getName(),
            )
        )
    for testresult in tests["NOT_PASSED"]:
        fil.write(
            '    <testcase time="%.3f" classname="%s" name="%s">\n'
            % (
                testresult.getElapsedTime(),
                testresult.getClassname(),
                testresult.getName(),
            )
        )
        fil.write(
            '      <failure type="%s"> "%s" </failure>\n'
            % (testresult.getStatus(), testresult.getResultDesc())
        )
        fil.write("    </testcase>\n")
    fil.write("  </testsuite>\n")
    fil.write("</testsuites>\n")

    fil.close()

print("Parse Success - Failures: ", len(tests["NOT_PASSED"]))

if len(tests["NOT_PASSED"]) > 0:
    exit(1)
exit(0)
