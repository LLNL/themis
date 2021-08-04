#!/bin/bash

# Run tests in both python 2 and 3.
# Pass the name of the suites to run on the command line, e.g. 'unit'
# Arguments after "--" are passed to the python testing scripts.
# This allows you to pass command-line arguments to the unittest framework
# For instance: ``./run_tests.sh unit integration -- -vvf`` runs the unit
# and integration tests in verbose (-vv), fail-fast mode (-f).
# Export the UQP_PY2 and UQP_PY3 environment variables to determine the
# Python 2 and 3 installations to use for the tests.

die() { echo "$*" 1>&2 ; exit 1; }

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ -z "$UQP_PY2" ]; then UQP_PY2=$(which python2); fi
if [ -z "$UQP_PY3" ]; then UQP_PY3=$(which python3); fi
if [[ ! -x $UQP_PY2 ]] || [[ -z $UQP_PY2 ]]; then die "Python 2 not found!"; fi
if [[ ! -x $UQP_PY3 ]] || [[ -z $UQP_PY3 ]]; then die "Python 3 not found!"; fi

TEST_SUITES=()

while [[ $# -gt 0 ]]; do
	if [[ "$1" == "--" ]]; then
		shift
		break
	else
		TEST_SUITES+=("$1")
		shift
	fi
done

SUITE_DIRS=()

for SUITE in "${TEST_SUITES[@]}"; do
	echo "Running ${SUITE} tests in python 2..."
	mkdir -p "${SUITE}_2"; cd "${SUITE}_2"
	$UQP_PY2 $DIR/run_${SUITE}.py $@ >& ${SUITE}_py2.log
	SUITE_DIRS+=("${SUITE}_2")
	cd ..
	echo "Running ${SUITE} tests in python 3..."
	mkdir -p "${SUITE}_3"; cd "${SUITE}_3"
	$UQP_PY3 $DIR/run_${SUITE}.py $@ >& ${SUITE}_py3.log
	SUITE_DIRS+=("${SUITE}_3")
	cd ..
done

echo
$UQP_PY3 $DIR/collect.py "${TEST_SUITES[@]}" -d "${SUITE_DIRS[@]}"
