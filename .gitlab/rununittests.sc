#!/bin/bash

set -x

BBD=$1
#HOST=$2
UQPREPO=$BBD
#/collab/usr/gapps/uq/UQPipeline/bin/runPipeline --run-unit-tests 2>&1 | tee $BBD/unittest.log
$UQPREPO/tests/run_tests.sh unit integration  2>&1
#| tee $BBD/test.log
