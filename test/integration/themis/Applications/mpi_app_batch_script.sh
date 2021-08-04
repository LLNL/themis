#!/bin/bash
#SBATCH -J sleep_hello_world
#SBATCH -p pdebug
#SBATCH -t 1
#SBATCH -N 1
set -e

echo "BATCH SCRIPT RUNNING" >> batch_script.log 2>&1

%%themis_launch%% %%python%% $@ >> batch_script.log 2>&1

echo "BATCH SCRIPT COMPLETE" >> batch_script.log 2>&1
