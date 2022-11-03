#!/bin/bash
#BSUB -J sleep_hello_world
#BSUB -q pdebug
#BSUB -W 5
#BSUB -nnodes 1
set -e

echo "hello"

lrun -n1 sleep %%sleep_time%% &

wait
