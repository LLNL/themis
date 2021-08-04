#!/bin/bash
#SBATCH -J sleep_hello_world
#SBATCH -p %%partition%%
#SBATCH -t 1
#SBATCH -N 1
set -e

echo "hello"

srun -n1 sleep %%sleep_time%% &

wait
