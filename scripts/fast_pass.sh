#!/usr/bin/env bash
# A REDUCED pass of the full study, for when there is no GPU and no time.
# Every number it produces is a preview: fewer training devices, fewer
# epochs, fewer PPO iterations, five of the fifteen budget cells and a
# coarser oracle.  It writes to results_fast/ so it cannot be mistaken for
# the real thing.  The split, the devices, the metric and the action space
# are untouched.
set -e
cd "$(dirname "$0")/.."

export ADQ_RESULTS=results_fast
export ADQ_CHECKPOINTS=checkpoints_fast
export ADQ_DEVICE=cpu
export OMP_NUM_THREADS=16

export ADQ_TRAIN_SUBSET=250     # of 500 train devices
export ADQ_EPOCHS=20            # of 50
export ADQ_REPEATS=3            # of 4
export ADQ_ITERATIONS=100       # of 300 PPO iterations
export ADQ_CELLS="4x40,5x50,6x60,8x50,8x60"   # 5 of 15, distinct op counts
export ADQ_ORACLE_RES=8         # of 12x12

for s in run_2_train_decoder run_3_train_agent run_4_compare_arms run_5_policy_anatomy; do
    echo "############ $s ############"
    date
    python -u scripts/$s.py
done
date
echo "############ done ############"
