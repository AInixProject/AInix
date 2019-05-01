#!/usr/bin/env bash

rm -r runs/gram
mkdir -p runs/gram

RUNDIR=$(pwd)
cd ../../..
for i in {0..4}
do
    python3 -m ainix_kernel.training.trainer \
        --train_percent 80 \
        --randomize_seed \
        --quiet_dump \
        --eval_replace_samples 35 \
        | tee ${RUNDIR}/runs/gram/run${i} \
        || exit 1
done
