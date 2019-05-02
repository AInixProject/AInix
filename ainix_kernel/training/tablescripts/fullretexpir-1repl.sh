#!/usr/bin/env bash

#rm -r runs/fullret-1repl || exit 0
mkdir -p runs/fullret-1repl

RUNDIR=$(pwd)
cd ../../..
for i in {0..4}
do
    python3 -m ainix_kernel.training.fullret_try \
        --train_percent 80 \
        --randomize_seed \
        --nointeractive \
        --eval_replace_samples 35 \
        --replace_samples 1 \
        | tee ${RUNDIR}/runs/fullret-1repl/run${i} \
        || exit 1
done
