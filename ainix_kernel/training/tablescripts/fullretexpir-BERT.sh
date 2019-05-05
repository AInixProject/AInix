#!/usr/bin/env bash

FN="fullret-BERT"

rm -r runs/${FN}
mkdir -p runs/${FN}

RUNDIR=$(pwd)
cd ../../..
for i in {0..4}
do
    python3 -m ainix_kernel.training.fullret_try \
        --train_percent 80 \
        --randomize_seed \
        --nointeractive \
        --eval_replace_samples 35 \
        --encoder_name BERT \
        | tee ${RUNDIR}/runs/${FN}/run${i} \
        || exit 1
done
