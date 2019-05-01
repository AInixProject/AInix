#!/usr/bin/env bash

rm -r runs/seq2seq
mkdir -p runs/seq2seq

cd ../opennmt
for i in {0..4}
do
    ./expir3.sh | tee ../tablescripts/runs/seq2seq/run${i}
done
