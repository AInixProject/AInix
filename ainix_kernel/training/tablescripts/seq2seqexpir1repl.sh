#!/usr/bin/env bash

rm -r runs/seq2seq-1repl
mkdir -p runs/seq2seq-1repl

cd ../opennmt
for i in {0..4}
do
    ./expir3.sh -r 1 -s 3000 | tee ../tablescripts/runs/seq2seq-1repl/run${i}
done
