#!/usr/bin/env bash

# Make sure in right dir before deleting existing checkpoints
#cd .. || exit 0
#cd ./allen || exit 0
#rm -r checkpoints/*
#
## Copy over data. Note this doesn't run export data in the opennmt dir
#python3 totsv.py

#python3 basseq2seq.py

# Basic copy seq2seq
cd ../../..
python3 -m ainix_kernel.training.eval_external \
    --src_xs ./ainix_kernel/training/opennmt/data_val_x.txt \
    --predictions ./ainix_kernel/training/allen/pred.txt \
    --tgt_ys ./ainix_kernel/training/opennmt/data_val_y.txt \
    --tgt_yids ./ainix_kernel/training/opennmt/data_val_yids.txt \
    --tokenizer_name nonascii \
    || exit 1
