#!/usr/bin/env bash

BATCH_SIZE=64
REPLACE_SAMPLES=10
TRAIN_EPOCHS=40

echo "Exporting latest data"
cd ../../..
python3 -m ainix_kernel.training.export_data \
    --replace_samples ${REPLACE_SAMPLES} \
    || exit 1
mv data_train* ./ainix_kernel/training/opennmt
mv data_val* ./ainix_kernel/training/opennmt
cd ./ainix_kernel/training/opennmt

echo "Preproc data"
rm expirs/exp1*
python3 ./OpenNMT-py/preprocess.py \
  -train_src data_train_x.txt \
  -train_tgt data_train_y.txt \
  -valid_src data_val_x.txt \
  -valid_tgt data_val_y.txt \
  --save_data expirs/exp1 \
  --src_words_min_frequency 3 \
  --tgt_words_min_frequency 3 \
  || exit 1

echo "Train"
data_size=$(wc -l < data_train_x.txt)
steps_to_do=$[(TRAIN_EPOCHS*BATCH_SIZE)/REPLACE_SAMPLES/BATCH_SIZE]
echo ${steps_to_do}
CUDA_VISIBLE_DEVICES=0 python3 ./OpenNMT-py/train.py \
    -data expirs/exp1 \
    -save_model data/demo-model \
    --src_word_vec_size 64 \
    --tgt_word_vec_size 64 \
    --rnn_size 128 \
    --batch_size ${BATCH_SIZE} \
    --train_steps 7500 \
    --report_every 50 \
    --start_decay_steps 4000 \
    --decay_steps 2000 \
    --gpu_rank 0 \
    || exit 1

echo "Predict"
python3 ./OpenNMT-py/translate.py \
    -model data/demo-model_step_5000.pt \
    -src data_val_x.txt \
    -tgt data_val_y.txt \
    -output pred.txt \
    -replace_unk \
    -verbose \
    --replace_unk \
    || exit 1


cd ../../..
echo "evaling"
python3 -m ainix_kernel.training.eval_external \
    --src_xs ./ainix_kernel/training/opennmt/data_val_x.txt \
    --predictions ./ainix_kernel/training/opennmt/pred.txt \
    --tgt_ys ./ainix_kernel/training/opennmt/data_val_y.txt \
    --tgt_yids ./ainix_kernel/training/opennmt/data_val_yids.txt \
    --tokenizer_name nonascii \
    || exit 1

    #--optim adagrad \
    #--learning_rate 1 \


echo "Done."
