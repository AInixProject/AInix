#!/usr/bin/env bash

# Run just basic rnn with small glove

BATCH_SIZE=64
REPLACE_SAMPLES=10
#TRAIN_EPOCHS=40
WORD_VEC_SIZE=300
TRAIN_STEPS=7500

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

echo "prepare glove"
cd ./OpenNMT-py/
python3 -m tools.embeddings_to_torch \
    -emb_file_both "../glove_dir/glove.840B.${WORD_VEC_SIZE}d.txt" \
    -dict_file "../expirs/exp1.vocab.pt" \
    -output_file "../data/embeddings" \
    || exit 1
cd ..

echo "Train"
data_size=$(wc -l < data_train_x.txt)
#steps_to_do=$[(TRAIN_EPOCHS*BATCH_SIZE)/REPLACE_SAMPLES/BATCH_SIZE]
echo ${steps_to_do}
CUDA_VISIBLE_DEVICES=0 python3 ./OpenNMT-py/train.py \
    -data expirs/exp1 \
    -save_model data/demo-model \
    --src_word_vec_size 64 \
    --tgt_word_vec_size 64 \
    --rnn_size 128 \
    --batch_size ${BATCH_SIZE} \
    --train_steps ${TRAIN_STEPS} \
    --report_every 50 \
    --start_decay_steps 4000 \
    --decay_steps 2000 \
    --gpu_rank 0 \
    --word_vec_size ${WORD_VEC_SIZE} \
    --pre_word_vecs_enc "data/embeddings.enc.pt" \
    --pre_word_vecs_dec "data/embeddings.dec.pt" \
    || exit 1

echo "Predict"
python3 ./OpenNMT-py/translate.py \
    -model data/demo-model_step_${TRAIN_STEPS}.pt \
    -src data_val_x.txt \
    -tgt data_val_y.txt \
    -output pred.txt \
    -replace_unk \
    -verbose \
    --replace_unk \
    || exit 1
    #--beam_size 1 \


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
