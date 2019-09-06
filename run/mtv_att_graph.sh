#!/usr/bin/env bash

RGPU=$1
SEED=1234

EMB_DIR=./ud2_embeddings/
DATA_DIR=./data2.2/
SRC_DIR=./

SRC_LANG=en
AUX_LANG=$2 # set the languages here

MODEL_DIR="$SRC_LANG ${AUX_LANG[@]}"
MODEL_DIR=${MODEL_DIR// /_}

DISC_TYPE=$3 # options: 'weak', 'not-so-weak', 'strong'
TRAIN_LEVEL=$4 # options: 'word', 'sent'

N_CRITIC=5
DELAY=100
LAMBDA=$5

FILENMAE=MOTIV_${MODEL_DIR}_${DISC_TYPE}_${TRAIN_LEVEL}_d${DELAY}_k${N_CRITIC}_l${LAMBDA//./_}

MODEL_DIR=./${MODEL_DIR}/
MODEL_NAME=${FILENMAE}.pt

LOG_FILENAME=${MODEL_DIR}${FILENMAE}.log
ANAL_FILENAME=${MODEL_DIR}${FILENMAE}.anal
CSV_FILENAME=${MODEL_DIR}${FILENMAE}.csv

echo "Current seed is $SEED"
echo $FILENMAE

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python ${SRC_DIR}/examples/GraphParser.py \
--use_bert True \
--no_word True \
--motivate True \
--n_critic $N_CRITIC \
--discriminator $DISC_TYPE \
--train_level $TRAIN_LEVEL \
--delay $DELAY \
--lambdaG $LAMBDA \
--encoder_type 'Transformer' \
--partitioned False \
--partition_type 'lexical-delexical' \
--mode FastLSTM \
--hidden_size 300 \
--num_layers 6 \
--use_all_encoder_layers False \
--d_k 64 \
--d_v 64 \
--arc_space 512 \
--type_space 128 \
--opt adam \
--decay_rate 0.75 \
--epsilon 1e-4 \
--gamma 0.0 \
--clip 5.0 \
--schedule 20 \
--double_schedule_decay 5 \
--use_warmup_schedule True \
--check_dev 2 \
--unk_replace 0.5 \
--freeze \
--pos \
--pool_type weight \
--num_head 8 \
--word_embedding word2vec \
--word_path ${MODEL_DIR}/alphabets/joint_embed.vec \
--char_embedding random \
--punctuation 'PUNCT' 'SYM' \
--data_dir $DATA_DIR \
--src_lang $SRC_LANG \
--aux_lang ${AUX_LANG[*]} \
--vocab_path $MODEL_DIR \
--model_path $MODEL_DIR \
--model_name $MODEL_NAME \
--p_in 0.2 \
--p_out 0.2 \
--p_rnn 0.2 0.1 0.2 \
--learning_rate 0.0001 \
--num_epochs 400 \
--trans_hid_size 512 \
--pos_dim 50 \
--char_dim 50 \
--num_filters 50 \
--input_concat_embeds \
--input_concat_position \
--position_dim 0 \
--enc_clip_dist 10 \
--batch_size 80 \
--seed $SEED \
|& tee $LOG_FILENAME


RGPU=$RGPU bash ${SRC_DIR}/run/analyze.sh test biaffine $MODEL_DIR $MODEL_NAME \
|& tee $ANAL_FILENAME

cat $ANAL_FILENAME | grep -E "Running with lang|test Wo Punct" |& tee ${FILENMAE}.tmp
python ${SRC_DIR}/examples/extract_score.py ${FILENMAE}.tmp $CSV_FILENAME
rm ${FILENMAE}.tmp
echo "Finished testing model: "${FILENMAE}
