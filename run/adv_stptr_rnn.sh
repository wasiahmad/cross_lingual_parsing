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
TRAIN_TYPE=$5 # options: 'GR', 'GAN', 'WGAN'

N_CRITIC=5
DELAY=100
LAMBDA=$6

FILENMAE=SR_${MODEL_DIR}_${DISC_TYPE}_${TRAIN_LEVEL}_${TRAIN_TYPE}_d${DELAY}_k${N_CRITIC}_l${LAMBDA//./_}

MODEL_DIR=./${MODEL_DIR}/
MODEL_NAME=${FILENMAE}.pt

LOG_FILENAME=${MODEL_DIR}${FILENMAE}.log
ANAL_FILENAME=${MODEL_DIR}${FILENMAE}.anal
CSV_FILENAME=${MODEL_DIR}${FILENMAE}.csv

echo "Current seed is $SEED"
echo $FILENMAE

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python ${SRC_DIR}/examples/StackPointerParser.py \
--use_bert True \
--no_word True \
--adv_training True \
--n_critic $N_CRITIC \
--discriminator $DISC_TYPE \
--train_level $TRAIN_LEVEL \
--train_type $TRAIN_TYPE \
--delay $DELAY \
--lambdaG $LAMBDA \
--encoder_type 'RNN' \
--mode FastLSTM \
--decoder_input_size 256 \
--hidden_size 300 \
--encoder_layers 3 \
--use_all_encoder_layers False \
--decoder_layers 1 \
--arc_space 512 \
--type_space 128 \
--opt adam \
--decay_rate 0.75 \
--epsilon 1e-4 \
--coverage 0.0 \
--gamma 0.0 \
--clip 5.0 \
--schedule 20 \
--double_schedule_decay 5 \
--check_dev 5 \
--unk_replace 0.5 \
--label_smooth 1.0 \
--beam 1 \
--freeze \
--pos \
--pool_type weight \
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
--p_in 0.33 \
--p_out 0.33 \
--p_rnn 0.33 0.33 \
--learning_rate 0.001 \
--num_epochs 400 \
--pos_dim 50 \
--char_dim 50 \
--num_filters 50 \
--input_concat_embeds \
--input_concat_position \
--prior_order left2right \
--grandPar \
--batch_size 80 \
--seed $SEED \
|& tee $LOG_FILENAME


RGPU=$RGPU bash ${SRC_DIR}/run/analyze.sh test stackptr $MODEL_DIR $MODEL_NAME \
|& tee $ANAL_FILENAME

cat $ANAL_FILENAME | grep -E "Running with lang|test Wo Punct" |& tee ${FILENMAE}.tmp
python ${SRC_DIR}/examples/extract_score.py ${FILENMAE}.tmp $CSV_FILENAME
rm ${FILENMAE}.tmp
echo "Finished testing model: "${FILENMAE}

