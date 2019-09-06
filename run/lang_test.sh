#!/usr/bin/env bash

RGPU=$1

EMB_DIR=./ud2_embeddings/
DATA_DIR=./data2.2/
SRC_DIR=./

LANGS=('en fr de es ru pt la') # set the languages here

PRE_MODEL_DIR=$2 # ex., en_ru
PRE_MODEL_NAME=$3 # ex., 'MOTIV_en_ru_weak_word_d100_k5_l0_01.pt'

MODEL_DIR="${LANGS[@]}"
MODEL_DIR=${MODEL_DIR// /_}

FILENMAE=LaClass_${MODEL_DIR}

PRE_MODEL_DIR=./${PRE_MODEL_DIR}/
MODEL_DIR=./${MODEL_DIR}/

if [[ ! -d $MODEL_DIR ]]; then
    mkdir $MODEL_DIR
fi

MODEL_NAME=${FILENMAE}.pt
LOG_FILENAME=${MODEL_DIR}${FILENMAE}.log

echo $FILENMAE

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python ${SRC_DIR}/examples/LanguageTest.py \
--parser biaffine \
--nclass 7 \
--langs ${LANGS[*]} \
--data_dir $DATA_DIR \
--model_path $MODEL_DIR \
--model_name $MODEL_NAME \
--pre_model_path $PRE_MODEL_DIR \
--pre_model_name $PRE_MODEL_NAME \
--gpu \
--embed_dir $EMB_DIR \
--num_epochs 50 \
--batch_size 128 \
--train_level 'word' \
|& tee $LOG_FILENAME

