#!/usr/bin/env bash

EMB_DIR=./ud2_embeddings/
DATA_DIR=./data2.2/
SRC_DIR=./

LANGS=$1 # set the languages here ( ex., LANGS = 'en pt' )

embed_paths=()
data_paths=()

for i in ${LANGS[@]};
do
    embed_paths+=("${EMB_DIR}/wiki.multi.${i}.vec")
    data_paths+=("${DATA_DIR}/${i}_train.conllu")
    data_paths+=("${DATA_DIR}/${i}_dev.conllu")
    data_paths+=("${DATA_DIR}/${i}_test.conllu")
done

MODEL_DIR_NAME=${LANGS[@]}
MODEL_DIR_NAME=${MODEL_DIR_NAME// /_}

echo "Languages: ${LANGS[*]}"
echo "Embed Paths: ${embed_paths[*]}"
echo "Data Paths: ${data_paths[*]}"
echo "Model Path: $MODEL_DIR_NAME"
echo ""

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES= python ${SRC_DIR}/vocab/build_joint_vocab.py \
    --embed_paths ${embed_paths[*]} \
 	--embed_lang_ids ${LANGS[*]} \
 	--data_paths ${data_paths[*]} \
 	--model_path ./${MODEL_DIR_NAME}/
