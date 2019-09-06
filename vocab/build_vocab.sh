#!/usr/bin/env bash

EMB_DIR=./embeddings/
DATA_DIR=./data2.2/
SRC_DIR=./
OUT_DIR=./ud2_embeddings/

function run_lang () {

LANG=$1
embed_path=${EMB_DIR}/wiki.multi.${LANG}.vec

data0=${DATA_DIR}/${LANG}_train.conllu
data1=${DATA_DIR}/${LANG}_dev.conllu
data2=${DATA_DIR}/${LANG}_test.conllu

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES= python ${SRC_DIR}/vocab/build_vocab.py \
    --word_embedding word2vec \
    --embed_lang_id $LANG \
    --word_paths $embed_path \
	--train $data0 \
	--extra $data1 $data2 \
	--model_path ${OUT_DIR}/

TARGET_VEC_FILE=wiki.multi.${LANG}.vec
cp ${OUT_DIR}/${LANG}_alphabets/${TARGET_VEC_FILE} ${OUT_DIR}/
rm -r ${OUT_DIR}/${LANG}_alphabets/
rm ${OUT_DIR}/vocab.log.txt

}

for cur_lang in ar bg ca cs da de en es et "fi" fr he hi hr id it ko la lv nl no pl pt ro ru sk sl sv uk ja zh;
do
    run_lang $cur_lang;
done
