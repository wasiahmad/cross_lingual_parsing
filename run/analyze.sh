#!/usr/bin/env bash

MODEL_DIR=$3
MODEL_NAME=$4
#EMB_DIR=../embeddings/
EMB_DIR=./ud2_embeddings/
DATA_DIR=./data2.2/
SRC_DIR=./

#
function run_lang () {

echo "======================"
echo "Running with lang = $1_$2_$3"

cur_lang=$1
which_set=$2
which_model=$3

# try them both, will fail on one

if [ "$which_model" == "biaffine" ]; then

echo PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python ${SRC_DIR}/examples/analyze.py --parser biaffine --ordered --gpu \
--punctuation 'PUNCT' 'SYM' --out_filename analyzer.$cur_lang.$which_set.$MODEL_NAME.out --model_name $MODEL_NAME \
--test "${DATA_DIR}/${cur_lang}_${which_set}.conllu" --model_path $MODEL_DIR --extra_embed "${EMB_DIR}/wiki.multi.${cur_lang}.vec"

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python ${SRC_DIR}/examples/analyze.py --parser biaffine --ordered --gpu \
--punctuation 'PUNCT' 'SYM' --out_filename analyzer.$cur_lang.$which_set.$MODEL_NAME.out --model_name $MODEL_NAME \
--test "${DATA_DIR}/${cur_lang}_${which_set}.conllu" --model_path $MODEL_DIR --extra_embed "${EMB_DIR}/wiki.multi.${cur_lang}.vec"

elif [ "$which_model" == "stackptr" ]; then

echo PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python ${SRC_DIR}/examples/analyze.py --parser stackptr --beam 5 --ordered --gpu \
--punctuation 'PUNCT' 'SYM' --out_filename analyzer.$cur_lang.$which_set.$MODEL_NAME.out --model_name $MODEL_NAME \
--test "${DATA_DIR}/${cur_lang}_${which_set}.conllu" --model_path $MODEL_DIR --extra_embed "${EMB_DIR}/wiki.multi.${cur_lang}.vec"

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python ${SRC_DIR}/examples/analyze.py --parser stackptr --beam 5 --ordered --gpu \
--punctuation 'PUNCT' 'SYM' --out_filename analyzer.$cur_lang.$which_set.$MODEL_NAME.out --model_name $MODEL_NAME \
--test "${DATA_DIR}/${cur_lang}_${which_set}.conllu" --model_path $MODEL_DIR --extra_embed "${EMB_DIR}/wiki.multi.${cur_lang}.vec"

fi

}

# =====

echo "Run them all with $1_$2"

# running with which dev, which set?
#for cur_lang in ar bg ca cs da de en es et "fi" fr he hi hr id it ko la lv nl no pl pt ro ru sk sl sv uk ja zh;
for cur_lang in ar he id lv da de en nl no sv hi la ca es fr it pt ro bg cs hr pl ru sk sl uk ko et "fi";
do
    run_lang $cur_lang $1 $2;
done

# RGPU=0 bash ../src/examples/run_more/run_analyze.sh param1 param2 |& tee log_test
# param1 = [train/dev/test], param2 = [biaffine/stackptr]
# see the results?
# -> cat log_test | grep -E "python2|Running with lang|uas|Error"
# -> cat log_test | grep -E "Running with lang|uas"
# -> cat log_test | grep -E "Running with lang|test Wo Punct"
# -> cat log_test | python3 print_log_test.py
