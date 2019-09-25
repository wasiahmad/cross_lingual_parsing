## Cross-lingual Dependency Parsing with Unlabeled Auxiliary Languages

This repo contains the code for the CoNLL 2019 paper: "Cross-lingual Dependency Parsing with Unlabeled Auxiliary Languages". [[arxiv]](https://arxiv.org/abs/1909.09265)[[paper]]()[[bib]]()

The source code is build based upon [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2) and PyTorch-0.4.1. In addition, we use resources from this [repository](https://github.com/uclanlp/CrossLingualDepParser).

## Table of Contents
 - [Requirements](#requirements)
 - [Data Preparation](#data-preparation)
 - [Multilingual BERT](#multilingual-bert)
 - [Training and Testing](#training-and-testing)
 - [Notes](#notes)
 - [Citation](#citation)
 

### Requirements

- [python 3.6](https://repo.continuum.io/archive/)
- [pytorch 0.4.1](https://pytorch.org/get-started/previous-versions/#via-conda)
- [gensim](https://pypi.org/project/gensim/)
- [tqdm](https://pypi.org/project/tqdm/)
- [boto3](https://pypi.org/project/boto3/)
- [requests](https://pypi.org/project/requests/)


### Data Preparation

- We conduct experiments on [Universal Dependencies (UD)](https://universaldependencies.org/) v2.2 and the data can be prepared following instruction provided in the [CrossLingualDepParser](https://github.com/uclanlp/CrossLingualDepParser) repository. [[Direct Link]](https://github.com/uclanlp/CrossLingualDepParser#data-preparation)
- We use the old version of fasttext embeddings and [fastText_multilingual](https://github.com/Babylonpartners/fastText_multilingual) for alignment.
- However, for a quick start, we provide the processed dataset and multilingual word embeddings which can be downloaded by running the `setup.sh` file.


### Multilingual BERT

We used Multilingual BERT implementation from [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers) to (thanks to the authors) extract features for input sentences. [tqdm](https://pypi.org/project/tqdm/), [boto3](https://pypi.org/project/boto3/), and [requests](https://pypi.org/project/requests/) APIs are required.


### Training and Testing

There are two steps.

**Step 1.** Create the joint vocabulary for the languages we want to do experiments. Run the following commands with the language codes.

```
bash vocab/build_joint_vocab.sh LANGUAGE_CODE
```

For example, if we want to create the joint vocabulary for English and Russian language, run the following command. 
```
bash vocab/build_joint_vocab.sh 'en ru'
```

After running the command, a directory with the name `en_ru` will be created that will contain a sub-directory named as `alphabets` and a log file.

**Step 2.** To run training/testing, we need to run the following command.

```
bash run/SCRIPT_NAME.sh GPU_ID AUX_LANG_NAME DISC_TYPE TRAIN_LEVEL TRAIN_TYPE LAMBDA
```

Where,

- `SCRIPT_NAME`: {`adv_att_graph`, `adv_stptr_rnn`}
- `AUX_LANG_NAME`: {`ru`, `pt`, `ru pt`, `ru la pt` ...} [any language combination]
- `DISC_TYPE`: {`weak`, `not-so-weak`, `strong`}
- `TRAIN_LEVEL`: {`word`, `sent`}
- `TRAIN_TYPE`: {`GR`, `GAN`, `WGAN`}
- `LAMBDA`: any value between 0 and 1.

So, if we want to train/test the `Self-Attentive Graph` parser using Russian as the auxiliary language, we can run the following command.

```
bash run/adv_att_graph.sh 1,3,6 ru weak word GAN 0.01
```

#### Run the Multi-task Training

Instead of adversarial training, we can utilize auxiliary languages to train the parsers in a multi-task learning fashion. To run training/testing, we need to run the following command.

```
bash run/mtv_att_graph.sh GPU_ID AUX_LANG_NAME DISC_TYPE TRAIN_LEVEL LAMBDA
```

Where, the flag values are same as mentioned for adversarial training. We didn't perform multi-task training for the `RNN-StackPtr` parser. However, the code is provided, so we can run and check the performance by ourselves!


#### Run the Language Test

Details of the language test is provided in the paper. To perform the language test, run the following command.

```
bash run/lang_test.sh GPU_ID PRE_MODEL_DIR PRE_MODEL_NAME
```

Let's say, if we want to perform the language test for a multi-task trained `SelfAttentive-Graph` parser, we can run the following command.

```
bash run/lang_test.sh 1 en_ru MOTIV_en_ru_weak_word_d100_k5_l0_01.pt
```

Here, we perform the language test for a model which was trained on English and Russian language pair. So, we provide the directory path as `en_ru` and the model file as `MOTIV_en_ru_weak_word_d100_k5_l0_01.pt`. Please note, the model files are with the extension `.pt`.


#### Running experiments on CPU/GPU/Multi-GPU

- If `GPU_ID` is set to -1, CPU will be used.
- If `GPU_ID` is set to one specific number, only one GPU will be used.
- If `GPU_ID` is set to multiple numbers (e.g., 0,1,2), then parallel computing will be used.


#### Notable Flags

| Flag             |  Type |  Description | 
| :--- | ---: | ---: |
| use_bert         |  Boolean  | Use multilingual BERT to form the input representation |
| no_word          |  Boolean  | Use multilingual word embeddings to form the input representation |
| adv_training     |  Boolean  | Use adversarial training |
| encoder_type     |  String   | Use any of {'Transformer', 'RNN', 'SelfAttn'} |


### Notes

- We trained the SelfAttentive-Graph parser with a batch size of 80 which requires 3 GeForce 1080 GPUs (11gb) and the RNN-StackPtr parser with a batch size of 32 which requires 1 GPU. (We didn't tune the batch size)
- Training time for one iteration: [SelfAttentive-Graph] ~80 seconds [RNN-StackPtr] ~180 seconds.
- In every 5 epochs, we perform the validation check and for RNN-StackPtr parser it takes more time compared to the SelfAttentive-Graph parser.
- We didn't perform the language test for the RNN-StackPtr parser.


### Citation

If you find the resources in this repo useful, please cite our works.

```
@inproceedings{ahmad2019,
  title = {Cross-lingual Dependency Parsing with Unlabeled Auxiliary Languages},
  author = {Ahmad, Wasi Uddin and Zhang, Zhisong and Ma, Zuezhe and Chang, Kai-Wei and Peng, Nanyun},
  booktitle = {Proceedings of the 2019 Conference on Computational Natural Language Learning},
  year = {2019}
}
```
