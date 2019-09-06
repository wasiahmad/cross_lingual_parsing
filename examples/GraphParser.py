from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys
import os

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid
import json
import random
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from neuronlp2.io import get_logger, conllx_data
from neuronlp2.models import BiRecurrentConvBiAffine, Adversarial, Motivator
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser
from neuronlp2.nn.utils import freeze_embedding, generate_optimizer

from neuronlp2.io_multi import guess_language_id

uid = uuid.uuid4().hex[:6]


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def main():
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.register('type', 'bool', str2bool)

    args_parser.add_argument('--seed', type=int, default=1234, help='random seed for reproducibility')
    args_parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU', 'FastLSTM'],
                             help='architecture of rnn', required=True)
    args_parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    args_parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    args_parser.add_argument('--arc_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--type_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of encoder.')
    args_parser.add_argument('--num_filters', type=int, default=50, help='Number of filters in CNN')
    args_parser.add_argument('--pos', action='store_true', help='use part-of-speech embedding.')
    args_parser.add_argument('--char', action='store_true', help='use character embedding and CNN.')
    args_parser.add_argument('--pos_dim', type=int, default=50, help='Dimension of POS embeddings')
    args_parser.add_argument('--char_dim', type=int, default=50, help='Dimension of Character embeddings')
    args_parser.add_argument('--opt', choices=['adam', 'sgd', 'adamax'], help='optimization algorithm')
    args_parser.add_argument('--objective', choices=['cross_entropy', 'crf'], default='cross_entropy',
                             help='objective function of training procedure.')
    args_parser.add_argument('--decode', choices=['mst', 'greedy'], default='mst', help='decoding algorithm')
    args_parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    # args_parser.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
    args_parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    args_parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    args_parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--p_rnn', nargs='+', type=float, required=True, help='dropout rate for RNN')
    args_parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    args_parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    # args_parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_parser.add_argument('--unk_replace', type=float, default=0.,
                             help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--word_embedding', choices=['word2vec', 'glove', 'senna', 'sskip', 'polyglot'],
                             help='Embedding for words', required=True)
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters',
                             required=True)
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--data_dir', help='Data directory path')
    args_parser.add_argument('--src_lang', required=True, help='Src language to train dependency parsing model')
    args_parser.add_argument('--aux_lang', nargs='+', help='Language names for adversarial training')
    args_parser.add_argument('--vocab_path', help='path for prebuilt alphabets.', default=None)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--model_name', help='name for saving model file.', required=True)
    #
    args_parser.add_argument('--attn_on_rnn', action='store_true', help='use self-attention on top of context RNN.')
    args_parser.add_argument('--no_word', type='bool', default=False, help='do not use word embedding.')
    args_parser.add_argument('--use_bert', type='bool', default=False, help='use multilingual BERT.')
    #
    # lrate schedule with warmup in the first iter.
    args_parser.add_argument('--use_warmup_schedule', type='bool', default=False, help="Use warmup lrate schedule.")
    args_parser.add_argument('--decay_rate', type=float, default=0.75, help='Decay rate of learning rate')
    args_parser.add_argument('--max_decay', type=int, default=9, help='Number of decays before stop')
    args_parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_parser.add_argument('--double_schedule_decay', type=int, default=5, help='Number of decays to double schedule')
    args_parser.add_argument('--check_dev', type=int, default=5,
                             help='Check development performance in every n\'th iteration')
    # encoder selection
    args_parser.add_argument('--encoder_type', choices=['Transformer', 'RNN', 'SelfAttn'],
                             default='RNN', help='do not use context RNN.')
    args_parser.add_argument('--pool_type', default='mean', choices=['max', 'mean', 'weight'],
                             help='pool type to form fixed length vector from word embeddings')
    # Tansformer encoder
    args_parser.add_argument('--trans_hid_size', type=int, default=1024,
                             help='#hidden units in point-wise feed-forward in transformer')
    args_parser.add_argument('--d_k', type=int, default=64, help='d_k for multi-head-attention in transformer encoder')
    args_parser.add_argument('--d_v', type=int, default=64, help='d_v for multi-head-attention in transformer encoder')
    args_parser.add_argument('--num_head', type=int, default=8, help='Value of h in multi-head attention')
    args_parser.add_argument('--use_all_encoder_layers', type='bool', default=False,
                             help='Use a weighted representations of all encoder layers')
    # - positional
    args_parser.add_argument('--enc_use_neg_dist', action='store_true',
                             help="Use negative distance for enc's relational-distance embedding.")
    args_parser.add_argument('--enc_clip_dist', type=int, default=0,
                             help="The clipping distance for relative position features.")
    args_parser.add_argument('--position_dim', type=int, default=50, help='Dimension of Position embeddings.')
    args_parser.add_argument('--position_embed_num', type=int, default=200,
                             help='Minimum value of position embedding num, which usually is max-sent-length.')
    args_parser.add_argument('--train_position', action='store_true', help='train positional encoding for transformer.')

    args_parser.add_argument('--input_concat_embeds', action='store_true',
                             help="Concat input embeddings, otherwise add.")
    args_parser.add_argument('--input_concat_position', action='store_true',
                             help="Concat position embeddings, otherwise add.")
    args_parser.add_argument('--partitioned', type='bool', default=False,
                             help="Partition the content and positional attention for multi-head attention.")
    args_parser.add_argument('--partition_type', choices=['content-position', 'lexical-delexical'],
                             default='content-position', help="How to apply partition in the self-attention.")
    #
    args_parser.add_argument('--train_len_thresh', type=int, default=100,
                             help='In training, discard sentences longer than this.')

    #
    # regarding adversarial training
    args_parser.add_argument('--pre_model_path', type=str, default=None,
                             help='Path of the pretrained model.')
    args_parser.add_argument('--pre_model_name', type=str, default=None,
                             help='Name of the pretrained model.')
    args_parser.add_argument('--adv_training', type='bool', default=False,
                             help='Use adversarial training.')
    args_parser.add_argument('--lambdaG', type=float, default=0.001,
                             help='Scaling parameter to control generator loss.')
    args_parser.add_argument('--discriminator', choices=['weak', 'not-so-weak', 'strong'],
                             default='weak', help='architecture of the discriminator')
    args_parser.add_argument('--delay', type=int, default=0,
                             help='Number of epochs to be run first for the source task')
    args_parser.add_argument('--n_critic', type=int, default=5,
                             help='Number of training steps for discriminator per iter')
    args_parser.add_argument('--clip_disc', type=float, default=5.0,
                             help='Lower and upper clip value for disc. weights')
    args_parser.add_argument('--debug', type='bool', default=False,
                             help='Use debug portion of the training data')
    args_parser.add_argument('--train_level', type=str, default='word',
                             choices=['word', 'sent'],
                             help='Use X-level adversarial training')
    args_parser.add_argument('--train_type', type=str, default='GAN',
                             choices=['GR', 'GAN', 'WGAN'],
                             help='Type of adversarial training')
    #
    # regarding motivational training
    args_parser.add_argument('--motivate', type='bool', default=False,
                             help='This is opposite of the adversarial training')

    #
    args = args_parser.parse_args()

    # fix data-prepare seed
    random.seed(1234)
    np.random.seed(1234)
    # model's seed
    torch.manual_seed(args.seed)

    # if output directory doesn't exist, create it
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    logger = get_logger("GraphParser")

    logger.info('\ncommand-line params : {0}\n'.format(sys.argv[1:]))
    logger.info('{0}\n'.format(args))

    logger.info("Visible GPUs: %s", str(os.environ["CUDA_VISIBLE_DEVICES"]))
    args.parallel = False
    if torch.cuda.device_count() > 1:
        args.parallel = True

    mode = args.mode
    obj = args.objective
    decoding = args.decode

    train_path = args.data_dir + args.src_lang + "_train.debug.1_10.conllu" \
        if args.debug else args.data_dir + args.src_lang + '_train.conllu'
    dev_path = args.data_dir + args.src_lang + "_dev.conllu"
    test_path = args.data_dir + args.src_lang + "_test.conllu"

    #
    vocab_path = args.vocab_path if args.vocab_path is not None else args.model_path
    model_path = args.model_path
    model_name = args.model_name

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    arc_space = args.arc_space
    type_space = args.type_space
    num_layers = args.num_layers
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    opt = args.opt
    momentum = 0.9
    betas = (0.9, 0.9)
    eps = args.epsilon
    decay_rate = args.decay_rate
    clip = args.clip
    gamma = args.gamma
    schedule = args.schedule
    p_rnn = tuple(args.p_rnn)
    p_in = args.p_in
    p_out = args.p_out
    unk_replace = args.unk_replace
    punctuation = args.punctuation

    freeze = args.freeze
    use_word_emb = not args.no_word
    word_embedding = args.word_embedding
    word_path = args.word_path

    use_char = args.char
    char_embedding = args.char_embedding
    char_path = args.char_path

    attn_on_rnn = args.attn_on_rnn
    encoder_type = args.encoder_type
    if attn_on_rnn:
        assert encoder_type == 'RNN'

    t_types = (args.adv_training, args.motivate)
    t_count = sum(1 for tt in t_types if tt)
    if t_count > 1:
        assert False, "Only one of: adv_training or motivate can be true"

    # ------------------- Loading/initializing embeddings -------------------- #

    use_pos = args.pos
    pos_dim = args.pos_dim
    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)
    char_dict = None
    char_dim = args.char_dim
    if char_embedding != 'random':
        char_dict, char_dim = utils.load_embedding_dict(char_embedding, char_path)

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(vocab_path, 'alphabets/')
    model_name = os.path.join(model_path, model_name)

    # TODO (WARNING): must build vocabs previously
    assert os.path.isdir(alphabet_path), "should have build vocabs previously"
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_sent_length = conllx_data.create_alphabets(
        alphabet_path, train_path, data_paths=[dev_path, test_path], max_vocabulary_size=50000, embedd_dict=word_dict)
    max_sent_length = max(max_sent_length, args.position_embed_num)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    # ------------------------------------------------------------------------- #
    # --------------------- Loading/building the model ------------------------ #

    logger.info("Reading Data")
    use_gpu = torch.cuda.is_available()

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(
            -scale, scale, [1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale,
                                                                                                        [1,
                                                                                                         word_dim]).astype(
                    np.float32)
                oov += 1
            table[index, :] = embedding
        print('word OOV: %d' % oov)
        return torch.from_numpy(table)

    def construct_char_embedding_table():
        if char_dict is None:
            return None

        scale = np.sqrt(3.0 / char_dim)
        table = np.empty([num_chars, char_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
        oov = 0
        for char, index, in char_alphabet.items():
            if char in char_dict:
                embedding = char_dict[char]
            else:
                embedding = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('character OOV: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table() if use_word_emb else None
    char_table = construct_char_embedding_table() if use_char else None

    def load_model_arguments_from_json():
        arguments = json.load(open(pre_model_path, 'r'))
        return arguments['args'], arguments['kwargs']

    window = 3
    if obj == 'cross_entropy':
        if args.pre_model_path and args.pre_model_name:
            pre_model_name = os.path.join(args.pre_model_path, args.pre_model_name)
            pre_model_path = pre_model_name + '.arg.json'
            model_args, kwargs = load_model_arguments_from_json()

            network = BiRecurrentConvBiAffine(use_gpu=use_gpu, *model_args, **kwargs)
            network.load_state_dict(torch.load(pre_model_name))
            logger.info("Model reloaded from %s" % pre_model_path)

            # Adjust the word embedding layer
            if network.embedder.word_embedd is not None:
                network.embedder.word_embedd = nn.Embedding(num_words, word_dim, _weight=word_table)

        else:
            network = BiRecurrentConvBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters,
                                              window, mode, hidden_size, num_layers, num_types, arc_space, type_space,
                                              embedd_word=word_table, embedd_char=char_table, p_in=p_in, p_out=p_out,
                                              p_rnn=p_rnn, biaffine=True, pos=use_pos, char=use_char,
                                              train_position=args.train_position, encoder_type=encoder_type,
                                              trans_hid_size=args.trans_hid_size, d_k=args.d_k, d_v=args.d_v,
                                              num_head=args.num_head, enc_use_neg_dist=args.enc_use_neg_dist,
                                              enc_clip_dist=args.enc_clip_dist, position_dim=args.position_dim,
                                              max_sent_length=max_sent_length, use_gpu=use_gpu,
                                              use_word_emb=use_word_emb, input_concat_embeds=args.input_concat_embeds,
                                              input_concat_position=args.input_concat_position,
                                              attn_on_rnn=attn_on_rnn, partitioned=args.partitioned,
                                              partition_type=args.partition_type,
                                              use_all_encoder_layers=args.use_all_encoder_layers,
                                              use_bert=args.use_bert)

    elif obj == 'crf':
        raise NotImplementedError
    else:
        raise RuntimeError('Unknown objective: %s' % obj)

    # ------------------------------------------------------------------------- #
    # --------------------- Loading data -------------------------------------- #

    train_data = dict()
    dev_data = dict()
    test_data = dict()
    num_data = dict()
    lang_ids = dict()
    reverse_lang_ids = dict()

    # ===== the reading =============================================
    def _read_one(path, is_train):
        lang_id = guess_language_id(path)
        logger.info("Reading: guess that the language of file %s is %s." % (path, lang_id))
        one_data = conllx_data.read_data_to_variable(path, word_alphabet, char_alphabet, pos_alphabet,
                                                     type_alphabet, use_gpu=False,
                                                     volatile=(not is_train), symbolic_root=True,
                                                     lang_id=lang_id, use_bert=args.use_bert,
                                                     len_thresh=(args.train_len_thresh if is_train else 100000))
        return one_data

    data_train = _read_one(train_path, True)
    train_data[args.src_lang] = data_train
    num_data[args.src_lang] = sum(data_train[1])
    lang_ids[args.src_lang] = len(lang_ids)
    reverse_lang_ids[lang_ids[args.src_lang]] = args.src_lang

    data_dev = _read_one(dev_path, False)
    data_test = _read_one(test_path, False)
    dev_data[args.src_lang] = data_dev
    test_data[args.src_lang] = data_test

    # ===============================================================

    # ===== reading data for adversarial training ===================
    if t_count > 0:
        for language in args.aux_lang:
            aux_train_path = args.data_dir + language + "_train.debug.1_10.conllu" \
                if args.debug else args.data_dir + language + '_train.conllu'
            aux_train_data = _read_one(aux_train_path, True)
            num_data[language] = sum(aux_train_data[1])
            train_data[language] = aux_train_data
            lang_ids[language] = len(lang_ids)
            reverse_lang_ids[lang_ids[language]] = language
    # ===============================================================

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    def save_args():
        arg_path = model_name + '.arg.json'
        arguments = [word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, window,
                     mode, hidden_size, num_layers, num_types, arc_space, type_space]
        kwargs = {
            'p_in': p_in, 'p_out': p_out, 'p_rnn': p_rnn, 'biaffine': True, 'pos': use_pos, 'char': use_char,
            'train_position': args.train_position, 'encoder_type': args.encoder_type,
            'trans_hid_size': args.trans_hid_size, 'd_k': args.d_k, 'd_v': args.d_v,
            'num_head': args.num_head, 'enc_use_neg_dist': args.enc_use_neg_dist, 'enc_clip_dist': args.enc_clip_dist,
            'position_dim': args.position_dim, 'max_sent_length': max_sent_length, 'use_word_emb': use_word_emb,
            'input_concat_embeds': args.input_concat_embeds, 'input_concat_position': args.input_concat_position,
            'attn_on_rnn': attn_on_rnn, 'partitioned': args.partitioned, 'partition_type': args.partition_type,
            'use_all_encoder_layers': args.use_all_encoder_layers, 'use_bert': args.use_bert
        }
        json.dump({'args': arguments, 'kwargs': kwargs}, open(arg_path, 'w'), indent=4)

    if use_word_emb and freeze:
        freeze_embedding(network.embedder.word_embedd)

    if args.parallel:
        network = torch.nn.DataParallel(network)

    if use_gpu:
        network = network.cuda()

    save_args()

    param_dict = {}
    encoder = network.module.encoder if args.parallel else network.encoder
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            param_dict[name] = np.prod(param.size())

    total_params = np.sum(list(param_dict.values()))
    logger.info('Total Encoder Parameters = %d' % total_params)

    # ------------------------------------------------------------------------- #

    # =============================================
    if args.adv_training:
        disc_feat_size = network.module.encoder.output_dim if args.parallel else network.encoder.output_dim
        reverse_grad = args.train_type == 'GR'
        nclass = len(lang_ids) if args.train_type == 'GR' else 1

        kwargs = {'input_size': disc_feat_size, 'disc_type': args.discriminator,
                  'train_level': args.train_level, 'train_type': args.train_type,
                  'reverse_grad': reverse_grad, 'soft_label': True,
                  'nclass': nclass, 'scale': args.lambdaG, 'use_gpu': use_gpu,
                  'opt': 'adam', 'lr': 0.001, 'betas': (0.9, 0.999), 'gamma': 0, 'eps': 1e-8,
                  'momentum': 0, 'clip_disc': args.clip_disc}
        AdvAgent = Adversarial(**kwargs)
        if use_gpu:
            AdvAgent.cuda()

    elif args.motivate:
        disc_feat_size = network.module.encoder.output_dim if args.parallel else network.encoder.output_dim
        nclass = len(lang_ids)

        kwargs = {'input_size': disc_feat_size, 'disc_type': args.discriminator,
                  'train_level': args.train_level, 'nclass': nclass,
                  'scale': args.lambdaG, 'use_gpu': use_gpu, 'opt': 'adam',
                  'lr': 0.001, 'betas': (0.9, 0.999), 'gamma': 0, 'eps': 1e-8,
                  'momentum': 0, 'clip_disc': args.clip_disc}
        MtvAgent = Motivator(**kwargs)
        if use_gpu:
            MtvAgent.cuda()

    # =============================================

    # --------------------- Initializing the optimizer ------------------------ #

    lr = learning_rate
    optim = generate_optimizer(opt, lr, network.parameters(), betas, gamma, eps, momentum)
    opt_info = 'opt: %s, ' % opt
    if opt == 'adam':
        opt_info += 'betas=%s, eps=%.1e' % (betas, eps)
    elif opt == 'sgd':
        opt_info += 'momentum=%.2f' % momentum
    elif opt == 'adamax':
        opt_info += 'betas=%s, eps=%.1e' % (betas, eps)

    # =============================================

    total_data = min(num_data.values())

    word_status = 'frozen' if freeze else 'fine tune'
    char_status = 'enabled' if use_char else 'disabled'
    pos_status = 'enabled' if use_pos else 'disabled'
    logger.info("Embedding dim: word=%d (%s), char=%d (%s), pos=%d (%s)" % (
        word_dim, word_status, char_dim, char_status, pos_dim, pos_status))
    logger.info("CNN: filter=%d, kernel=%d" % (num_filters, window))
    logger.info("RNN: %s, num_layer=%d, hidden=%d, arc_space=%d, type_space=%d" % (
        mode, num_layers, hidden_size, arc_space, type_space))
    logger.info("train: obj: %s, l2: %f, (#data: %d, batch: %d, clip: %.2f, unk replace: %.2f)" % (
        obj, gamma, total_data, batch_size, clip, unk_replace))
    logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (p_in, p_out, p_rnn))
    logger.info("decoding algorithm: %s" % decoding)
    logger.info(opt_info)

    # ------------------------------------------------------------------------- #
    # --------------------- Form the mini-batches ----------------------------- #
    num_batches = total_data // batch_size + 1
    aux_lang = []
    if t_count > 0:
        for language in args.aux_lang:
            aux_lang.extend([language] * num_data[language])

        assert num_data[args.src_lang] <= len(aux_lang)
    # ------------------------------------------------------------------------- #

    dev_ucorrect = 0.0
    dev_lcorrect = 0.0
    dev_ucomlpete_match = 0.0
    dev_lcomplete_match = 0.0

    dev_ucorrect_nopunc = 0.0
    dev_lcorrect_nopunc = 0.0
    dev_ucomlpete_match_nopunc = 0.0
    dev_lcomplete_match_nopunc = 0.0
    dev_root_correct = 0.0

    best_epoch = 0

    if decoding == 'greedy':
        decode = network.module.decode if args.parallel else network.decode
    elif decoding == 'mst':
        decode = network.module.decode_mst if args.parallel else network.decode_mst
    else:
        raise ValueError('Unknown decoding algorithm: %s' % decoding)

    patient = 0
    decay = 0
    max_decay = args.max_decay
    double_schedule_decay = args.double_schedule_decay

    # lrate schedule
    step_num = 0
    use_warmup_schedule = args.use_warmup_schedule

    if use_warmup_schedule:
        logger.info("Use warmup lrate for the first epoch, from 0 up to %s." % (lr,))

    skip_adv_tuning = 0
    loss_fn = network.module.loss if args.parallel else network.loss
    for epoch in range(1, num_epochs + 1):
        print(
            'Epoch %d (%s, optim: %s, learning rate=%.6f, eps=%.1e, decay rate=%.2f (schedule=%d, patient=%d, decay=%d)): ' % (
                epoch, mode, opt, lr, eps, decay_rate, schedule, patient, decay))
        train_err = 0.
        train_err_arc = 0.
        train_err_type = 0.
        train_total = 0.
        start_time = time.time()
        num_back = 0

        skip_adv_tuning += 1
        loss_d_real, loss_d_fake = [], []
        acc_d_real, acc_d_fake, = [], []
        gen_loss, parsing_loss = [], []
        disent_loss = []

        if t_count > 0 and skip_adv_tuning > args.delay:
            batch_size = args.batch_size // 2
            num_batches = total_data // batch_size + 1

        # ---------------------- Sample the mini-batches -------------------------- #
        if t_count > 0:
            sampled_aux_lang = random.sample(aux_lang, num_batches)
            lang_in_batch = [(args.src_lang, sampled_aux_lang[k]) for k in range(num_batches)]
        else:
            lang_in_batch = [(args.src_lang, None) for _ in range(num_batches)]
        assert len(lang_in_batch) == num_batches
        # ------------------------------------------------------------------------- #

        network.train()
        warmup_factor = (lr + 0.) / num_batches
        for batch in range(1, num_batches + 1):
            update_generator = True
            update_discriminator = False

            # lrate schedule (before each step)
            step_num += 1
            if use_warmup_schedule and epoch <= 1:
                cur_lrate = warmup_factor * step_num
                # set lr
                for param_group in optim.param_groups:
                    param_group['lr'] = cur_lrate

            # considering source language as real and auxiliary languages as fake
            real_lang, fake_lang = lang_in_batch[batch - 1]
            real_idx, fake_idx = lang_ids.get(real_lang), lang_ids.get(fake_lang, -1)

            #
            word, char, pos, heads, types, masks, lengths, bert_inputs = conllx_data.get_batch_variable(
                train_data[real_lang],
                batch_size,
                unk_replace=unk_replace)

            if use_gpu:
                word = word.cuda()
                char = char.cuda()
                pos = pos.cuda()
                heads = heads.cuda()
                types = types.cuda()
                masks = masks.cuda()
                lengths = lengths.cuda()
                if bert_inputs[0] is not None:
                    bert_inputs[0] = bert_inputs[0].cuda()
                    bert_inputs[1] = bert_inputs[1].cuda()
                    bert_inputs[2] = bert_inputs[2].cuda()

            real_enc = network(word, char, pos, input_bert=bert_inputs, mask=masks, length=lengths, hx=None)

            # ========== Update the discriminator ==========
            if t_count > 0 and skip_adv_tuning > args.delay:
                # fake examples = 0
                word_f, char_f, pos_f, heads_f, types_f, masks_f, lengths_f, bert_inputs = conllx_data.get_batch_variable(
                    train_data[fake_lang],
                    batch_size,
                    unk_replace=unk_replace
                )

                if use_gpu:
                    word_f = word_f.cuda()
                    char_f = char_f.cuda()
                    pos_f = pos_f.cuda()
                    heads_f = heads_f.cuda()
                    types_f = types_f.cuda()
                    masks_f = masks_f.cuda()
                    lengths_f = lengths_f.cuda()
                    if bert_inputs[0] is not None:
                        bert_inputs[0] = bert_inputs[0].cuda()
                        bert_inputs[1] = bert_inputs[1].cuda()
                        bert_inputs[2] = bert_inputs[2].cuda()

                fake_enc = network(word_f, char_f, pos_f, input_bert=bert_inputs, mask=masks_f, length=lengths_f,
                                   hx=None)

                # TODO: temporary crack
                if t_count > 0 and skip_adv_tuning > args.delay:
                    # skip discriminator training for '|n_critic|' iterations if 'n_critic' < 0
                    if args.n_critic > 0 or (batch - 1) % (-1 * args.n_critic) == 0:
                        update_discriminator = True

            if update_discriminator:
                if args.adv_training:
                    real_loss, fake_loss, real_acc, fake_acc = AdvAgent.update(real_enc['output'].detach(),
                                                                               fake_enc['output'].detach(),
                                                                               real_idx, fake_idx)

                    loss_d_real.append(real_loss)
                    loss_d_fake.append(fake_loss)
                    acc_d_real.append(real_acc)
                    acc_d_fake.append(fake_acc)

                elif args.motivate:
                    real_loss, fake_loss, real_acc, fake_acc = MtvAgent.update(real_enc['output'].detach(),
                                                                               fake_enc['output'].detach(),
                                                                               real_idx, fake_idx)

                    loss_d_real.append(real_loss)
                    loss_d_fake.append(fake_loss)
                    acc_d_real.append(real_acc)
                    acc_d_fake.append(fake_acc)

                else:
                    raise NotImplementedError()

                if args.n_critic > 0 and (batch - 1) % args.n_critic != 0:
                    update_generator = False

            # ==============================================

            # =========== Update the generator =============
            if update_generator:
                others_loss = None
                if args.adv_training and skip_adv_tuning > args.delay:
                    # for GAN: L_G= L_parsing - (lambda_G * L_D)
                    # for GR : L_G= L_parsing +  L_D
                    others_loss = AdvAgent.gen_loss(real_enc['output'], fake_enc['output'],
                                                    real_idx, fake_idx)
                    gen_loss.append(others_loss.item())

                elif args.motivate and skip_adv_tuning > args.delay:
                    others_loss = MtvAgent.gen_loss(real_enc['output'], fake_enc['output'],
                                                    real_idx, fake_idx)
                    gen_loss.append(others_loss.item())

                optim.zero_grad()

                loss_arc, loss_type = loss_fn(real_enc['output'], heads, types, mask=masks, length=lengths)
                loss = loss_arc + loss_type

                num_inst = word.size(0) if obj == 'crf' else masks.sum() - word.size(0)
                train_err += loss.item() * num_inst
                train_err_arc += loss_arc.item() * num_inst
                train_err_type += loss_type.item() * num_inst
                train_total += num_inst
                parsing_loss.append(loss.item())

                if others_loss is not None:
                    loss = loss + others_loss

                loss.backward()
                clip_grad_norm_(network.parameters(), clip)
                optim.step()

                time_ave = (time.time() - start_time) / batch
                time_left = (num_batches - batch) * time_ave

        if (args.adv_training or args.motivate) and skip_adv_tuning > args.delay:
            logger.info(
                'epoch: %d train: %d loss: %.4f, arc: %.4f, type: %.4f, dis_loss: (%.2f, %.2f), dis_acc: (%.2f, %.2f), '
                'gen_loss: %.2f, time: %.2fs' % (
                    epoch, num_batches,
                    train_err / train_total,
                    train_err_arc / train_total,
                    train_err_type / train_total,
                    sum(loss_d_real) / len(loss_d_real),
                    sum(loss_d_fake) / len(loss_d_fake),
                    sum(acc_d_real) / len(acc_d_real),
                    sum(acc_d_fake) / len(acc_d_fake),
                    sum(gen_loss) / len(gen_loss),
                    time.time() - start_time))
        else:
            logger.info('epoch: %d train: %d loss: %.4f, arc: %.4f, type: %.4f, time: %.2fs' % (
                epoch, num_batches,
                train_err / train_total,
                train_err_arc / train_total,
                train_err_type / train_total,
                time.time() - start_time))

        ################# Validation on Dependency Parsing Only #################################
        if epoch % args.check_dev != 0:
            continue

        with torch.no_grad():
            # evaluate performance on dev data
            network.eval()

            dev_ucorr = 0.0
            dev_lcorr = 0.0
            dev_total = 0
            dev_ucomlpete = 0.0
            dev_lcomplete = 0.0
            dev_ucorr_nopunc = 0.0
            dev_lcorr_nopunc = 0.0
            dev_total_nopunc = 0
            dev_ucomlpete_nopunc = 0.0
            dev_lcomplete_nopunc = 0.0
            dev_root_corr = 0.0
            dev_total_root = 0.0
            dev_total_inst = 0.0

            for lang, data_dev in dev_data.items():
                for batch in conllx_data.iterate_batch_variable(data_dev, batch_size):
                    word, char, pos, heads, types, masks, lengths, bert_inputs = batch

                    if use_gpu:
                        word = word.cuda()
                        char = char.cuda()
                        pos = pos.cuda()
                        heads = heads.cuda()
                        types = types.cuda()
                        masks = masks.cuda()
                        lengths = lengths.cuda()
                        if bert_inputs[0] is not None:
                            bert_inputs[0] = bert_inputs[0].cuda()
                            bert_inputs[1] = bert_inputs[1].cuda()
                            bert_inputs[2] = bert_inputs[2].cuda()

                    heads_pred, types_pred = decode(word, char, pos, input_bert=bert_inputs, mask=masks, length=lengths,
                                                    leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                    word = word.cpu().numpy()
                    pos = pos.cpu().numpy()
                    lengths = lengths.cpu().numpy()
                    heads = heads.cpu().numpy()
                    types = types.cpu().numpy()

                    stats, stats_nopunc, stats_root, num_inst = parser.eval(word, pos, heads_pred, types_pred, heads,
                                                                            types,
                                                                            word_alphabet, pos_alphabet, lengths,
                                                                            punct_set=punct_set, symbolic_root=True)
                    ucorr, lcorr, total, ucm, lcm = stats
                    ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
                    corr_root, total_root = stats_root

                    dev_ucorr += ucorr
                    dev_lcorr += lcorr
                    dev_total += total
                    dev_ucomlpete += ucm
                    dev_lcomplete += lcm

                    dev_ucorr_nopunc += ucorr_nopunc
                    dev_lcorr_nopunc += lcorr_nopunc
                    dev_total_nopunc += total_nopunc
                    dev_ucomlpete_nopunc += ucm_nopunc
                    dev_lcomplete_nopunc += lcm_nopunc

                    dev_root_corr += corr_root
                    dev_total_root += total_root
                    dev_total_inst += num_inst

            print('W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
                dev_ucorr, dev_lcorr, dev_total, dev_ucorr * 100 / dev_total, dev_lcorr * 100 / dev_total,
                dev_ucomlpete * 100 / dev_total_inst, dev_lcomplete * 100 / dev_total_inst))
            print('Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
                dev_ucorr_nopunc, dev_lcorr_nopunc, dev_total_nopunc, dev_ucorr_nopunc * 100 / dev_total_nopunc,
                dev_lcorr_nopunc * 100 / dev_total_nopunc,
                dev_ucomlpete_nopunc * 100 / dev_total_inst, dev_lcomplete_nopunc * 100 / dev_total_inst))
            print('Root: corr: %d, total: %d, acc: %.2f%%' % (
                dev_root_corr, dev_total_root, dev_root_corr * 100 / dev_total_root))

            if dev_lcorrect_nopunc < dev_lcorr_nopunc or (
                            dev_lcorrect_nopunc == dev_lcorr_nopunc and dev_ucorrect_nopunc < dev_ucorr_nopunc):
                dev_ucorrect_nopunc = dev_ucorr_nopunc
                dev_lcorrect_nopunc = dev_lcorr_nopunc
                dev_ucomlpete_match_nopunc = dev_ucomlpete_nopunc
                dev_lcomplete_match_nopunc = dev_lcomplete_nopunc

                dev_ucorrect = dev_ucorr
                dev_lcorrect = dev_lcorr
                dev_ucomlpete_match = dev_ucomlpete
                dev_lcomplete_match = dev_lcomplete

                dev_root_correct = dev_root_corr

                best_epoch = epoch
                patient = 0

                state_dict = network.module.state_dict() if args.parallel else network.state_dict()
                torch.save(state_dict, model_name)

            else:
                if dev_ucorr_nopunc * 100 / dev_total_nopunc < dev_ucorrect_nopunc * 100 / dev_total_nopunc - 5 or patient >= schedule:
                    state_dict = torch.load(model_name)
                    if args.parallel:
                        network.module.load_state_dict(state_dict)
                    else:
                        network.load_state_dict(state_dict)

                    lr = lr * decay_rate
                    optim = generate_optimizer(opt, lr, network.parameters(),
                                               betas, gamma, eps, momentum)

                    if decoding == 'greedy':
                        decode = network.module.decode if args.parallel else network.decode
                    elif decoding == 'mst':
                        decode = network.module.decode_mst if args.parallel else network.decode_mst
                    else:
                        raise ValueError('Unknown decoding algorithm: %s' % decoding)

                    patient = 0
                    decay += 1
                    if decay % double_schedule_decay == 0:
                        schedule *= 2
                else:
                    patient += 1

            print(
                '----------------------------------------------------------------------------------------------------------------------------')
            print(
                'best dev  W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                    dev_ucorrect, dev_lcorrect, dev_total, dev_ucorrect * 100 / dev_total,
                    dev_lcorrect * 100 / dev_total,
                    dev_ucomlpete_match * 100 / dev_total_inst, dev_lcomplete_match * 100 / dev_total_inst,
                    best_epoch))
            print(
                'best dev  Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                    dev_ucorrect_nopunc, dev_lcorrect_nopunc, dev_total_nopunc,
                    dev_ucorrect_nopunc * 100 / dev_total_nopunc, dev_lcorrect_nopunc * 100 / dev_total_nopunc,
                    dev_ucomlpete_match_nopunc * 100 / dev_total_inst,
                    dev_lcomplete_match_nopunc * 100 / dev_total_inst,
                    best_epoch))
            print('best dev  Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
                dev_root_correct, dev_total_root, dev_root_correct * 100 / dev_total_root, best_epoch))
            print(
                '----------------------------------------------------------------------------------------------------------------------------')
            if decay == max_decay:
                break

        torch.cuda.empty_cache()  # release memory that can be released


if __name__ == '__main__':
    main()
