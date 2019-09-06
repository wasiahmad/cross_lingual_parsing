from __future__ import print_function

__author__ = 'max'
"""
Implementation of a language classifier.
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
from neuronlp2.io import get_logger, conllx_stacked_data, conllx_data
from neuronlp2.models import StackPtrNet, BiRecurrentConvBiAffine

from neuronlp2.io_multi import guess_language_id, create_alphabets, lang_specific_word
from neuronlp2.io_multi.multi_vocab import iter_file
from neuronlp2.utils import load_embedding_dict
from neuronlp2.io.utils import DIGIT_RE

uid = uuid.uuid4().hex[:6]

UD_languages = ['en', 'no', 'sv', 'fr', 'pt', 'da', 'es', 'it', 'ca', 'hr', 'pl', 'sl' 'uk', 'bg', 'cs', 'de', 'he',
                'nl', 'ru', 'ro', 'id', 'sk', 'lv', 'et', 'fi', 'ar', 'la', 'ko', 'hi']


def main():
    args_parser = argparse.ArgumentParser(description='Tuning with stack pointer parser')
    args_parser.add_argument('--parser', choices=['stackptr', 'biaffine'], help='Parser', required=True)
    args_parser.add_argument('--langs', nargs='+', help='Languages to train the classifier')
    args_parser.add_argument('--test_lang', default=None, help='Language to be tested')
    args_parser.add_argument('--data_dir', help='Data directory path')
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--model_name', help='name for saving model file.', required=True)
    args_parser.add_argument('--pre_model_path', type=str, required=True,
                             help='Path of the pretrained model.')
    args_parser.add_argument('--pre_model_name', type=str, required=True,
                             help='Name of the pretrained model.')
    args_parser.add_argument('--gpu', action='store_true', help='Using GPU')
    args_parser.add_argument('--nclass', type=int, required=True, help='Number of language classes')
    #
    args_parser.add_argument('--embed_dir', type=str,
                             help="Path for extra embedding file for extra language testing.")
    args_parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    args_parser.add_argument('--train_level', type=str, default='word', choices=['word', 'sent'],
                             help='Use X-level classifier training')

    # fix data-prepare seed
    random.seed(1234)
    np.random.seed(1234)
    # model's seed
    torch.manual_seed(1234)

    args = args_parser.parse_args()
    logger = get_logger("Classifier")

    model_path = args.model_path
    model_name = args.model_name
    pre_model_path = args.pre_model_path
    pre_model_name = args.pre_model_name

    use_gpu = args.gpu
    parser = args.parser

    if parser == 'stackptr':
        raise NotImplementedError("I'm lazy!")
    elif parser == 'biaffine':
        biaffine(model_path, model_name, pre_model_path, pre_model_name, use_gpu, logger, args)
    else:
        raise ValueError('Unknown parser: %s' % parser)


def biaffine(model_path, model_name, pre_model_path, pre_model_name, use_gpu, logger, args):
    alphabet_path = os.path.join(pre_model_path, 'alphabets/')
    logger.info("Alphabet Path: %s" % alphabet_path)
    pre_model_name = os.path.join(pre_model_path, pre_model_name)
    model_name = os.path.join(model_path, model_name)

    # Load pre-created alphabets
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_sent_length = conllx_data.create_alphabets(
        alphabet_path, None, data_paths=[None, None], max_vocabulary_size=50000, embedd_dict=None)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    logger.info('use gpu: %s' % (use_gpu))

    if args.test_lang:
        extra_embed = args.embed_dir + ("wiki.multi.%s.vec" % args.test_lang)
        extra_word_dict, _ = load_embedding_dict('word2vec', extra_embed)
        test_path = args.data_dir + args.test_lang + '_test.conllu'
        extra_embeds_arr = augment_with_extra_embedding(word_alphabet, extra_word_dict, test_path, logger)
    else:
        extra_embeds_arr = []
        for language in args.langs:
            extra_embed = args.embed_dir + ("wiki.multi.%s.vec" % language)
            extra_word_dict, _ = load_embedding_dict('word2vec', extra_embed)

            test_path = args.data_dir + language + '_train.conllu'
            embeds_arr1 = augment_with_extra_embedding(word_alphabet, extra_word_dict, test_path, logger)
            test_path = args.data_dir + language + '_dev.conllu'
            embeds_arr2 = augment_with_extra_embedding(word_alphabet, extra_word_dict, test_path, logger)
            test_path = args.data_dir + language + '_test.conllu'
            embeds_arr3 = augment_with_extra_embedding(word_alphabet, extra_word_dict, test_path, logger)
            extra_embeds_arr.extend(embeds_arr1 + embeds_arr2 + embeds_arr3)

    # ------------------------------------------------------------------------- #
    # --------------------- Loading model ------------------------------------- #

    def load_model_arguments_from_json():
        arguments = json.load(open(arg_path, 'r'))
        return arguments['args'], arguments['kwargs']

    arg_path = pre_model_name + '.arg.json'
    margs, kwargs = load_model_arguments_from_json()
    network = BiRecurrentConvBiAffine(use_gpu=use_gpu, *margs, **kwargs)
    network.load_state_dict(torch.load(pre_model_name))
    args.use_bert = kwargs.get('use_bert', False)

    #
    augment_network_embed(word_alphabet.size(), network, extra_embeds_arr)

    network.eval()
    logger.info('model: %s' % pre_model_name)

    # Freeze the network
    for p in network.parameters():
        p.requires_grad = False

    nclass = args.nclass
    classifier = nn.Sequential(
        nn.Linear(network.encoder.output_dim, 512),
        nn.Linear(512, nclass)
    )

    if use_gpu:
        network.cuda()
        classifier.cuda()
    else:
        network.cpu()
        classifier.cpu()

    batch_size = args.batch_size

    # ===== the reading
    def _read_one(path, is_train=False, max_size=None):
        lang_id = guess_language_id(path)
        logger.info("Reading: guess that the language of file %s is %s." % (path, lang_id))
        one_data = conllx_data.read_data_to_variable(path, word_alphabet, char_alphabet, pos_alphabet,
                                                     type_alphabet, use_gpu=use_gpu, volatile=(not is_train),
                                                     use_bert=args.use_bert, symbolic_root=True, lang_id=lang_id,
                                                     max_size=max_size)
        return one_data

    def compute_accuracy(data, lang_idx):
        total_corr, total = 0, 0
        classifier.eval()
        with torch.no_grad():
            for batch in conllx_data.iterate_batch_variable(data, batch_size):
                word, char, pos, _, _, masks, lengths, bert_inputs = batch
                if use_gpu:
                    word = word.cuda()
                    char = char.cuda()
                    pos = pos.cuda()
                    masks = masks.cuda()
                    lengths = lengths.cuda()
                    if bert_inputs[0] is not None:
                        bert_inputs[0] = bert_inputs[0].cuda()
                        bert_inputs[1] = bert_inputs[1].cuda()
                        bert_inputs[2] = bert_inputs[2].cuda()

                output = network.forward(word, char, pos, input_bert=bert_inputs,
                                         mask=masks, length=lengths, hx=None)
                output = output['output'].detach()

                if args.train_level == 'word':
                    output = classifier(output)
                    output = output.contiguous().view(-1, output.size(2))
                else:
                    output = torch.mean(output, dim=1)
                    output = classifier(output)

                preds = output.max(1)[1].cpu()
                labels = torch.LongTensor([lang_idx])
                labels = labels.expand(*preds.size())
                n_correct = preds.eq(labels).sum().item()
                total_corr += n_correct
                total += output.size(0)

            return {'total_corr': total_corr, 'total': total}

    if args.test_lang:
        classifier.load_state_dict(torch.load(model_name))
        path = args.data_dir + args.test_lang + '_train.conllu'
        test_data = _read_one(path)

        # TODO: fixed indexing is not GOOD
        lang_idx = 0 if args.test_lang == args.src_lang else 1
        result = compute_accuracy(test_data, lang_idx)
        accuracy = (result['total_corr'] * 100.0) / result['total']
        logger.info('[Classifier performance] Language: %s || accuracy: %.2f%%' % (args.test_lang, accuracy))

    else:
        # if output directory doesn't exist, create it
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

        # --------------------- Loading data -------------------------------------- #
        train_data = dict()
        dev_data = dict()
        test_data = dict()
        num_data = dict()
        lang_ids = dict()
        reverse_lang_ids = dict()

        # loading language data
        for language in args.langs:
            lang_ids[language] = len(lang_ids)
            reverse_lang_ids[lang_ids[language]] = language

            train_path = args.data_dir + language + '_train.conllu'
            # Utilize at most 10000 examples
            tmp_data = _read_one(train_path, max_size=10000)
            num_data[language] = sum(tmp_data[1])
            train_data[language] = tmp_data

            dev_path = args.data_dir + language + '_dev.conllu'
            tmp_data = _read_one(dev_path)
            dev_data[language] = tmp_data

            test_path = args.data_dir + language + '_test.conllu'
            tmp_data = _read_one(test_path)
            test_data[language] = tmp_data

        # ------------------------------------------------------------------------- #

        optim = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        def compute_loss(lang_name, land_idx):
            word, char, pos, _, _, masks, lengths, bert_inputs = conllx_data.get_batch_variable(train_data[lang_name],
                                                                                                batch_size,
                                                                                                unk_replace=0.5)

            if use_gpu:
                word = word.cuda()
                char = char.cuda()
                pos = pos.cuda()
                masks = masks.cuda()
                lengths = lengths.cuda()
                if bert_inputs[0] is not None:
                    bert_inputs[0] = bert_inputs[0].cuda()
                    bert_inputs[1] = bert_inputs[1].cuda()
                    bert_inputs[2] = bert_inputs[2].cuda()

            output = network.forward(word, char, pos, input_bert=bert_inputs,
                                     mask=masks, length=lengths, hx=None)
            output = output['output'].detach()

            if args.train_level == 'word':
                output = classifier(output)
                output = output.contiguous().view(-1, output.size(2))
            else:
                output = torch.mean(output, dim=1)
                output = classifier(output)

            labels = torch.empty(output.size(0)).fill_(land_idx).type_as(output).long()
            loss = criterion(output, labels)
            return loss

        # ---------------------- Form the mini-batches -------------------------- #
        num_batches = 0
        batch_lang_labels = []
        for lang in args.langs:
            nbatches = num_data[lang] // batch_size + 1
            batch_lang_labels.extend([lang] * nbatches)
            num_batches += nbatches

        assert len(batch_lang_labels) == num_batches
        # ------------------------------------------------------------------------- #

        best_dev_accuracy = 0
        patience = 0
        for epoch in range(1, args.num_epochs + 1):
            # shuffling the data
            lang_in_batch = copy.copy(batch_lang_labels)
            random.shuffle(lang_in_batch)

            classifier.train()
            for batch in range(1, num_batches + 1):
                lang_name = lang_in_batch[batch - 1]
                lang_id = lang_ids.get(lang_name)

                loss = compute_loss(lang_name, lang_id)
                loss.backward()
                optim.step()

            # Validation
            avg_acc = dict()
            for dev_lang in dev_data.keys():
                lang_idx = lang_ids.get(dev_lang)
                result = compute_accuracy(dev_data[dev_lang], lang_idx)
                accuracy = (result['total_corr'] * 100.0) / result['total']
                avg_acc[dev_lang] = accuracy

            acc = ', '.join('%s: %.2f' % (key, val) for (key, val) in avg_acc.items())
            logger.info('Epoch: %d, Performance[%s]' % (epoch, acc))

            avg_acc = sum(avg_acc.values()) / len(avg_acc)
            if best_dev_accuracy < avg_acc:
                best_dev_accuracy = avg_acc
                patience = 0
                state_dict = classifier.state_dict()
                torch.save(state_dict, model_name)
            else:
                patience += 1

            if patience >= 5:
                break

        # Testing
        logger.info('Testing model %s' % pre_model_name)
        total_corr, total = 0, 0
        for test_lang in UD_languages:
            if test_lang in test_data:
                lang_idx = lang_ids.get(test_lang)
                result = compute_accuracy(test_data[test_lang], lang_idx)
                accuracy = (result['total_corr'] * 100.0) / result['total']
                print('[LANG]: %s, [ACC]: %.2f' % (test_lang.upper(), accuracy))
                total_corr += result['total_corr']
                total += result['total']
        print('[Avg. Performance]: %.2f' % ((total_corr * 100.0) / total))


# ==========
# about augmenting the vocabs and embeddings with the current test file and extra embeddings
def augment_with_extra_embedding(the_alphabet, extra_word_dict, test_file, logger):
    extra_embeds_arr = []
    if extra_word_dict is not None:
        # reopen the vocab
        the_alphabet.open()
        lang_id = guess_language_id(test_file)
        for one_sent in iter_file(test_file):
            for w in one_sent["word"]:
                already_spec = w.startswith("!en_")
                if already_spec:
                    normed_word = w
                else:
                    normed_word = DIGIT_RE.sub("0", w)
                    normed_word = lang_specific_word(normed_word, lang_id=lang_id)
                #
                if normed_word in the_alphabet.instance2index:
                    continue
                # TODO: assume english is the source for run-translate
                if already_spec:
                    w = w[4:]
                    assert False, 'English is the source language'
                else:
                    check_dict = extra_word_dict
                #
                if w in check_dict:
                    new_embed_arr = check_dict[w]
                elif w.lower() in check_dict:
                    new_embed_arr = check_dict[w.lower()]
                else:
                    new_embed_arr = None
                if new_embed_arr is not None:
                    extra_embeds_arr.append(new_embed_arr)
                    the_alphabet.add(normed_word)
        # close the vocab
        the_alphabet.close()

    logger.info(
        "Augmenting the vocab with new words of %s, now vocab is %s." % (len(extra_embeds_arr), the_alphabet.size()))
    return extra_embeds_arr


def augment_network_embed(new_size, network, extra_embeds_arr):
    if len(extra_embeds_arr) == 0:
        return
    old_embed = network.embedder.word_embedd
    if old_embed is None:
        return
    old_arr = old_embed.weight.detach().cpu().numpy()
    new_arr = np.concatenate([old_arr, extra_embeds_arr], 0)
    assert new_arr.shape[0] == new_size, str(new_arr.shape[0]) + ' != ' + str(new_size)
    new_embed = nn.Embedding(new_size, new_arr.shape[1], _weight=torch.from_numpy(new_arr))
    network.embedder.word_embedd = new_embed


# ==========

if __name__ == '__main__':
    main()
