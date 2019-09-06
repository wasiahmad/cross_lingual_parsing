#

# build the vocab/dictionary from outside to all related lexicons

from __future__ import print_function

import os
import sys
import argparse

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from neuronlp2 import utils
from neuronlp2.io import get_logger, conllx_stacked_data
from neuronlp2.io_multi import create_alphabets, lang_specific_word


#
# Only use for building multi-lingual vocabs, this is only a simple workaround.
# However, we might also want multi-lingual embeddings before training for convenience.
# Usage:
# python2 examples/vocab/build_vocab.py --word_embedding <embde-type> --word_paths [various languages' embeddings: e1 e2 ...]
# --train <english-train-file> --extra [various languages' test-files: ... ] --model_path <path>

#
def parse_cmd(args):
    args_parser = argparse.ArgumentParser(description='Building the alphabets/vocabularies.')
    #
    args_parser.add_argument('--word_embedding', type=str, choices=['word2vec', 'glove', 'senna', 'sskip', 'polyglot'],
                             help='Embedding for words', required=True)
    args_parser.add_argument('--embed_lang_id', type=str, help='lang id for the embeddings', required=True)
    args_parser.add_argument('--word_paths', type=str, nargs='+', help='path for word embedding dict', required=True)
    args_parser.add_argument('--train', type=str, help="The main file to build vocab.", required=True)
    args_parser.add_argument('--extra', type=str, nargs='+', help="Extra files to build vocab, usually dev/tests.",
                             required=True)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    res = args_parser.parse_args(args)
    return res


def _get_keys(wd):
    try:
        return wd.keys()
    except:
        # Word2VecKeyedVectors
        return wd.vocab.keys()


# todo(warn): if not care about the specific language of the embeddings
def combine_embeds(word_dicts):
    num_dicts = len(word_dicts)
    count_ins, count_repeats = [0 for _ in range(num_dicts)], [0 for _ in range(num_dicts)]
    res = dict()
    for idx, one in enumerate(word_dicts):
        for k in _get_keys(one):
            if k in res:
                count_repeats[idx] += 1
            else:
                count_ins[idx] += 1
                res[k] = 0
    return res, count_ins, count_repeats


def main(a=None):
    if a is None:
        a = sys.argv[1:]
    args = parse_cmd(a)
    # if output directory doesn't exist, create it
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    logger = get_logger("VocabBuilder", args.model_path + '/vocab.log.txt')
    logger.info('\ncommand-line params : {0}\n'.format(sys.argv[1:]))
    logger.info('{0}\n'.format(args))

    # load embeds
    logger.info("Load embeddings")
    word_embeds = [WordVectors.load(one_embed_path) for one_embed_path in args.word_paths]
    combined_word_dict = WordVectors.combine_embeds(word_embeds)
    logger.info("Final combined un-pruned embeddings size: %d." % len(combined_word_dict))

    # create vocabs
    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(args.model_path, args.embed_lang_id + '_alphabets/')
    assert not os.path.exists(alphabet_path), "Alphabet path exists, please build with a new path."
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_sent_length = create_alphabets(
        alphabet_path, args.train, data_paths=args.extra, max_vocabulary_size=100000, embedd_dict=combined_word_dict)

    # save filtered embed
    hit_keys = set()
    for one_w in word_alphabet.instance2index:
        if one_w in combined_word_dict:
            hit_keys.add(one_w)
        elif one_w.lower() in combined_word_dict:
            hit_keys.add(one_w.lower())
    logger.info("Hit keys: %d" % len(hit_keys))
    filtered_embed = combined_word_dict.filter(hit_keys)
    filtered_embed.save(os.path.join(alphabet_path, 'wiki.multi.' + args.embed_lang_id + '.vec'))


class WordVectors:
    def __init__(self):
        self.num_words = None
        self.embed_size = None
        self.words = []
        self.vecs = {}

    def __len__(self):
        return len(self.vecs)

    def __contains__(self, item):
        return item in self.vecs

    def has_key(self, k, lc_back=True):
        if k in self.vecs:
            return True
        elif lc_back:
            return str.lower(k) in self.vecs
        return False

    def get_vec(self, k, df=None, lc_back=True):
        if k in self.vecs:
            return self.vecs[k]
        elif lc_back:
            # back to lowercased
            lc = str.lower(k)
            if lc in self.vecs:
                return self.vecs[lc]
        return df

    def save(self, fname):
        print("Saving w2v num_words=%d, embed_size=%d to %s." % (self.num_words, self.embed_size, fname))
        with open(fname, "w") as fd:
            fd.write("%d %d\n" % (self.num_words, self.embed_size))
            for w in self.words:
                vec = self.vecs[w]
                print_list = [w.encode('utf-8')] + ["%.6f" % float(z) for z in vec]
                fd.write(" ".join(print_list) + "\n")

    def filter(self, key_set):
        one = WordVectors()
        one.num_words, one.embed_size = self.num_words, self.embed_size
        for w in self.words:
            if w in key_set:
                one.words.append(w)
                one.vecs[w] = self.vecs[w]
        one.num_words = len(one.vecs)
        print(
            "Filter from num_words=%d/embed_size=%d to num_words=%s" % (self.num_words, self.embed_size, one.num_words))
        return one

    @staticmethod
    def load(fname):
        print("Loading pre-trained w2v from %s ..." % fname)
        one = WordVectors()
        with open(fname) as fd:
            # first line
            line = fd.readline().strip()
            try:
                one.num_words, one.embed_size = [int(x) for x in line.split()]
                print("Reading w2v num_words=%d, embed_size=%d." % (one.num_words, one.embed_size))
                line = fd.readline().strip()
            except:
                print("Reading w2v.")
            # the rest
            while len(line) > 0:
                fields = line.split(" ")
                word, vec = fields[0], [float(x) for x in fields[1:]]
                assert word not in one.vecs, "Repeated key."
                if one.embed_size is None:
                    one.embed_size = len(vec)
                else:
                    assert len(vec) == one.embed_size, "Unmatched embed dimension."
                one.vecs[word] = vec
                one.words.append(word)
                line = fd.readline().strip()
        # final
        if one.num_words is None:
            one.num_words = len(one.vecs)
            print("Reading w2v num_words=%d, embed_size=%d." % (one.num_words, one.embed_size))
        else:
            assert one.num_words == len(one.vecs), "Unmatched num of words."
        return one

    @staticmethod
    def combine_embeds(word_dicts):
        number_to_combine = len(word_dicts)
        #
        one = WordVectors()
        one.embed_size = word_dicts[0].embed_size
        for idx in range(number_to_combine):
            cur_embed = word_dicts[idx]
            for one_w in cur_embed.words:
                prefixed_w = one_w
                one.words.append(prefixed_w)
                one.vecs[prefixed_w] = cur_embed.vecs[one_w]
        one.num_words = len(one.vecs)
        return one


if __name__ == '__main__':
    main()
