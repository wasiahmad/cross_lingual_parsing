__author__ = 'max'

from .instance import DependencyInstance, NERInstance
from .instance import Sentence
from .conllx_data import ROOT, ROOT_POS, ROOT_CHAR, ROOT_TYPE, END, END_POS, END_CHAR, END_TYPE
from . import utils
from ..io_multi import get_word_index_with_spec

from pytorch_transformers import BertTokenizer


def convert_example_to_features(tokens, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    def convert_tokens_to_ids(tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(tokenizer.vocab.get(token, tokenizer.vocab['[UNK]']))
            # ids.append(tokenizer.vocab[token])
        return ids

    assert isinstance(tokens, list)

    input_tokens = []
    val_pos = []
    end_idx = 0
    for word in tokens:
        b_token = tokenizer.tokenize(word)  # we expect |token| = 1
        input_tokens.extend(b_token)
        val_pos.append(end_idx)
        end_idx += len(b_token)

    input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]
    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = convert_tokens_to_ids(input_tokens)

    assert len(input_ids) == len(input_tokens)
    assert max(input_ids) < len(tokenizer.vocab)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    return {
        'input_len': len(tokens),
        'input_tokens': input_tokens,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'out_positions': val_pos
    }


class CoNLLXReader(object):
    def __init__(self,
                 file_path,
                 word_alphabet,
                 char_alphabet,
                 pos_alphabet,
                 type_alphabet,
                 lang_id="",
                 use_bert=False):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet
        self.lang_id = lang_id
        self.use_bert = use_bert
        if use_bert:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                                                do_lower_case=False)

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False):
        lines = []
        line = self.__source_file.readline()
        while line is not None and len(line.strip()) > 0:
            line = line.strip()
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        types = []
        type_ids = []
        heads = []

        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))
            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])
            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))
            types.append(ROOT_TYPE)
            type_ids.append(self.__type_alphabet.get_index(ROOT_TYPE))
            heads.append(0)

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > utils.MAX_CHAR_LENGTH:
                chars = chars[:utils.MAX_CHAR_LENGTH]
                char_ids = char_ids[:utils.MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = utils.DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            pos = tokens[4]
            head = int(tokens[6])
            type = utils.get_main_deplabel(tokens[7])

            words.append(word)
            # ===== modified: with lang_id prefix (with backoff to default lang)
            one_word_id = get_word_index_with_spec(self.__word_alphabet, word, self.lang_id)
            word_ids.append(one_word_id)
            # =====

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            heads.append(head)

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))
            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))
            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))
            heads.append(0)

        bert_tokens = None
        if self.use_bert:
            bert_tokens = convert_example_to_features(words, self.bert_tokenizer)

        return DependencyInstance(Sentence(words, word_ids, char_seqs, char_id_seqs, bert_tokens), postags, pos_ids,
                                  heads, types, type_ids)


class CoNLL03Reader(object):
    def __init__(self,
                 file_path,
                 word_alphabet,
                 char_alphabet,
                 pos_alphabet,
                 chunk_alphabet,
                 ner_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__chunk_alphabet = chunk_alphabet
        self.__ner_alphabet = ner_alphabet

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True):
        lines = []
        line = self.__source_file.readline()
        while line is not None and len(line.strip()) > 0:
            line = line.strip()
            lines.append(line.split(' '))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        chunk_tags = []
        chunk_ids = []
        ner_tags = []
        ner_ids = []

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > utils.MAX_CHAR_LENGTH:
                chars = chars[:utils.MAX_CHAR_LENGTH]
                char_ids = char_ids[:utils.MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = utils.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
            pos = tokens[2]
            chunk = tokens[3]
            ner = tokens[4]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            chunk_tags.append(chunk)
            chunk_ids.append(self.__chunk_alphabet.get_index(chunk))

            ner_tags.append(ner)
            ner_ids.append(self.__ner_alphabet.get_index(ner))

        return NERInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), postags, pos_ids, chunk_tags, chunk_ids,
                           ner_tags, ner_ids)
