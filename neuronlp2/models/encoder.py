import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..nn import VarMaskedGRU, VarMaskedRNN, VarMaskedLSTM, VarMaskedFastLSTM

from ..transformer import TransformerEncoder
from pytorch_transformers import BertModel


class Embedder(nn.Module):
    def __init__(self,
                 word_dim,
                 num_words,
                 char_dim,
                 num_chars,
                 pos_dim,
                 num_pos,
                 num_filters,
                 kernel_size,
                 embedd_word=None,
                 embedd_char=None,
                 embedd_pos=None,
                 p_in=0.33,
                 p_out=0.33,
                 pos=True,
                 char=True,
                 train_position=False,
                 position_dim=50,
                 max_sent_length=200,
                 use_gpu=False,
                 use_word_emb=True,
                 input_concat_embeds=False,
                 input_concat_position=False,
                 use_bert=False):
        super(Embedder, self).__init__()

        self.word_embedd = nn.Embedding(num_words, word_dim, _weight=embedd_word) if use_word_emb else None
        self.pos_embedd = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos) if pos else None
        self.char_embedd = nn.Embedding(num_chars, char_dim, _weight=embedd_char) if char else None
        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1) if char else None

        if use_bert:
            self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
            self.bert.eval()

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)

        self.use_gpu = use_gpu
        self.use_pos = pos
        self.use_char = char
        self.use_word = use_word_emb
        self.use_bert = use_bert

        self.output_dim = {'word_dim': 0, 'char_dim': 0, 'pos_dim': 0}
        if self.use_word:
            self.output_dim['word_dim'] = word_dim
        if self.use_char:
            self.output_dim['char_dim'] = num_filters
        if self.use_pos:
            self.output_dim['pos_dim'] = pos_dim
        if self.use_bert:
            self.output_dim['word_dim'] = self.bert.config.hidden_size

        self.input_concat_embeds = input_concat_embeds
        if self.input_concat_embeds:
            self.output_dim['total'] = sum(self.output_dim.values())
        else:
            # all the embeddings will be added
            assert len(set(self.output_dim.values())) == 1, "embedding size mismatch"
            self.output_dim['total'] = self.output_dim.values()[0]

        # Positional Embeddings
        if position_dim > 0:
            self.position_embedding = nn.Embedding(max_sent_length, position_dim)
            if not train_position:
                self.position_embedding.weight.requires_grad = False  # turn off pos embedding training
                # keep dim 0 for padding token position encoding zero vector
                position_enc = np.array([
                    [pos / np.power(10000, 2 * (j // 2) / position_dim)
                     for j in range(position_dim)]
                    if pos != 0 else np.zeros(position_dim)
                    for pos in range(max_sent_length)
                ])
                position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
                position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
                self.position_embedding.weight.data.copy_(torch.from_numpy(position_enc).type(torch.FloatTensor))

        self.input_concat_position = input_concat_position
        self.output_dim['position_dim'] = position_dim
        if self.input_concat_position:
            self.output_dim['total'] += position_dim

    def forward(self, input_word, input_char, input_pos, input_bert):
        src_encoding = None

        if self.use_word:
            # [batch, length, word_dim]
            word = self.word_embedd(input_word)
            # apply dropout on input
            word = self.dropout_in(word)
            src_encoding = word

        if self.use_bert:
            bert_inputs_ids, bert_input_mask, bert_out_positions = input_bert
            outputs = self.bert(bert_inputs_ids, attention_mask=bert_input_mask)
            bert_out = outputs[0]
            bert_out = bert_out[:, 1:-1, :].contiguous()
            bert_out = bert_out[torch.arange(bert_out.size(0)).unsqueeze(-1), bert_out_positions]
            if src_encoding is None:
                src_encoding = bert_out
            else:
                src_encoding = torch.cat([src_encoding, bert_out], dim=2) if self.input_concat_embeds \
                    else src_encoding + bert_out

        if self.use_char:
            # [batch, length, char_length, char_dim]
            char = self.char_embedd(input_char)
            char_size = char.size()
            # first transform to [batch *length, char_length, char_dim]
            # then transpose to [batch * length, char_dim, char_length]
            char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
            # put into cnn [batch*length, char_filters, char_length]
            # then put into maxpooling [batch * length, char_filters]
            char, _ = self.conv1d(char).max(dim=2)
            # reshape to [batch, length, char_filters]
            char = torch.tanh(char).view(char_size[0], char_size[1], -1)
            # apply dropout on input
            char = self.dropout_in(char)
            # concatenate word and char [batch, length, word_dim+char_filter]
            if src_encoding is None:
                src_encoding = char
            else:
                src_encoding = torch.cat([src_encoding, char], dim=2) if self.input_concat_embeds \
                    else src_encoding + char

        if self.use_pos:
            # [batch, length, pos_dim]
            pos = self.pos_embedd(input_pos)
            # # apply dropout on input
            # pos = self.dropout_in(pos)

            if src_encoding is None:
                src_encoding = pos
            else:
                src_encoding = torch.cat([src_encoding, pos], dim=2) if self.input_concat_embeds \
                    else src_encoding + pos

        assert src_encoding is not None

        if self.output_dim['position_dim'] > 0:
            position_encoding = torch.arange(start=0, end=src_encoding.size(1)).type(torch.LongTensor)
            if self.use_gpu:
                position_encoding = position_encoding.cuda()
            position_encoding = position_encoding.expand(*src_encoding.size()[:-1])
            position_encoding = self.position_embedding(position_encoding)
            src_encoding = torch.cat([src_encoding, position_encoding], dim=2) if self.input_concat_position \
                else src_encoding + position_encoding

        return src_encoding


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 rnn_mode,
                 hidden_size,
                 num_layers,
                 p_rnn=(0.33, 0.33),
                 pool_type=None,
                 encoder_type='RNN',
                 trans_hid_size=1028,
                 d_k=64,
                 d_v=64,
                 num_head=8,
                 enc_use_neg_dist=False,
                 enc_clip_dist=0,
                 use_gpu=False,
                 attn_on_rnn=False,
                 partitioned=False,
                 partition_type=None,
                 use_all_layers=False):
        super(Encoder, self).__init__()

        self.encoder_type = encoder_type
        self.attn_on_rnn = attn_on_rnn
        self.use_gpu = use_gpu
        self.pool_type = pool_type
        self.num_layers = num_layers
        self.use_all_layers = use_all_layers

        self.output_dim = 0
        self.hidden_dim = 0

        if self.encoder_type == 'RNN':

            if rnn_mode == 'RNN':
                RNN = VarMaskedRNN
            elif rnn_mode == 'LSTM':
                RNN = VarMaskedLSTM
            elif rnn_mode == 'FastLSTM':
                RNN = VarMaskedFastLSTM
            elif rnn_mode == 'GRU':
                RNN = VarMaskedGRU
            else:
                raise ValueError('Unknown RNN mode: %s' % rnn_mode)

            self.encoder = RNN(input_dim['total'],
                               hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True,
                               dropout=p_rnn)

            self.output_dim = 2 * hidden_size
            self.hidden_dim = 2 * hidden_size

            if self.attn_on_rnn:
                self.linear_query = nn.Linear(self.output_dim, self.output_dim)
                self.linear_key = nn.Linear(self.output_dim, self.output_dim)
                self.output_dim = self.output_dim * 2

        elif self.encoder_type == 'Transformer':
            d_content = None
            if partitioned:
                if partition_type == 'content-position':
                    assert input_dim['position_dim'] > 0
                    d_content = input_dim['total'] - input_dim['position_dim']
                    assert d_content > 0
                elif partition_type == 'lexical-delexical':
                    assert input_dim['pos_dim'] > 0
                    d_content = input_dim['total'] - input_dim['pos_dim']
                    assert d_content > 0
                else:
                    assert False

            self.encoder = TransformerEncoder(num_layers,
                                              d_model=input_dim['total'],
                                              heads=num_head,
                                              d_ff=trans_hid_size,
                                              d_k=d_k,
                                              d_v=d_v,
                                              attn_drop=p_rnn[0],
                                              relu_drop=p_rnn[1],
                                              res_drop=p_rnn[2],
                                              clip_dist=enc_clip_dist,
                                              use_neg_dist=enc_use_neg_dist,
                                              d_content=d_content)

            self.output_dim = input_dim['total']
            self.hidden_dim = input_dim['total']

            if self.use_all_layers:
                self.layer_weights = nn.Linear(self.output_dim, 1)

        else:
            raise NotImplementedError()

        if self.encoder_type == 'Transformer' and self.pool_type == 'weight':
            self.weight_pool = nn.Linear(self.output_dim, 1)

    def _apply_pooling(self, input):
        if self.pool_type == 'mean':
            temp_hidden = torch.sum(input, 1).div(input.size(1)).unsqueeze(1)
            return temp_hidden, torch.zeros_like(temp_hidden)
        elif self.pool_type == 'max':
            temp_hidden = torch.max(input, dim=1)[0].unsqueeze(1)
            return temp_hidden, torch.zeros_like(temp_hidden)
        elif self.pool_type == 'weight':
            att_weights = self.weight_pool(input).squeeze(2)  # B x S
            att_weights = F.softmax(att_weights, dim=-1)
            attn_rep = torch.bmm(input.transpose(1, 2), att_weights.unsqueeze(2)).squeeze(2)
            attn_rep = attn_rep.unsqueeze(1)
            return attn_rep, torch.zeros_like(attn_rep)
        else:
            return None

    def forward(self, input, mask, hx):

        if self.encoder_type == 'RNN':
            output, hn = self.encoder(input, mask, hx=hx)
            if isinstance(hn, tuple):
                hn = hn[0].transpose(0, 1), hn[1].transpose(0, 1)
            else:
                hn = hn.transpose(0, 1)

            if self.attn_on_rnn:
                # apply self-attention to modify context vectors
                Q = self.linear_query(output)  # B x S x h
                K = self.linear_key(output)  # B x S x h
                attn_weights = torch.bmm(Q, K.transpose(1, 2))  # B x S x S
                attn_weights = F.softmax(attn_weights, dim=-1)
                E = torch.ones_like(attn_weights)
                I = torch.eye(attn_weights.size(1)).expand_as(E)
                I = I.cuda() if self.use_gpu else I
                masked_encoding = torch.bmm(E - I, output)
                context = torch.bmm(attn_weights, masked_encoding)  # B x S x h
                output = torch.cat([output, context], dim=2)

        elif self.encoder_type == 'Transformer':
            output, _ = self.encoder(input)
            if self.use_all_layers:
                output = torch.stack(output, dim=2)  # B x S x nlayers x h
                layer_scores = self.layer_weights(output).squeeze(3)  # B x S x nlayers
                layer_scores = F.softmax(layer_scores, dim=-1)
                output = torch.matmul(output.transpose(2, 3), layer_scores.unsqueeze(3)).squeeze(3)  # B x S x h
            else:
                output = output[-1]
            hn = self._apply_pooling(output)

        else:
            raise NotImplementedError()

        return output, hn
