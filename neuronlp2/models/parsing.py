__author__ = 'max'

import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn import TreeCRF, VarMaskedGRU, VarMaskedRNN, VarMaskedLSTM, VarMaskedFastLSTM
from ..nn import SkipConnectFastLSTM, SkipConnectGRU, SkipConnectLSTM, SkipConnectRNN
from ..nn import BiAAttention, BiLinear
from neuronlp2.tasks import parser
from ..nn.modules.attention_aug import AugFeatureHelper, AugBiAAttention

from .encoder import Encoder, Embedder


class PriorOrder(Enum):
    DEPTH = 0
    INSIDE_OUT = 1
    LEFT2RIGTH = 2


class BiRecurrentConvBiAffine(nn.Module):
    def __init__(self,
                 word_dim,
                 num_words,
                 char_dim,
                 num_chars,
                 pos_dim,
                 num_pos,
                 num_filters,
                 kernel_size,
                 rnn_mode,
                 hidden_size,
                 num_layers,
                 num_labels,
                 arc_space,
                 type_space,
                 embedd_word=None,
                 embedd_char=None,
                 embedd_pos=None,
                 p_in=0.33,
                 p_out=0.33,
                 p_rnn=(0.33, 0.33),
                 biaffine=True,
                 pos=True,
                 char=True,
                 train_position=False,
                 encoder_type='RNN',
                 trans_hid_size=1028,
                 d_k=64, d_v=64, num_head=8,
                 enc_use_neg_dist=False,
                 enc_clip_dist=0,
                 position_dim=50,
                 max_sent_length=200,
                 use_gpu=False,
                 use_word_emb=True,
                 input_concat_embeds=False,
                 input_concat_position=False,
                 attn_on_rnn=False,
                 partitioned=False,
                 partition_type=None,
                 use_all_encoder_layers=False,
                 use_bert=False):
        super(BiRecurrentConvBiAffine, self).__init__()

        if encoder_type == 'Transformer':
            if enc_clip_dist > 0 and position_dim > 0:
                assert False, "both enc_clip_dist and position_dim can not be > 0"

        if partitioned:
            assert partition_type is not None
            if partition_type == 'content-position':
                assert position_dim > 0, "position dim must be > 0 when partitioned flag is on"
                assert enc_clip_dist == 0, "clipping distance must be 0 when partitioned flag is on"
                assert input_concat_position
            elif partition_type == 'lexical-delexical':
                assert not input_concat_position or position_dim == 0
                assert input_concat_embeds
                assert pos_dim > 0
                assert (word_dim + num_filters) > 0

        self.embedder = Embedder(word_dim,
                                 num_words,
                                 char_dim,
                                 num_chars,
                                 pos_dim,
                                 num_pos,
                                 num_filters,
                                 kernel_size,
                                 embedd_word=embedd_word,
                                 embedd_char=embedd_char,
                                 embedd_pos=embedd_pos,
                                 p_in=p_in,
                                 p_out=p_out,
                                 pos=pos,
                                 char=char,
                                 train_position=train_position,
                                 position_dim=position_dim,
                                 max_sent_length=max_sent_length,
                                 use_gpu=use_gpu,
                                 use_word_emb=use_word_emb,
                                 input_concat_embeds=input_concat_embeds,
                                 input_concat_position=input_concat_position,
                                 use_bert=use_bert)

        emb_out_dim = self.embedder.output_dim

        self.encoder = Encoder(emb_out_dim,
                               rnn_mode,
                               hidden_size,
                               num_layers,
                               p_rnn=p_rnn,
                               encoder_type=encoder_type,
                               trans_hid_size=trans_hid_size,
                               d_k=d_k,
                               d_v=d_v,
                               num_head=num_head,
                               enc_use_neg_dist=enc_use_neg_dist,
                               enc_clip_dist=enc_clip_dist,
                               use_gpu=use_gpu,
                               attn_on_rnn=attn_on_rnn,
                               partitioned=partitioned,
                               partition_type=partition_type,
                               use_all_layers=use_all_encoder_layers)

        enc_out_dim = self.encoder.output_dim

        self.num_labels = num_labels
        self.dropout_out = nn.Dropout2d(p=p_out)

        self.arc_h = nn.Linear(enc_out_dim, arc_space)
        self.arc_c = nn.Linear(enc_out_dim, arc_space)
        self.attention = BiAAttention(arc_space, arc_space, 1, biaffine=biaffine)

        self.type_h = nn.Linear(enc_out_dim, type_space)
        self.type_c = nn.Linear(enc_out_dim, type_space)
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)

    def forward(self,
                input_word,
                input_char,
                input_pos,
                input_bert=None,
                mask=None,
                length=None,
                hx=None):
        embeddings = self.embedder(input_word, input_char, input_pos, input_bert)
        encodings, hn = self.encoder(embeddings, mask=mask, hx=hx)
        encodings = self.dropout_out(encodings.transpose(1, 2)).transpose(1, 2)
        output = {'output': encodings, 'hn': hn}
        return output

    def forward_partial(self,
                        output,
                        mask=None,
                        length=None):
        # output size [batch, length, arc_space]
        arc_h = F.elu(self.arc_h(output))
        arc_c = F.elu(self.arc_c(output))

        # output size [batch, length, type_space]
        type_h = F.elu(self.type_h(output))
        type_c = F.elu(self.type_c(output))

        # apply dropout
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=1)
        type = torch.cat([type_h, type_c], dim=1)

        arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        arc_h, arc_c = arc.chunk(2, 1)

        type = self.dropout_out(type.transpose(1, 2)).transpose(1, 2)
        type_h, type_c = type.chunk(2, 1)
        type_h = type_h.contiguous()
        type_c = type_c.contiguous()

        arc = (arc_h, arc_c)
        type = (type_h, type_c)

        # [batch, length, length]
        out_arc = self.attention(arc[0], arc[1], mask_d=mask, mask_e=mask).squeeze(dim=1)
        return out_arc, type, mask, length

    def loss(self,
             output,
             heads,
             types,
             mask=None,
             length=None):
        # out_arc shape [batch, length, length]
        out_arc, out_type, mask, length = self.forward_partial(output, mask=mask, length=length)
        batch, max_len, _ = out_arc.size()

        if length is not None and heads.size(1) != mask.size(1):
            heads = heads[:, :max_len]
            types = types[:, :max_len]

        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type

        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(out_arc.data).long()
        # get vector for heads [batch, length, type_space],
        type_h = type_h[batch_index, heads.data.t()].transpose(0, 1).contiguous()
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask is not None:
            minus_inf = -1e8
            minus_mask = (1 - mask) * minus_inf
            out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # loss_arc shape [batch, length, length]
        loss_arc = F.log_softmax(out_arc, dim=1)
        # loss_type shape [batch, length, num_labels]
        loss_type = F.log_softmax(out_type, dim=2)

        # mask invalid position to 0 for sum loss
        if mask is not None:
            loss_arc = loss_arc * mask.unsqueeze(2) * mask.unsqueeze(1)
            loss_type = loss_type * mask.unsqueeze(2)
            # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
            num = mask.sum() - batch
        else:
            # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
            num = float(max_len - 1) * batch

        # first create index matrix [length, batch]
        child_index = torch.arange(0, max_len).view(max_len, 1).expand(max_len, batch)
        child_index = child_index.type_as(out_arc.data).long()
        # [length-1, batch]
        loss_arc = loss_arc[batch_index, heads.data.t(), child_index][1:]
        loss_type = loss_type[batch_index, child_index, types.data.t()][1:]

        return -loss_arc.sum() / num, -loss_type.sum() / num

    def compute_loss(self,
                     input_word,
                     input_char,
                     input_pos,
                     heads,
                     types,
                     input_bert=None,
                     mask=None,
                     length=None,
                     hx=None):
        # output from rnn [batch, length, tag_space]
        output = self.forward(input_word, input_char, input_pos,
                              input_bert=input_bert, mask=mask, length=length, hx=hx)
        output = output['output']
        arc_loss, type_loss = self.loss(output, heads, types, mask=None, length=None)
        return arc_loss, type_loss

    def _decode_types(self,
                      out_type,
                      heads,
                      leading_symbolic):
        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type
        batch, max_len, _ = type_h.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(type_h.data).long()
        # get vector for heads [batch, length, type_space],
        type_h = type_h[batch_index, heads.t()].transpose(0, 1).contiguous()
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)
        # remove the first #leading_symbolic types.
        out_type = out_type[:, :, leading_symbolic:]
        # compute the prediction of types [batch, length]
        _, types = out_type.max(dim=2)
        return types + leading_symbolic

    def decode(self,
               input_word,
               input_char,
               input_pos,
               input_bert=None,
               mask=None,
               length=None,
               hx=None,
               leading_symbolic=0):
        # out_arc shape [batch, length, length]
        output = self.forward(input_word, input_char, input_pos, input_bert=input_bert, mask=mask, length=length, hx=hx)
        output = output['output']

        out_arc, out_type, mask, length = self.forward_partial(output, mask=mask, length=length)

        out_arc = out_arc.data
        batch, max_len, _ = out_arc.size()
        # set diagonal elements to -inf
        out_arc = out_arc + torch.diag(out_arc.new(max_len).fill_(-np.inf))
        # set invalid positions to -inf
        if mask is not None:
            # minus_mask = (1 - mask.data).byte().view(batch, max_len, 1)
            minus_mask = (1 - mask.data).byte().unsqueeze(2)
            out_arc.masked_fill_(minus_mask, -np.inf)

        # compute naive predictions.
        # predition shape = [batch, length]
        _, heads = out_arc.max(dim=1)

        types = self._decode_types(out_type, heads, leading_symbolic)

        return heads.cpu().numpy(), types.data.cpu().numpy()

    def decode_mst(self,
                   input_word,
                   input_char,
                   input_pos,
                   input_bert=None,
                   mask=None,
                   length=None,
                   hx=None,
                   leading_symbolic=0):
        """
        Args:
            input_word: Tensor
                the word input tensor with shape = [batch, length]
            input_char: Tensor
                the character input tensor with shape = [batch, length, char_length]
            input_pos: Tensor
                the pos input tensor with shape = [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            length: Tensor or None
                the length tensor with shape = [batch]
            hx: Tensor or None
                the initial states of RNN
            leading_symbolic: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)
        Returns: (Tensor, Tensor)
                predicted heads and types.
        """
        # out_arc shape [batch, length, length]
        output = self.forward(input_word, input_char, input_pos, input_bert=input_bert, mask=mask, length=length, hx=hx)
        output = output['output']

        out_arc, out_type, mask, length = self.forward_partial(output, mask=mask, length=length)

        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type
        batch, max_len, type_space = type_h.size()

        # compute lengths
        if length is None:
            if mask is None:
                length = [max_len for _ in range(batch)]
            else:
                length = mask.data.sum(dim=1).long().cpu().numpy()

        type_h = type_h.unsqueeze(2).expand(batch, max_len, max_len, type_space).contiguous()
        type_c = type_c.unsqueeze(1).expand(batch, max_len, max_len, type_space).contiguous()
        # compute output for type [batch, length, length, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask is not None:
            minus_inf = -1e8
            minus_mask = (1 - mask) * minus_inf
            out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # loss_arc shape [batch, length, length]
        loss_arc = F.log_softmax(out_arc, dim=1)
        # loss_type shape [batch, length, length, num_labels]
        loss_type = F.log_softmax(out_type, dim=3).permute(0, 3, 1, 2)
        # [batch, num_labels, length, length]
        energy = torch.exp(loss_arc.unsqueeze(1) + loss_type)

        return parser.decode_MST(energy.data.cpu().numpy(), length, leading_symbolic=leading_symbolic, labeled=True)


class StackPtrNet(nn.Module):
    def __init__(self,
                 word_dim,
                 num_words,
                 char_dim,
                 num_chars,
                 pos_dim,
                 num_pos,
                 num_filters,
                 kernel_size,
                 rnn_mode,
                 input_size_decoder,
                 hidden_size,
                 encoder_layers,
                 decoder_layers,
                 num_labels,
                 arc_space,
                 type_space,
                 pool_type,
                 num_head,
                 max_sent_length,
                 trans_hid_size,
                 d_k,
                 d_v,
                 train_position=False,
                 embedd_word=None,
                 embedd_char=None,
                 embedd_pos=None,
                 p_in=0.33,
                 p_out=0.33,
                 p_rnn=(0.33, 0.33),
                 biaffine=True,
                 use_word_emb=True,
                 pos=True,
                 char=True,
                 prior_order='inside_out',
                 skipConnect=False,
                 grandPar=False,
                 sibling=False,
                 use_gpu=False,
                 encoder_type='RNN',
                 dec_max_dist=0,
                 dec_use_neg_dist=False,
                 dec_use_encoder_pos=False,
                 dec_use_decoder_pos=False,
                 dec_dim_feature=10,
                 dec_drop_f_embed=0.,
                 attn_on_rnn=False,
                 enc_clip_dist=0,
                 enc_use_neg_dist=False,
                 partitioned=False,
                 input_concat_embeds=False,
                 input_concat_position=False,
                 position_dim=50,
                 partition_type=None,
                 use_all_encoder_layers=False,
                 use_bert=False):

        super(StackPtrNet, self).__init__()

        if encoder_type == 'Transformer':
            if enc_clip_dist > 0 and position_dim > 0:
                assert False, "both enc_clip_dist and position_dim can not be > 0"

        if partitioned:
            assert partition_type is not None
            if partition_type == 'content-position':
                assert position_dim > 0, "position dim must be > 0 when partitioned flag is on"
                assert enc_clip_dist == 0, "clipping distance must be 0 when partitioned flag is on"
                assert input_concat_position
            elif partition_type == 'lexical-delexical':
                assert not input_concat_position or position_dim == 0
                assert input_concat_embeds
                assert pos_dim > 0
                assert (word_dim + num_filters) > 0

        self.embedder = Embedder(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size,
                                 embedd_word=embedd_word, embedd_char=embedd_char, embedd_pos=embedd_pos, p_in=p_in,
                                 p_out=p_out, pos=pos, char=char, train_position=train_position,
                                 position_dim=position_dim, max_sent_length=max_sent_length, use_gpu=use_gpu,
                                 use_word_emb=use_word_emb, input_concat_embeds=input_concat_embeds,
                                 input_concat_position=input_concat_position, use_bert=use_bert)

        emb_out_dim = self.embedder.output_dim

        self.encoder_type = encoder_type
        self.encoder = Encoder(emb_out_dim, rnn_mode, hidden_size, encoder_layers, p_rnn=p_rnn,
                               encoder_type=encoder_type, trans_hid_size=trans_hid_size, d_k=d_k, d_v=d_v,
                               num_head=num_head, enc_use_neg_dist=enc_use_neg_dist, enc_clip_dist=enc_clip_dist,
                               use_gpu=use_gpu, attn_on_rnn=attn_on_rnn, partitioned=partitioned,
                               partition_type=partition_type, pool_type=pool_type,
                               use_all_layers=use_all_encoder_layers)

        enc_out_dim = self.encoder.output_dim
        enc_hid_dim = self.encoder.hidden_dim

        # decoder setup

        self.skipConnect = skipConnect
        self.grandPar = grandPar
        self.sibling = sibling
        self.num_labels = num_labels
        self.dropout_out = nn.Dropout2d(p=p_out)

        if rnn_mode == 'RNN':
            RNN_DECODER = SkipConnectRNN if skipConnect else VarMaskedRNN
        elif rnn_mode == 'LSTM':
            RNN_DECODER = SkipConnectLSTM if skipConnect else VarMaskedLSTM
        elif rnn_mode == 'FastLSTM':
            RNN_DECODER = SkipConnectFastLSTM if skipConnect else VarMaskedFastLSTM
        elif rnn_mode == 'GRU':
            RNN_DECODER = SkipConnectGRU if skipConnect else VarMaskedGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        if prior_order in ['deep_first', 'shallow_first']:
            self.prior_order = PriorOrder.DEPTH
        elif prior_order == 'inside_out':
            self.prior_order = PriorOrder.INSIDE_OUT
        elif prior_order == 'left2right':
            self.prior_order = PriorOrder.LEFT2RIGTH
        else:
            raise ValueError('Unknown prior order: %s' % prior_order)

        dim_dec = input_size_decoder
        self.src_dense = nn.Linear(enc_out_dim, dim_dec)

        self.decoder_layers = decoder_layers
        drop_rnn = p_rnn[:2] if len(p_rnn) > 2 else p_rnn
        self.decoder = RNN_DECODER(dim_dec, hidden_size,
                                   num_layers=decoder_layers, batch_first=True,
                                   bidirectional=False, dropout=drop_rnn)

        self.hx_dense = nn.Linear(enc_hid_dim, hidden_size)
        self.arc_h = nn.Linear(hidden_size, arc_space)  # arc dense for decoder
        self.arc_c = nn.Linear(enc_out_dim, arc_space)  # arc dense for encoder

        self.attention_helper = AugFeatureHelper(dec_max_dist,
                                                 dec_use_neg_dist,
                                                 num_pos,
                                                 dec_use_encoder_pos,
                                                 dec_use_decoder_pos)
        self.attention = AugBiAAttention(arc_space,
                                         arc_space,
                                         1,
                                         num_features=self.attention_helper.get_num_features(),
                                         dim_feature=dec_dim_feature,
                                         drop_f_embed=dec_drop_f_embed,
                                         biaffine=biaffine)

        self.type_h = nn.Linear(hidden_size, type_space)  # type dense for decoder
        self.type_c = nn.Linear(enc_out_dim, type_space)  # type dense for encoder
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)

    def forward(self,
                input_word,
                input_char,
                input_pos,
                input_bert=None,
                mask_e=None,
                length_e=None,
                hx=None):
        src_encoding = self.embedder(input_word, input_char, input_pos, input_bert)
        encodings, hn = self.encoder(src_encoding, mask=mask_e, hx=hx)
        encodings = self.dropout_out(encodings.transpose(1, 2)).transpose(1, 2)
        output = {'output': encodings}
        return output, hn, mask_e, length_e

    def _get_decoder_output(self,
                            output_enc,
                            heads,
                            heads_stack,
                            siblings,
                            hx,
                            mask_d=None,
                            length_d=None):
        batch, _, _ = output_enc.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(output_enc).long()
        # get vector for heads [batch, length_decoder, input_dim],
        src_encoding = output_enc[batch_index, heads_stack.t()].transpose(0, 1)

        if self.sibling:
            # [batch, length_decoder, hidden_size * 2]
            mask_sibs = siblings.ne(0).float().unsqueeze(2)
            output_enc_sibling = output_enc[batch_index, siblings.t()].transpose(0, 1) * mask_sibs
            src_encoding = src_encoding + output_enc_sibling

        if self.grandPar:
            # [length_decoder, batch]
            gpars = heads[batch_index, heads_stack.t()]
            # [batch, length_decoder, hidden_size * 2]
            output_enc_gpar = output_enc[batch_index, gpars].transpose(0, 1)
            src_encoding = src_encoding + output_enc_gpar

        # transform to decoder input
        # [batch, length_decoder, dec_dim]
        src_encoding = F.elu(self.src_dense(src_encoding))

        # output from rnn [batch, length, hidden_size]
        output, hn = self.decoder(src_encoding, mask_d, hx=hx)

        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn, mask_d, length_d

    def _get_decoder_output_with_skip_connect(self,
                                              output_enc,
                                              heads,
                                              heads_stack,
                                              siblings,
                                              skip_connect,
                                              hx,
                                              mask_d=None,
                                              length_d=None):
        batch, _, _ = output_enc.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(output_enc).long()
        # get vector for heads [batch, length_decoder, input_dim],
        src_encoding = output_enc[batch_index, heads_stack.t()].transpose(0, 1)

        if self.sibling:
            # [batch, length_decoder, hidden_size * 2]
            mask_sibs = siblings.ne(0).float().unsqueeze(2)
            output_enc_sibling = output_enc[batch_index, siblings.t()].transpose(0, 1) * mask_sibs
            src_encoding = src_encoding + output_enc_sibling

        if self.grandPar:
            # [length_decoder, batch]
            gpars = heads[batch_index, heads_stack.t()]
            # [batch, length_decoder, hidden_size * 2]
            output_enc_gpar = output_enc[batch_index, gpars].transpose(0, 1)
            src_encoding = src_encoding + output_enc_gpar

        # transform to decoder input
        # [batch, length_decoder, dec_dim]
        src_encoding = F.elu(self.src_dense(src_encoding))

        # output from rnn [batch, length, hidden_size]
        output, hn = self.decoder(src_encoding, skip_connect, mask_d, hx=hx)

        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn, mask_d, length_d

    def _transform_decoder_init_state(self, hn):
        if isinstance(hn, tuple):
            if self.encoder_type == 'RNN':
                hn, cn = hn
                # take the last layers
                # [2, batch, hidden_size]
                cn = cn[-2:]
                # hn [2, batch, hidden_size]
                _, batch, hidden_size = cn.size()
                # first convert cn t0 [batch, 2, hidden_size]
                cn = cn.transpose(0, 1).contiguous()
                # then view to [batch, 1, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
                cn = cn.view(batch, 1, 2 * hidden_size).transpose(0, 1)
            else:
                cn, hn = hn

            # take hx_dense to [1, batch, hidden_size]
            cn = self.hx_dense(cn)
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                cn = torch.cat([cn, cn.new(self.decoder_layers - 1, batch, hidden_size).zero_()], dim=0)
            # hn is tanh(cn)
            hn = torch.tanh(cn)
            hn = (hn, cn)
        else:
            # take the last layers
            # [2, batch, hidden_size]
            hn = hn[-2:]
            # hn [2, batch, hidden_size]
            _, batch, hidden_size = hn.size()
            # first convert hn t0 [batch, 2, hidden_size]
            hn = hn.transpose(0, 1).contiguous()
            # then view to [batch, 1, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            if self.encoder_type == 'RNN':
                hn = hn.view(batch, 1, 2 * hidden_size).transpose(0, 1)
            else:
                hn = hn.view(batch, 1, hidden_size).transpose(0, 1)
            # take hx_dense to [1, batch, hidden_size]
            hn = torch.tanh(self.hx_dense(hn))
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                hn = torch.cat([hn, hn.new(self.decoder_layers - 1, batch, hidden_size).zero_()], dim=0)
        return hn

    def compute_loss(self,
                     output_enc,
                     hn,
                     mask_e,
                     input_pos,
                     heads,
                     stacked_heads,
                     children,
                     siblings,
                     stacked_types,
                     label_smooth,
                     skip_connect=None,
                     mask_d=None,
                     length_d=None):
        # output size [batch, length_encoder, arc_space]
        arc_c = F.elu(self.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        type_c = F.elu(self.type_c(output_enc))

        # transform hn to [decoder_layers, batch, hidden_size]
        hn = self._transform_decoder_init_state(hn)

        # output from decoder [batch, length_decoder, tag_space]
        if self.skipConnect:
            output_dec, _, mask_d, _ = self._get_decoder_output_with_skip_connect(output_enc,
                                                                                  heads,
                                                                                  stacked_heads,
                                                                                  siblings,
                                                                                  skip_connect,
                                                                                  hn,
                                                                                  mask_d=mask_d,
                                                                                  length_d=length_d)
        else:
            output_dec, _, mask_d, _ = self._get_decoder_output(output_enc,
                                                                heads,
                                                                stacked_heads,
                                                                siblings,
                                                                hn,
                                                                mask_d=mask_d,
                                                                length_d=length_d)

        # output size [batch, length_decoder, arc_space]
        arc_h = F.elu(self.arc_h(output_dec))
        type_h = F.elu(self.type_h(output_dec))

        _, max_len_d, _ = arc_h.size()
        if mask_d is not None and children.size(1) != mask_d.size(1):
            stacked_heads = stacked_heads[:, :max_len_d]
            children = children[:, :max_len_d]
            stacked_types = stacked_types[:, :max_len_d]

        # apply dropout
        # [batch, length_decoder, dim] + [batch, length_encoder, dim] --> [batch, length_decoder + length_encoder, dim]
        arc = self.dropout_out(torch.cat([arc_h, arc_c], dim=1).transpose(1, 2)).transpose(1, 2)
        arc_h = arc[:, :max_len_d]
        arc_c = arc[:, max_len_d:]

        type = self.dropout_out(torch.cat([type_h, type_c], dim=1).transpose(1, 2)).transpose(1, 2)
        type_h = type[:, :max_len_d].contiguous()
        type_c = type[:, max_len_d:]

        # [batch, length_decoder, length_encoder]
        if self.attention.use_features:
            batch, max_len_e, _ = arc_c.size()
            batch_index = torch.arange(0, batch).type_as(arc_c).long()
            child_pos = input_pos  # [batch, len-e]
            head_pos = input_pos[batch_index, stacked_heads.t()].transpose(0, 1)  # [batch, len-d]
            child_position_idxes = torch.arange(max_len_e).type_as(arc_c).long().expand(batch, -1).unsqueeze(
                -2)  # [batch, 1, len-e]
            head_position_idxes = stacked_heads.unsqueeze(-1)  # [batch, len-d, 1]
            raw_distances = head_position_idxes.expand(-1, -1, child_position_idxes.size()[
                -1]) - child_position_idxes.expand(-1, head_position_idxes.size()[-2], -1)  # [batch, len-d, len-e]
            input_features = self.attention_helper.get_final_features(raw_distances, child_pos, head_pos)
        else:
            input_features = None

        out_arc = self.attention(arc_h, arc_c, input_features=input_features, mask_d=mask_d, mask_e=mask_e).squeeze(
            dim=1)

        batch, max_len_e, _ = arc_c.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(arc_c).long()
        # get vector for heads [batch, length_decoder, type_space],
        type_c = type_c[batch_index, children.t()].transpose(0, 1).contiguous()
        # compute output for type [batch, length_decoder, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask_e is not None:
            minus_inf = -1e8
            minus_mask_d = (1 - mask_d) * minus_inf
            minus_mask_e = (1 - mask_e) * minus_inf
            out_arc = out_arc + minus_mask_d.unsqueeze(2) + minus_mask_e.unsqueeze(1)

        # [batch, length_decoder, length_encoder]
        loss_arc = F.log_softmax(out_arc, dim=2)
        # [batch, length_decoder, num_labels]
        loss_type = F.log_softmax(out_type, dim=2)

        # compute coverage loss
        # [batch, length_decoder, length_encoder]
        coverage = torch.exp(loss_arc).cumsum(dim=1)

        # get leaf and non-leaf mask
        # shape = [batch, length_decoder]
        mask_leaf = torch.eq(children, stacked_heads).float()
        mask_non_leaf = (1.0 - mask_leaf)

        # mask invalid position to 0 for sum loss
        if mask_e is not None:
            loss_arc = loss_arc * mask_d.unsqueeze(2) * mask_e.unsqueeze(1)
            coverage = coverage * mask_d.unsqueeze(2) * mask_e.unsqueeze(1)
            loss_type = loss_type * mask_d.unsqueeze(2)
            mask_leaf = mask_leaf * mask_d
            mask_non_leaf = mask_non_leaf * mask_d

            # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
            num_leaf = mask_leaf.sum()
            num_non_leaf = mask_non_leaf.sum()
        else:
            # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
            num_leaf = max_len_e
            num_non_leaf = max_len_e - 1

        # first create index matrix [length, batch]
        head_index = torch.arange(0, max_len_d).view(max_len_d, 1).expand(max_len_d, batch)
        head_index = head_index.type_as(out_arc).long()
        # [batch, length_decoder]
        if 0.0 < label_smooth < 1.0 - 1e-4:
            # label smoothing
            loss_arc1 = loss_arc[batch_index, head_index, children.t()].transpose(0, 1)
            loss_arc2 = loss_arc.sum(dim=2) / mask_e.sum(dim=1).unsqueeze(1)
            loss_arc = loss_arc1 * label_smooth + loss_arc2 * (1 - label_smooth)

            loss_type1 = loss_type[batch_index, head_index, stacked_types.t()].transpose(0, 1)
            loss_type2 = loss_type.sum(dim=2) / self.num_labels
            loss_type = loss_type1 * label_smooth + loss_type2 * (1 - label_smooth)
        else:
            loss_arc = loss_arc[batch_index, head_index, children.t()].transpose(0, 1)
            loss_type = loss_type[batch_index, head_index, stacked_types.t()].transpose(0, 1)

        loss_arc_leaf = loss_arc * mask_leaf
        loss_arc_non_leaf = loss_arc * mask_non_leaf

        loss_type_leaf = loss_type * mask_leaf
        loss_type_non_leaf = loss_type * mask_non_leaf

        loss_cov = (coverage - 2.0).clamp(min=0.)

        return -loss_arc_leaf.sum() / num_leaf, -loss_arc_non_leaf.sum() / num_non_leaf, \
               -loss_type_leaf.sum() / num_leaf, -loss_type_non_leaf.sum() / num_non_leaf, \
               loss_cov.sum() / (num_leaf + num_non_leaf), num_leaf, num_non_leaf

    def loss(self,
             input_word,
             input_char,
             input_pos,
             heads,
             stacked_heads,
             children,
             siblings,
             stacked_types,
             label_smooth,
             input_bert=None,
             skip_connect=None,
             mask_e=None,
             length_e=None,
             mask_d=None,
             length_d=None,
             hx=None):
        # output from encoder [batch, length_encoder, tag_space]
        output_enc, hn, mask_e, _ = self.forward(input_word,
                                                 input_char,
                                                 input_pos,
                                                 input_bert=input_bert,
                                                 mask_e=mask_e,
                                                 length_e=length_e,
                                                 hx=hx)
        output_enc = output_enc['output']
        hn = (hn[0].transpose(0, 1), hn[1].transpose(0, 1)) if isinstance(hn, tuple) else hn.transpose(0, 1)

        return self.compute_loss(output_enc,
                                 hn,
                                 mask_e,
                                 input_pos,
                                 heads,
                                 stacked_heads,
                                 children,
                                 siblings,
                                 stacked_types,
                                 label_smooth,
                                 skip_connect=skip_connect,
                                 mask_d=mask_d,
                                 length_d=length_d)

    def _decode_per_sentence(self,
                             output_enc,
                             arc_c,
                             type_c,
                             hx,
                             length,
                             beam,
                             ordered,
                             leading_symbolic,
                             input_pos):
        def valid_hyp(base_id, child_id, head):
            if constraints[base_id, child_id]:
                return False
            elif not ordered or self.prior_order == PriorOrder.DEPTH or child_orders[base_id, head] == 0:
                return True
            elif self.prior_order == PriorOrder.LEFT2RIGTH:
                return child_id > child_orders[base_id, head]
            else:
                if child_id < head:
                    return child_id < child_orders[base_id, head] < head
                else:
                    return child_id > child_orders[base_id, head]

        # output_enc [length, hidden_size * 2]
        # arc_c [length, arc_space]
        # type_c [length, type_space]
        # hx [decoder_layers, hidden_size]
        if length is not None:
            output_enc = output_enc[:length]
            arc_c = arc_c[:length]
            type_c = type_c[:length]
            input_pos = input_pos[:length]  # input_pos: [length]
        else:
            length = output_enc.size(0)

        # [decoder_layers, 1, hidden_size]
        # hack to handle LSTM
        if isinstance(hx, tuple):
            hx, cx = hx
            hx = hx.unsqueeze(1)
            cx = cx.unsqueeze(1)
            h0 = hx
            hx = (hx, cx)
        else:
            hx = hx.unsqueeze(1)
            h0 = hx

        stacked_heads = [[0] for _ in range(beam)]
        grand_parents = [[0] for _ in range(beam)] if self.grandPar else None
        siblings = [[0] for _ in range(beam)] if self.sibling else None
        skip_connects = [[h0] for _ in range(beam)] if self.skipConnect else None
        children = torch.zeros(beam, 2 * length.item() - 1).type_as(output_enc).long()
        stacked_types = children.new(children.size()).zero_()
        hypothesis_scores = output_enc.new(beam).zero_()
        constraints = np.zeros([beam, length], dtype=np.bool)
        constraints[:, 0] = True
        child_orders = np.zeros([beam, length], dtype=np.int64)

        # temporal tensors for each step.
        new_stacked_heads = [[] for _ in range(beam)]
        new_grand_parents = [[] for _ in range(beam)] if self.grandPar else None
        new_siblings = [[] for _ in range(beam)] if self.sibling else None
        new_skip_connects = [[] for _ in range(beam)] if self.skipConnect else None
        new_children = children.new(children.size()).zero_()
        new_stacked_types = stacked_types.new(stacked_types.size()).zero_()
        num_hyp = 1
        num_step = 2 * length - 1
        for t in range(num_step):
            # [num_hyp]
            heads = torch.LongTensor([stacked_heads[i][-1] for i in range(num_hyp)]).type_as(children)
            gpars = torch.LongTensor([grand_parents[i][-1] for i in range(num_hyp)]).type_as(
                children) if self.grandPar else None
            sibs = torch.LongTensor([siblings[i].pop() for i in range(num_hyp)]).type_as(
                children) if self.sibling else None

            # [decoder_layers, num_hyp, hidden_size]
            hs = torch.cat([skip_connects[i].pop() for i in range(num_hyp)], dim=1) if self.skipConnect else None

            # [num_hyp, hidden_size * 2]
            src_encoding = output_enc[heads]

            if self.sibling:
                mask_sibs = sibs.ne(0).float().unsqueeze(1)
                output_enc_sibling = output_enc[sibs] * mask_sibs
                src_encoding = src_encoding + output_enc_sibling

            if self.grandPar:
                output_enc_gpar = output_enc[gpars]
                src_encoding = src_encoding + output_enc_gpar

            # transform to decoder input
            # [num_hyp, dec_dim]
            src_encoding = F.elu(self.src_dense(src_encoding))

            # output [num_hyp, hidden_size]
            # hx [decoder_layer, num_hyp, hidden_size]
            output_dec, hx = self.decoder.step(src_encoding, hx=hx, hs=hs) if self.skipConnect else self.decoder.step(
                src_encoding, hx=hx)

            # arc_h size [num_hyp, 1, arc_space]
            arc_h = F.elu(self.arc_h(output_dec.unsqueeze(1)))
            # type_h size [num_hyp, type_space]
            type_h = F.elu(self.type_h(output_dec))

            # [num_hyp, length_encoder]
            if self.attention.use_features:
                # len-d == 1
                child_pos = input_pos.expand(num_hyp, *input_pos.size())  # [num-hyp, len-e]
                head_pos = input_pos[heads].unsqueeze(-1)  # [num-hyp, 1]
                child_position_idxes = torch.arange(length).type_as(input_pos).long().expand(num_hyp,
                                                                                             -1).unsqueeze(
                    -2)  # [batch, 1, len-e]
                head_position_idxes = heads.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
                raw_distances = head_position_idxes - child_position_idxes  # [batch, 1, len-e]
                input_features = self.attention_helper.get_final_features(raw_distances, child_pos, head_pos)
            else:
                input_features = None
            out_arc = self.attention(arc_h, arc_c.expand(num_hyp, *arc_c.size()),
                                     input_features=input_features).squeeze(dim=1).squeeze(dim=1)

            # [num_hyp, length_encoder]
            hyp_scores = F.log_softmax(out_arc, dim=1)

            new_hypothesis_scores = hypothesis_scores[:num_hyp].unsqueeze(1) + hyp_scores
            # [num_hyp * length_encoder]
            new_hypothesis_scores, hyp_index = torch.sort(new_hypothesis_scores.view(-1), dim=0, descending=True)
            base_index = hyp_index / length
            child_index = hyp_index % length

            cc = 0
            ids = []
            new_constraints = np.zeros([beam, length], dtype=np.bool)
            new_child_orders = np.zeros([beam, length], dtype=np.int64)
            for id in range(num_hyp * length):
                base_id = base_index[id]
                child_id = child_index[id].item()
                head = heads[base_id].item()
                new_hyp_score = new_hypothesis_scores[id]
                if child_id == head:
                    assert constraints[base_id, child_id], 'constrains error: %d, %d' % (base_id, child_id)
                    if head != 0 or t + 1 == num_step:
                        new_constraints[cc] = constraints[base_id]
                        new_child_orders[cc] = child_orders[base_id]

                        new_stacked_heads[cc] = [stacked_heads[base_id][i] for i in range(len(stacked_heads[base_id]))]
                        new_stacked_heads[cc].pop()

                        if self.grandPar:
                            new_grand_parents[cc] = [grand_parents[base_id][i] for i in
                                                     range(len(grand_parents[base_id]))]
                            new_grand_parents[cc].pop()

                        if self.sibling:
                            new_siblings[cc] = [siblings[base_id][i] for i in range(len(siblings[base_id]))]

                        if self.skipConnect:
                            new_skip_connects[cc] = [skip_connects[base_id][i] for i in
                                                     range(len(skip_connects[base_id]))]

                        new_children[cc] = children[base_id]
                        new_children[cc, t] = child_id

                        hypothesis_scores[cc] = new_hyp_score
                        ids.append(id)
                        cc += 1
                elif valid_hyp(base_id, child_id, head):
                    new_constraints[cc] = constraints[base_id]
                    new_constraints[cc, child_id] = True

                    new_child_orders[cc] = child_orders[base_id]
                    new_child_orders[cc, head] = child_id

                    new_stacked_heads[cc] = [stacked_heads[base_id][i] for i in range(len(stacked_heads[base_id]))]
                    new_stacked_heads[cc].append(child_id)

                    if self.grandPar:
                        new_grand_parents[cc] = [grand_parents[base_id][i] for i in range(len(grand_parents[base_id]))]
                        new_grand_parents[cc].append(head)

                    if self.sibling:
                        new_siblings[cc] = [siblings[base_id][i] for i in range(len(siblings[base_id]))]
                        new_siblings[cc].append(child_id)
                        new_siblings[cc].append(0)

                    if self.skipConnect:
                        new_skip_connects[cc] = [skip_connects[base_id][i] for i in range(len(skip_connects[base_id]))]
                        # hack to handle LSTM
                        if isinstance(hx, tuple):
                            new_skip_connects[cc].append(hx[0][:, base_id, :].unsqueeze(1))
                        else:
                            new_skip_connects[cc].append(hx[:, base_id, :].unsqueeze(1))
                        new_skip_connects[cc].append(h0)

                    new_children[cc] = children[base_id]
                    new_children[cc, t] = child_id

                    hypothesis_scores[cc] = new_hyp_score
                    ids.append(id)
                    cc += 1

                if cc == beam:
                    break

            # [num_hyp]
            num_hyp = len(ids)
            if num_hyp == 0:
                return None
            elif num_hyp == 1:
                index = base_index.new(1).fill_(ids[0])
            else:
                index = torch.from_numpy(np.array(ids)).type_as(base_index)
            base_index = base_index[index]
            child_index = child_index[index]

            # predict types for new hypotheses
            # compute output for type [num_hyp, num_labels]
            out_type = self.bilinear(type_h[base_index], type_c[child_index])
            hyp_type_scores = F.log_softmax(out_type, dim=1)
            # compute the prediction of types [num_hyp]
            hyp_type_scores, hyp_types = hyp_type_scores.max(dim=1)
            hypothesis_scores[:num_hyp] = hypothesis_scores[:num_hyp] + hyp_type_scores

            for i in range(num_hyp):
                base_id = base_index[i]
                new_stacked_types[i] = stacked_types[base_id]
                new_stacked_types[i, t] = hyp_types[i]

            stacked_heads = [[new_stacked_heads[i][j] for j in range(len(new_stacked_heads[i]))] for i in
                             range(num_hyp)]
            if self.grandPar:
                grand_parents = [[new_grand_parents[i][j] for j in range(len(new_grand_parents[i]))] for i in
                                 range(num_hyp)]
            if self.sibling:
                siblings = [[new_siblings[i][j] for j in range(len(new_siblings[i]))] for i in range(num_hyp)]
            if self.skipConnect:
                skip_connects = [[new_skip_connects[i][j] for j in range(len(new_skip_connects[i]))] for i in
                                 range(num_hyp)]
            constraints = new_constraints
            child_orders = new_child_orders
            children.copy_(new_children)
            stacked_types.copy_(new_stacked_types)
            # hx [decoder_layers, num_hyp, hidden_size]
            # hack to handle LSTM
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx[:, base_index, :]
                cx = cx[:, base_index, :]
                hx = (hx, cx)
            else:
                hx = hx[:, base_index, :]

        children = children.cpu().numpy()[0]
        stacked_types = stacked_types.cpu().numpy()[0]
        heads = np.zeros(length, dtype=np.int32)
        types = np.zeros(length, dtype=np.int32)
        stack = [0]
        for i in range(num_step):
            head = stack[-1]
            child = children[i]
            type = stacked_types[i]
            if child != head:
                heads[child] = head
                types[child] = type
                stack.append(child)
            else:
                stacked_types[i] = 0
                stack.pop()

        return heads, types, length, children, stacked_types

    def decode(self,
               input_word,
               input_char,
               input_pos,
               input_bert=None,
               mask=None,
               length=None,
               hx=None,
               beam=1,
               leading_symbolic=0,
               ordered=True):
        # reset noise for decoder
        self.decoder.reset_noise(0)

        # output from encoder [batch, length_encoder, tag_space]
        # output_enc [batch, length, input_size]
        # arc_c [batch, length, arc_space]
        # type_c [batch, length, type_space]
        # hn [num_direction, batch, hidden_size]
        output_enc, hn, mask, length = self.forward(input_word, input_char, input_pos, input_bert=input_bert,
                                                    mask_e=mask, length_e=length, hx=hx)
        output_enc = output_enc['output']
        hn = (hn[0].transpose(0, 1), hn[1].transpose(0, 1)) if isinstance(hn, tuple) else hn.transpose(0, 1)

        # output size [batch, length_encoder, arc_space]
        arc_c = F.elu(self.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        type_c = F.elu(self.type_c(output_enc))
        # [decoder_layers, batch, hidden_size
        hn = self._transform_decoder_init_state(hn)
        batch, max_len_e, _ = output_enc.size()

        heads = np.zeros([batch, max_len_e], dtype=np.int32)
        types = np.zeros([batch, max_len_e], dtype=np.int32)

        children = np.zeros([batch, 2 * max_len_e - 1], dtype=np.int32)
        stack_types = np.zeros([batch, 2 * max_len_e - 1], dtype=np.int32)

        for b in range(batch):
            sent_len = None if length is None else length[b]
            # hack to handle LSTM
            if isinstance(hn, tuple):
                hx, cx = hn
                hx = hx[:, b, :].contiguous()
                cx = cx[:, b, :].contiguous()
                hx = (hx, cx)
            else:
                hx = hn[:, b, :].contiguous()

            preds = self._decode_per_sentence(output_enc[b], arc_c[b], type_c[b], hx, sent_len, beam, ordered,
                                              leading_symbolic, input_pos[b])
            if preds is None:
                preds = self._decode_per_sentence(output_enc[b], arc_c[b], type_c[b], hx, sent_len, beam, False,
                                                  leading_symbolic, input_pos[b])
            hids, tids, sent_len, chids, stids = preds
            heads[b, :sent_len] = hids
            types[b, :sent_len] = tids

            children[b, :2 * sent_len - 1] = chids
            stack_types[b, :2 * sent_len - 1] = stids

        return heads, types, children, stack_types
