"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from .util_class import LayerNorm
from .multi_head_attn import MultiHeadedAttention
from .encoder import EncoderBase
from .position_ffn import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self,
                 d_model,
                 heads,
                 d_ff,
                 d_k,
                 d_v,
                 attn_drop,
                 relu_drop,
                 res_drop,
                 clip_dist=0,
                 use_neg_dist=False,
                 d_content=None):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(heads, d_model, d_k, d_v,
                                              dropout=attn_drop,
                                              clip_dist=clip_dist,
                                              use_neg_dist=use_neg_dist,
                                              d_content=d_content)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff,
                                                    d_content=d_content,
                                                    relu_drop=relu_drop,
                                                    res_drop=res_drop)
        self.dropout = nn.Dropout(res_drop)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.
        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        context, attn_per_head = self.self_attn(inputs, inputs, inputs, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out), attn_per_head


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    Returns:
        (`FloatTensor`, `FloatTensor`):
        * embeddings `[batch_size x src_len x model_dim]`
        * memory_bank `[batch_size x src_len x model_dim]`
    """

    def __init__(self,
                 num_layers,
                 d_model,
                 heads,
                 d_ff,
                 d_k,
                 d_v,
                 attn_drop,
                 relu_drop,
                 res_drop,
                 clip_dist=0,
                 use_neg_dist=False,
                 d_content=None,
                 use_all_layers=False):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.use_all_layers = use_all_layers
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, d_k, d_v,
                                     attn_drop, relu_drop, res_drop,
                                     clip_dist=clip_dist, use_neg_dist=use_neg_dist,
                                     d_content=d_content)
             for _ in range(num_layers)])
        self.each_layer_norm = nn.ModuleList([LayerNorm(d_model)
                                              for _ in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)

    def forward(self, emb, lengths=None, mask=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(emb, lengths)

        out = emb
        representations = []
        attention_scores = []
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.each_layer_norm[i](out)
            if self.use_all_layers and i != 0:
                representations.append(out)
            out, attn_per_head = self.transformer[i](out, mask=mask)
            attention_scores.append(attn_per_head)

        out = self.layer_norm(out)
        representations.append(out)

        return representations, attention_scores
