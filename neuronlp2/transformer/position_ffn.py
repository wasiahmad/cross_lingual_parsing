"""
Position feed-forward network from "Attention is All You Need"
"""

import torch
import torch.nn as nn

from .util_class import LayerNorm


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self,
                 d_model,
                 d_ff,
                 d_content=None,
                 relu_drop=0.1,
                 res_drop=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.partitioned = False if d_content is None else True
        self.d_content = d_content
        if d_content:
            d_position = d_model - d_content
            self.w_1c = nn.Linear(d_content, d_ff // 2)
            self.w_1p = nn.Linear(d_position, d_ff // 2)
            self.w_2c = nn.Linear(d_ff // 2, d_content)
            self.w_2p = nn.Linear(d_ff // 2, d_position)
            self.layer_norm_c = LayerNorm(d_content)
            self.layer_norm_p = LayerNorm(d_position)
        else:
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)
            self.layer_norm = LayerNorm(d_model)

        self.relu_dropout = nn.Dropout(relu_drop)
        self.relu = nn.ReLU()
        self.residual_dropout = nn.Dropout(res_drop)

    def forward(self, x, no_residual=False):
        """
        Layer definition.
        Args:
            input: [ batch_size, input_len, model_dim ]
        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        if self.partitioned:
            xc = x[:, :, :self.d_content]
            xp = x[:, :, self.d_content:]

            outputc = self.w_1c(self.layer_norm_c(xc))
            outputc = self.relu_dropout(self.relu(outputc))
            outputc = self.w_2c(outputc)

            outputp = self.w_1p(self.layer_norm_p(xp))
            outputp = self.relu_dropout(self.relu(outputp))
            outputp = self.w_2p(outputp)

            output = torch.cat([outputc, outputp], -1)
        else:
            output = self.w_1(self.layer_norm(x))
            output = self.relu_dropout(self.relu(output))
            output = self.w_2(output)

        if no_residual:
            return output
        else:
            output = self.residual_dropout(output)
            return output + x
