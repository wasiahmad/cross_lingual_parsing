""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn


class RelativePositionAttention(nn.Module):
    def __init__(self, clip_dist, dim, use_neg_dist):
        super(RelativePositionAttention, self).__init__()

        self.use_neg_dist = use_neg_dist
        self.clip_dist = clip_dist
        self.dim = dim

        c_dist = 2 * clip_dist + 1 if use_neg_dist else clip_dist + 1
        self.params = nn.Parameter(torch.FloatTensor(c_dist, dim))
        torch.nn.init.xavier_normal_(self.params)

        self.softmax = nn.Softmax(dim=-1)

    def _distance(self, key_len):
        dist_x = torch.arange(0, key_len).unsqueeze(0)
        dist_y = torch.arange(0, key_len).unsqueeze(1)
        distance = dist_x - dist_y
        distance = torch.clamp(distance, min=-self.clip_dist, max=self.clip_dist)
        if self.use_neg_dist:
            distance = (distance + self.clip_dist).long()  # [1,1,len,len]
        else:
            distance = torch.abs(distance)  # ignore directional information
        return distance

    def forward(self, key, mask):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :
           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """
        key_len = key.size(1)
        key_dim = key.size(2)
        assert self.dim == key_dim

        scores = torch.matmul(key, key.transpose(1, 2))  # [batch, key_len, key_len]

        distance = self._distance(key_len)
        distance = distance.cuda() if key.is_cuda else distance

        # implementation of Eq.5 by adding edge key vectors
        out = self.params.index_select(0, distance.view(-1))
        # 1 x key_len x key_len x dim
        out = out.view(1, key_len, key_len, key_dim)
        # bsz x key_len x key_len
        add_term = torch.matmul(key.unsqueeze(2), out.transpose(2, 3)).squeeze(2)
        scores = scores + add_term

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e20)

        attn = self.softmax(scores)  # bsz x key_len x key_len
        return attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self,
                 head_count,
                 model_dim,
                 d_k,
                 d_v,
                 dropout=0.1,
                 clip_dist=0,
                 use_neg_dist=False,
                 d_content=None):
        super(MultiHeadedAttention, self).__init__()

        self.head_count = head_count
        self.model_dim = model_dim
        self.d_k = d_k
        self.d_v = d_v
        self.clip_dist = clip_dist
        self.use_neg_dist = use_neg_dist
        self.partitioned = False if d_content is None else True

        if self.partitioned:
            self.d_position = self.model_dim - d_content
            self.d_content = d_content
            self.d_k_content = int((1.0 * d_k / self.model_dim) * d_content)
            self.d_k_position = d_k - self.d_k_content
            self.d_v_content = int((1.0 * d_v / self.model_dim) * d_content)
            self.d_v_position = d_v - self.d_v_content

            self.linear_keys1 = nn.Linear(self.d_content, head_count * self.d_k_content)
            self.linear_query1 = nn.Linear(self.d_content, head_count * self.d_k_content)
            self.linear_values1 = nn.Linear(self.d_content, head_count * self.d_v_content)

            self.linear_keys2 = nn.Linear(self.d_position, head_count * self.d_k_position)
            self.linear_query2 = nn.Linear(self.d_position, head_count * self.d_k_position)
            self.linear_values2 = nn.Linear(self.d_position, head_count * self.d_v_position)

        else:
            self.linear_keys = nn.Linear(model_dim, head_count * d_k)
            self.linear_query = nn.Linear(model_dim, head_count * d_k)
            self.linear_values = nn.Linear(model_dim, head_count * d_v)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        if self.partitioned:
            self.final_linear1 = nn.Linear(self.head_count * self.d_v_content, self.d_content, bias=False)
            self.final_linear2 = nn.Linear(self.head_count * self.d_v_position, self.d_position, bias=False)
        else:
            self.final_linear = nn.Linear(self.head_count * d_v, model_dim)

        # for relative-position aware self-attention
        if self.clip_dist > 0:
            c_dist = 2 * clip_dist + 1 if use_neg_dist else clip_dist + 1
            if self.partitioned:
                self.edge_keys_content = nn.Parameter(torch.FloatTensor(c_dist, self.d_k_content))
                self.edge_keys_position = nn.Parameter(torch.FloatTensor(c_dist, self.d_k_position))
                self.edge_values_content = nn.Parameter(torch.FloatTensor(c_dist, self.d_v_content))
                self.edge_values_position = nn.Parameter(torch.FloatTensor(c_dist, self.d_v_position))
                torch.nn.init.xavier_normal_(self.edge_keys_content)
                torch.nn.init.xavier_normal_(self.edge_keys_position)
                torch.nn.init.xavier_normal_(self.edge_values_content)
                torch.nn.init.xavier_normal_(self.edge_values_position)
            else:
                self.edge_keys = nn.Parameter(torch.FloatTensor(c_dist, d_k))
                self.edge_values = nn.Parameter(torch.FloatTensor(c_dist, d_v))
                torch.nn.init.xavier_normal_(self.edge_keys)
                torch.nn.init.xavier_normal_(self.edge_values)

    def forward(self, key, value, query, mask=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :
           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        batch_size = key.size(0)
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        value_len = value.size(1)
        use_gpu = key.is_cuda

        def shape(x, dim):
            """  projection """
            return x.view(batch_size, -1, head_count, dim).transpose(1, 2)

        def unshape(x, dim):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim)

        if self.clip_dist > 0:
            dist_x = torch.arange(0, key_len).unsqueeze(0)
            dist_y = torch.arange(0, key_len).unsqueeze(1)
            distance = dist_x - dist_y
            distance = torch.clamp(distance, min=-self.clip_dist, max=self.clip_dist)
            if self.use_neg_dist:
                distance = (distance + self.clip_dist).long()  # [1,1,len,len]
            else:
                distance = torch.abs(distance)  # ignore directional information
            distance = distance.cuda() if use_gpu else distance

        # 1) Project key, value, and query.
        if self.partitioned:
            key_up1 = shape(self.linear_keys1(key[:, :, :self.d_content]), self.d_k_content)
            key_up2 = shape(self.linear_keys2(key[:, :, :self.d_position]), self.d_k_position)
            key_up = torch.cat([key_up1, key_up2], dim=-1)  # bsz x nhead x key_len x d_k

            value_up1 = shape(self.linear_values1(value[:, :, :self.d_content]), self.d_v_content)
            value_up2 = shape(self.linear_values2(value[:, :, :self.d_position]), self.d_v_position)
            value_up = torch.cat([value_up1, value_up2], dim=-1)  # bsz x nhead x key_len x d_v

            query_up1 = shape(self.linear_query1(query[:, :, :self.d_content]), self.d_k_content)
            query_up2 = shape(self.linear_query2(query[:, :, :self.d_position]), self.d_k_position)
            query_up = torch.cat([query_up1, query_up2], dim=-1)  # bsz x nhead x query_len x d_k
        else:
            # key_up: bsz x nhead x key_len x d_k
            key_up = shape(self.linear_keys(key), self.d_k)  # x_j W^K
            # value_up: bsz x nhead x key_len x d_v
            value_up = shape(self.linear_values(value), self.d_v)  # x_j W^V
            # query_up: bsz x nhead x query_len x d_k
            query_up = shape(self.linear_query(query), self.d_k)  # x_j W^Q

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(self.d_k)  # (x_j W^Q) / sqrt(d_k)
        # bsz x nhead x query_len x key_len
        scores = torch.matmul(query_up, key_up.transpose(2, 3))
        if self.clip_dist > 0:
            if self.partitioned:
                out1 = self.edge_keys_content.index_select(0, distance.view(-1))
                out1 = out1.view(1, 1, query_len, query_len, self.d_k_content)
                add_term1 = torch.matmul(query_up[:, :, :, :self.d_k_content].unsqueeze(3),
                                         out1.transpose(3, 4)).squeeze(3)
                out2 = self.edge_keys_position.index_select(0, distance.view(-1))
                out2 = out2.view(1, 1, query_len, query_len, self.d_k_position)
                add_term2 = torch.matmul(query_up[:, :, :, -self.d_k_position:].unsqueeze(3),
                                         out2.transpose(3, 4)).squeeze(3)
                add_term = add_term1 + add_term2
            else:
                # implementation of Eq.5 by adding edge key vectors
                out = self.edge_keys.index_select(0, distance.view(-1))
                # 1 x 1 x key_len x key_len x d_k
                out = out.view(1, 1, query_len, query_len, self.d_k)
                # bsz x nhead x query_len x key_len
                add_term = torch.matmul(query_up.unsqueeze(3), out.transpose(3, 4)).squeeze(3)

            scores = scores + add_term

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e20)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)  # bsz x nhead x seq_len x seq_len
        context = torch.matmul(drop_attn, value_up)  # bsz x nhead x seq_len x d_v
        if self.clip_dist > 0:
            if self.partitioned:
                out1 = self.edge_values_content.index_select(0, distance.view(-1))
                out1 = out1.view(1, 1, key_len, key_len, self.d_v_content)
                add_term1 = torch.matmul(drop_attn.unsqueeze(3), out1).squeeze(3)
                out2 = self.edge_values_position.index_select(0, distance.view(-1))
                out2 = out2.view(1, 1, key_len, key_len, self.d_v_position)
                add_term2 = torch.matmul(drop_attn.unsqueeze(3), out2).squeeze(3)
                add_term = torch.cat([add_term1, add_term2], dim=-1)
            else:
                # split Eq.3 to save space
                out = self.edge_values.index_select(0, distance.view(-1))
                # 1 x 1 x key_len x key_len x dim
                out = out.view(1, 1, key_len, key_len, self.d_v)
                add_term = torch.matmul(drop_attn.unsqueeze(3), out).squeeze(3)  # bsz x nhead x seq_len x d_v

            context = context + add_term

        # bsz x nhead x seq_len x d_v --> bsz x seq_len x (nhead * d_v)
        context = unshape(context, self.d_v)

        # bsz x seq_len x (nhead * d_v) --> bsz x seq_len x model_dim
        if self.partitioned:
            context = context.view(batch_size, value_len, head_count, self.d_v).contiguous()
            context_l = context[:, :, :, :self.d_v_content].contiguous()
            context_l = context_l.view(batch_size, value_len, -1)
            context_n = context[:, :, :, -self.d_v_position:].contiguous()
            context_n = context_n.view(batch_size, value_len, -1)
            output = torch.cat([
                self.final_linear1(context_l),
                self.final_linear2(context_n),
            ], -1)
        else:
            output = self.final_linear(context)

        attn_per_head = [attn.squeeze(1).detach().cpu().numpy() if use_gpu else attn.squeeze(1).detach().numpy()
                         for attn in attn.chunk(head_count, dim=1)]
        return output, attn_per_head
