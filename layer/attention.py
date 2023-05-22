# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (C) 2018. pengshuang@Github for ScaledDotProductAttention.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIMultiHeadSelfAttentionS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention
    Ref: https://zhuanlan.zhihu.com/p/47812375
    """

    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, W_q, W_k, W_v, scale=None, mask=None):  # [b,f,emb]
        # 【公式5】
        attention = torch.bmm(W_q, W_k.transpose(1, 2))  # [b,f,f]
        if scale:
            attention = attention / scale  # 比例缩小
        if mask:
            attention = attention.masked_fill_(mask, -np.inf)  # 按照mask格式填充，填充值为负无穷-np.inf
        attention = self.softmax(attention)  # [b,f,f]

        # 【公式6】
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention, W_v)  # [b,f,emb]
        return output, attention


class MultiHeadAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, embedding_dim, attention_dim=None, num_heads=1, dropout_rate=0.,
                 use_residual=True, use_scale=False, layer_norm=False, align_to="input"):
        super(MultiHeadAttention, self).__init__()
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads
        self.layer_norm = layer_norm
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.align_to = align_to
        self.scale = attention_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(embedding_dim, self.attention_dim, bias=False)  # input_dim=10,第二次input_dim=20
        self.W_k = nn.Linear(embedding_dim, self.attention_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, self.attention_dim, bias=False)

        if embedding_dim != self.attention_dim:
            if align_to == "output":
                self.W_res = nn.Linear(embedding_dim, self.attention_dim, bias=False)
            elif align_to == "input":
                self.W_res = nn.Linear(self.attention_dim, embedding_dim, bias=False)
        else:
            self.W_res = None

        # normal
        if layer_norm & embedding_dim != self.attention_dim:
            if align_to == "output":
                self.layer_norm = nn.LayerNorm(self.attention_dim) if layer_norm else None
            elif align_to == "input":
                self.layer_norm = nn.LayerNorm(self.embedding_dim) if layer_norm else None

        self.dot_product_attention = ScaledDotProductAttention(dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, query, key, value, mask=None):
        residual = query
        batch_size = query.size(0)  # b

        # linear projection  [b,f,amb]
        queries = self.W_q(query)
        keys = self.W_k(key)
        values = self.W_v(value)

        # split by heads  分隔连接 [batch*num_heads,filed,head_dim]
        Q_ = queries.view(batch_size * self.num_heads, -1, self.head_dim)
        K_ = keys.view(batch_size * self.num_heads, -1, self.head_dim)
        V_ = values.view(batch_size * self.num_heads, -1, self.head_dim)

        if mask:
            mask = mask.repeat(self.num_heads, 1, 1)
        # scaled dot product attention  [b*num_heads,f,head_dim]
        output, attention = self.dot_product_attention(Q_, K_, V_, self.scale, mask)

        # concat heads  【公式7】  [b,f,amb]
        output = output.view(batch_size, -1, self.attention_dim)

        # final linear projection  【公式8】
        if self.W_res is not None:
            if self.align_to == "output":  # [b,f,amb]
                residual = self.W_res(residual)
            elif self.align_to == "input":
                output = self.W_res(output)  # [b,f,emb]

        if self.dropout is not None:
            output = self.dropout(output)

        # 使用残差
        if self.use_residual:
            output = output + residual  # "output"=[b,f,amb]，"input"=[b,f,emb]
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output, attention  # "output"=[b,f,amb]，"input"=[b,f,emb]


# 多头自注意机制
class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self, X):
        output, attention = super(MultiHeadSelfAttention, self).forward(X, X, X)
        return output


# 解耦注意力机制
class DisentangledSelfAttention(nn.Module):
    """ Disentangle self-attention for DESTINE. The implementation totally follows the original code:
        https://github.com/CRIPAC-DIG/DESTINE/blob/c68e182aa220b444df73286e5e928e8a072ba75e/layers/activation.py#L90
    """

    # embedding_dim=10,attention_dim=20,self.head_dim = attention_dim // num_heads
    def __init__(self, embedding_dim, attention_dim=64, num_heads=1, att_dropout=0.1,
                 use_residual=True, use_scale=False, layer_norm=False, relu_before_att=False, align_to="output"):
        super(DisentangledSelfAttention, self).__init__()
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_scale = use_scale
        self.relu_before_att = relu_before_att
        self.use_residual = use_residual
        self.align_to = align_to
        self.layer_norm = layer_norm

        self.W_query = nn.Linear(embedding_dim, self.attention_dim)
        self.W_key = nn.Linear(embedding_dim, self.attention_dim)
        self.W_value = nn.Linear(embedding_dim, self.attention_dim)
        self.W_key2 = nn.Linear(embedding_dim, self.attention_dim)

        self.W_unary = nn.Linear(embedding_dim, num_heads)

        if embedding_dim != self.attention_dim:
            if self.align_to == "output":
                self.W_res = nn.Linear(embedding_dim, self.attention_dim)
            elif self.align_to == "input":
                self.W_res = nn.Linear(self.attention_dim, embedding_dim)
        else:
            self.W_res = None

        # normal
        if layer_norm & embedding_dim != self.attention_dim:
            if align_to == "output":
                self.layer_norm = nn.LayerNorm(self.attention_dim) if layer_norm else None
            elif align_to == "input":
                self.layer_norm = nn.LayerNorm(embedding_dim) if layer_norm else None

        self.dropout = nn.Dropout(att_dropout) if att_dropout > 0 else None

    def forward(self, query, key, value):  #

        residual = query

        batch_size = query.shape[0]

        if self.relu_before_att:
            queries = F.relu(self.W_query(query))
            keys = F.relu(self.W_key(key))
            values = F.relu(self.W_value(value))
            keys2 = F.relu(self.W_key2(key))
        else:
            queries = self.W_query(query)  # [32,39,20]
            keys = self.W_key(key)
            values = self.W_value(value)
            keys2 = self.W_key2(key)

        # split to （num_heads * [batch, len(field_dims), head_dim]）
        Q = queries.split(split_size=self.head_dim, dim=2)
        K = keys.split(split_size=self.head_dim, dim=2)
        V = values.split(split_size=self.head_dim, dim=2)
        K2 = keys2.split(split_size=self.head_dim, dim=2)

        # concat to [num_heads * batch, len(field_dims), head_dim]
        Q_ = torch.cat(Q, dim=0)
        K_ = torch.cat(K, dim=0)
        V_ = torch.cat(V, dim=0)
        K2_ = torch.cat(K2, dim=0)

        # whiten Q and K  求均值  [num_heads * batch, 1, head_dim]
        mu_Q = Q_.mean(dim=1, keepdim=True)
        mu_K = K_.mean(dim=1, keepdim=True)
        mu_K2 = K2_.mean(dim=1, keepdim=True)

        Q_ -= mu_Q  # [num_heads * batch, len(field_dims), head_dim]
        K_ -= mu_K

        # [num_heads * batch, len(field_dims), len(field_dims)]  【对应公式4】
        pairwise = torch.bmm(Q_, K_.transpose(1, 2))
        if self.use_scale:
            pairwise /= K_.shape[-1] ** 0.5
        # [num_heads * batch, len(field_dims), len(field_dims)]
        pairwise = F.softmax(pairwise, dim=2)

        # 【公式5】
        unary = torch.bmm(mu_K2, K_.transpose(1, 2))  # [num_heads * batch, 1, len(fields_dims)]
        unary = F.softmax(unary, dim=1)  # [num_heads * batch, 1, len(fields_dims)]

        # 【公式3】
        output = pairwise + unary  # [num_heads * batch, len(field_dims), len(field_dims)]
        if self.dropout is not None:
            output = self.dropout(output)

        # weighted sum for values  【公式7】
        # [num_heads * batch, len(field_dims), head_dim]
        output = torch.bmm(output, V_)

        # restore shape   【公式8】
        # (num_heads * [batch, len(field_dims), head_dim])
        output = output.split(batch_size, dim=0)
        output = torch.cat(output, dim=2)  # [b,f,amb]

        if self.W_res is not None:  # output=[b,f,amb]， input=[b,f,emb]
            if self.align_to == "output":
                residual = self.W_res(residual)
            elif self.align_to == "input":
                output = self.W_res(output)

        if self.dropout is not None:
            output = self.dropout(output)

        # 使用残差
        if self.use_residual:
            output = output + residual  # "output"=[b,f,amb]。"input"=[b,f,emb]
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()

        return output  # [b,f,amb] or [b,f,emb]


# 挤压激励网络
class SqueezeExcitationLayer(nn.Module):
    def __init__(self, num_fields, reduction_ratio=3):
        super(SqueezeExcitationLayer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False),
                                        nn.ReLU(),
                                        nn.Linear(reduced_size, num_fields, bias=False),
                                        nn.ReLU())

    def forward(self, feature_emb):  # [b,f,emb]
        Z = torch.mean(feature_emb, dim=-1, out=None)  # 挤压层，平均池化 [b,f]
        A = self.excitation(Z)  # 激励层，两个全连接层 [b,f]
        V = feature_emb * A.unsqueeze(-1)  # 重权，[b,f,emb]
        return V
