# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
from .embedding import EmbeddingLayer
from .interaction import InnerProductLayer
from utils.torch_utils import get_activation

"""
nn.Liner和nn.Embedding的区别：https://blog.csdn.net/qq_43391414/article/details/120783887
Embedding和Linear几乎是一样的，区别就在于：输入不同，一个是输入数字，后者是输入one-hot向量。
习惯上，我们在模型的第一层使用的是Embedding，而不是Linear。模型的后续不会再使用Embedding，而是使用Linear。
"""

"""
在transformer中一般采用LayerNorm，
LayerNorm也是归一化的一种方法，与BatchNorm不同的是它是对每单个batch进行的归一化，而batchnorm是对所有batch一起进行归一化的
"""


class LR_Layer(nn.Module):
    def __init__(self, feature_map, output_activation=None, use_bias=True):
        super(LR_Layer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None
        self.output_activation = output_activation
        # A trick for quick one-hot encoding in LR
        self.embedding_layer = EmbeddingLayer(feature_map, 1, use_pretrain=False)

    def forward(self, X):  # [b,f]
        embed_weights = self.embedding_layer(X)  # [b,f,1]
        output = embed_weights.sum(dim=1)  # [b,1]
        if self.bias is not None:
            output += self.bias
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output


class FM_Layer(nn.Module):
    def __init__(self, feature_map, output_activation=None, use_bias=True):
        super(FM_Layer, self).__init__()
        self.inner_product_layer = InnerProductLayer(feature_map.num_fields, output="product_sum_pooling")
        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=use_bias)
        self.output_activation = output_activation

    def forward(self, X, feature_emb):
        lr_out = self.lr_layer(X)  # 线性部分 [b,1]
        dot_sum = self.inner_product_layer(feature_emb)  # 特征交叉部分  [b,1]
        output = dot_sum + lr_out  # [b,1]
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, layer_norm=True, use_residual=True):
        super(FeedForwardNetwork, self).__init__()
        self.use_residual = use_residual
        if hidden_dim is None:
            hidden_dim = 4 * input_dim
        self.ffn = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, input_dim))
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None

    def forward(self, X):
        output = self.ffn(X)
        if self.use_residual:
            output += X
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 hidden_activation="ReLU",
                 dropout_rate=0,
                 use_residual=True,
                 batch_norm=False):
        super(ResidualBlock, self).__init__()
        self.activation_layer = get_activation(hidden_activation)
        self.layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   self.activation_layer,
                                   nn.Linear(hidden_dim, input_dim))
        self.use_residual = use_residual
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, X):
        X_out = self.layer(X)
        if self.use_residual:
            X_out = X_out + X
        if self.batch_norm is not None:
            X_out = self.batch_norm(X_out)
        output = self.activation_layer(X_out)
        if self.dropout is not None:
            output = self.dropout(output)
        return output
