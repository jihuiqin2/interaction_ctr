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
from itertools import combinations


# FM、FwFM等特征交叉部分
class InnerProductLayer(nn.Module):
    """ output: product_sum_pooling (bs x 1),
                Bi_interaction_pooling (bs * dim),
                inner_product (bs x (f*f-f)/2),
                elementwise_product (bs x (f*f-f)/2 x emb_dim)
    """

    def __init__(self, num_fields=None, output="product_sum_pooling"):
        super(InnerProductLayer, self).__init__()
        self._output_type = output
        if output not in ["product_sum_pooling", "Bi_interaction_pooling", "inner_product", "elementwise_product"]:
            raise ValueError("InnerProductLayer output={} is not supported.".format(output))
        if num_fields is None:
            if output in ["inner_product", "elementwise_product"]:
                raise ValueError("num_fields is required when InnerProductLayer output={}.".format(output))
        else:
            # self.field_p=self.field_q=(f*f-f)/2个，取除去对角线的上三角[(0,1),(0,2)...(1,2)]，p取值为所有的行的列表，q为所有的列的列表
            p, q = zip(*list(combinations(range(num_fields), 2)))
            self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)  # 转为tensor，取第一个数
            self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)  # 转为tensor，取第二个数

            self.interaction_units = int(num_fields * (num_fields - 1) / 2)  # 特征无重复交叉的个数
            self.upper_triange_mask = nn.Parameter(
                torch.triu(torch.ones(num_fields, num_fields), 1).type(torch.bool),  # 除去对角线的上三角为1
                requires_grad=False)

    def forward(self, feature_emb):  # [b,f,emb]
        if self._output_type in ["product_sum_pooling", "Bi_interaction_pooling"]:  # FM交叉部分
            sum_of_square = torch.sum(feature_emb, dim=1) ** 2  # sum then square [b,emb]
            square_of_sum = torch.sum(feature_emb ** 2, dim=1)  # square then sum [b,emb]
            bi_interaction = (sum_of_square - square_of_sum) * 0.5  # [b,emb]
            if self._output_type == "Bi_interaction_pooling":  # NFM
                return bi_interaction  # [b, emb]
            else:
                return bi_interaction.sum(dim=-1, keepdim=True)  # [b,1]
        elif self._output_type == "elementwise_product":  # AFM特征交叉部分
            # 根据self.field_p的个数取feature_emb的f行数， [b，self.field_p，emb]
            emb1 = torch.index_select(feature_emb, 1, self.field_p)
            emb2 = torch.index_select(feature_emb, 1, self.field_q)
            return emb1 * emb2  # [b,(f*f-f)/2,emb]
        elif self._output_type == "inner_product":  # IPNN
            # 矩阵乘法 [b,f,f]
            inner_product_matrix = torch.bmm(feature_emb, feature_emb.transpose(1, 2))
            # 根据self.upper_triange_mask（为true的）条件从inner_product_matrix中取值，得到[b*(f*f-f)/2]个数
            flat_upper_triange = torch.masked_select(inner_product_matrix, self.upper_triange_mask)
            return flat_upper_triange.view(-1, self.interaction_units)  # [b*(f*f-f)/2]转为[b,(f*f-f)/2]


class CrossInteractionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, cross_type="weight_cross"):
        super(CrossInteractionLayer, self).__init__()
        if cross_type == "weight_cross":
            self.weight = nn.Linear(input_dim, 1, bias=False)
        elif cross_type == "attention_cross":
            if hidden_dim is None:
                hidden_dim = 4 * input_dim
            self.weight = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, 1, bias=False),
                                        nn.Softmax(dim=1))

        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):  # [b,f*emb]
        interaction_out = self.weight(X_i) * X_0 + self.bias  # [b,f*emb]
        return interaction_out


# DCN
class CrossNet(nn.Module):
    """
    cross_type两个类型[weight_cross, attention_cross]
    """

    def __init__(self, input_dim, num_layers, hidden_dim=None, cross_type="weight_cross"):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = nn.ModuleList(CrossInteractionLayer(input_dim, hidden_dim, cross_type)
                                       for _ in range(self.num_layers))

    def forward(self, X_0):  # [b,f*emb]
        X_i = X_0
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)  # [b,f*emb]
        return X_i  # [b,f*emb]


# CIN
class CompressedInteractionNet(nn.Module):
    def __init__(self, num_fields, cin_layer_units, output_dim=1):
        super(CompressedInteractionNet, self).__init__()
        self.cin_layer_units = cin_layer_units
        self.fc = nn.Linear(sum(cin_layer_units), output_dim)
        self.cin_layer = nn.ModuleDict()

        for i, unit in enumerate(self.cin_layer_units):  # 压缩
            in_channels = num_fields * self.cin_layer_units[i - 1] if i > 0 else num_fields ** 2
            out_channels = unit
            self.cin_layer["layer_" + str(i + 1)] = nn.Conv1d(in_channels,
                                                              out_channels,  # how many filters
                                                              kernel_size=1)  # kernel output shape

    def forward(self, feature_emb):  # [b,f,emb]
        pooling_outputs = []
        X_0 = feature_emb  # [b,f,emb]
        batch_size = X_0.shape[0]  # 样本数
        embedding_dim = X_0.shape[-1]  # 维度数
        X_i = X_0
        for i in range(len(self.cin_layer_units)):
            hadamard_tensor = torch.einsum("bhd,bmd->bhmd", X_0, X_i)  # [b,f,f,emb]
            hadamard_tensor = hadamard_tensor.view(batch_size, -1, embedding_dim)  # [b, f*f, emb]

            X_i = self.cin_layer["layer_" + str(i + 1)](hadamard_tensor) \
                .view(batch_size, -1, embedding_dim)  # [b,out_channels,emb]

            pooling_outputs.append(X_i.sum(dim=-1))  # [b,out_channels]（将一行的所有列相加）

        concate_vec = torch.cat(pooling_outputs, dim=-1)  # [b,out_channels*len(self.cin_layer_units)]
        output = self.fc(concate_vec)  # [b,1]
        return output


# InterHAt交叉层
class AttentionalAggregation(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=None):
        super(AttentionalAggregation, self).__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim
        self.agg = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 1, bias=False),
                                 nn.Softmax(dim=1))

    def forward(self, X):  # [b, f, emb]
        attentions = self.agg(X)  # [b,f,1]
        attention_out = (attentions * X).sum(dim=1)  # [b, emb]
        return attention_out


# FiBiNet
class BilinearInteractionLayer(nn.Module):
    def __init__(self, num_fields, embedding_dim, bilinear_type="field_interaction"):
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "field_all":  # 1
            self.bilinear_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        elif self.bilinear_type == "field_each":  # f
            self.bilinear_layer = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False)
                                                 for i in range(num_fields)])
        elif self.bilinear_type == "field_interaction":  # (f*f-f)/2
            self.bilinear_layer = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False)
                                                 for i, j in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, feature_emb):  # [b,f,emb]
        feature_emb_list = torch.split(feature_emb, 1, dim=1)  # f个[b,1,emb]
        if self.bilinear_type == "field_all":
            bilinear_list = [self.bilinear_layer(v_i) * v_j
                             for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j]
                             for i, j in combinations(range(len(feature_emb_list)), 2)]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0]) * v[1]
                             for i, v in enumerate(combinations(feature_emb_list, 2))]  # (f*f-f)/2个[b,1,emb]
        return torch.cat(bilinear_list, dim=1)  # [b, (f*f-f)/2,emb]


# todo 模型交叉部分
class CrossNetAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, order, num_fields):
        super(CrossNetAttention, self).__init__()
        self.order = order
        self.cross_net_score = nn.ModuleList([AttentionalAggregation(embedding_dim, hidden_dim)
                                              for _ in range(order)])

        self.attention_net_score = AttentionalAggregation(embedding_dim, hidden_dim)
        self.fc = nn.Linear(embedding_dim, 1)

        """
        num_fields是偶数24，lp=(f*f-f)/2，则有p=[23个0,22个1...,1个22]，q=[1-23,2-23...,22-23,23]，lp为276
        """
        p, q = zip(*list(combinations(range(num_fields), 2)))
        self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)  # 转为tensor，取第一个数，共有lp个
        self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)  # 转为tensor，取第二个数，共有lp个

    def forward(self, feature_emb):  # [b,f,emb]
        X_0 = feature_emb
        X_i = X_0
        pooling_outputs = []
        for i in range(self.order):
            if i == 0:
                emb1 = torch.index_select(X_0, 1, self.field_p)  # [b,lp,emb]
                emb2 = torch.index_select(X_i, 1, self.field_q)  # [b,lp,emb]
                cross_product = emb1 * emb2  # [b,lp,emb]
            else:
                cross_product = X_0 * X_i  # [b,f,emb]
            cross_result = self.cross_net_score[i](cross_product)  # [b,emb]
            X_i = cross_result.unsqueeze(1) * X_0 + X_i  # [b,f,emb]
            pooling_outputs.append(cross_result)  # [b,emb]

        U = torch.stack(pooling_outputs, dim=1)  # [b,order,emb]
        U_result = self.attention_net_score(U)  # [b,emb]

        output = self.fc(U_result)  # [b,1]
        return output
