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
from .BaseModel import BaseModel
from layer.embedding import EmbeddingLayer
from layer.shallow import FeedForwardNetwork
from layer.mlp_layer import MLP_Layer
from layer.interaction import AttentionalAggregation
from layer.attention import MultiHeadSelfAttention


class InterHAt(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="InterHAt",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 hidden_dim=None,
                 order=2,
                 num_heads=1,
                 attention_dim=10,
                 hidden_units=[64, 64],
                 hidden_activations="relu",
                 batch_norm=False,
                 layer_norm=True,
                 use_residual=True,
                 net_dropout=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(InterHAt, self).__init__(feature_map,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.order = order
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.multi_head_attention = MultiHeadSelfAttention(embedding_dim,
                                                           attention_dim,
                                                           num_heads,
                                                           dropout_rate=net_dropout,
                                                           use_residual=use_residual,
                                                           use_scale=True,
                                                           layer_norm=layer_norm,
                                                           align_to="input")
        self.feedforward = FeedForwardNetwork(embedding_dim,
                                              hidden_dim=hidden_dim,
                                              layer_norm=layer_norm,
                                              use_residual=use_residual)
        self.aggregation_layers = nn.ModuleList([AttentionalAggregation(embedding_dim, hidden_dim)
                                                 for _ in range(order)])
        self.attentional_score = AttentionalAggregation(embedding_dim, hidden_dim)
        self.mlp = MLP_Layer(input_dim=embedding_dim,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        X0 = self.embedding_layer(X)
        X1 = self.feedforward(self.multi_head_attention(X0))  # [b,f,emb]
        X_p = X1
        agg_u = []
        for p in range(self.order):
            u_p = self.aggregation_layers[p](X_p)  # [b,emb]
            agg_u.append(u_p)
            if p != self.order - 1:
                X_p = u_p.unsqueeze(1) * X1 + X_p  # [b,f,emb] = [b,1,emb] * [b,f,emb] + [b,f,emb]
        U = torch.stack(agg_u, dim=1)  # [b, order, emb]
        u_f = self.attentional_score(U)  # [b, emb]
        y_pred = self.mlp(u_f)  # [b,1]
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
