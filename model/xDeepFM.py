

import torch
from torch import nn
from .BaseModel import BaseModel

from layer.embedding import EmbeddingLayer
from layer.mlp_layer import MLP_Layer
from layer.shallow import LR_Layer
from layer.interaction import CompressedInteractionNet


class xDeepFM(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="xDeepFM",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[64, 64, 64],
                 dnn_activations="ReLU",
                 cin_layer_units=[16, 16, 16],
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(xDeepFM, self).__init__(feature_map,
                                      model_id=model_id,
                                      gpu=gpu,
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        input_dim = feature_map.num_fields * embedding_dim
        self.dnn = MLP_Layer(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm,
                             use_bias=True) \
            if dnn_hidden_units else None  # in case of only CIN used

        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=False)

        self.cin = CompressedInteractionNet(feature_map.num_fields, cin_layer_units, output_dim=1)

        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)  # X = [b,f]，y = [b,1]
        feature_emb = self.embedding_layer(X)  # [b,f,emb]
        lr_logit = self.lr_layer(X)  # 线性层  [b,1]
        cin_logit = self.cin(feature_emb)  # [b,1]
        if self.dnn is not None:
            dnn_logit = self.dnn(feature_emb.flatten(start_dim=1))
            y_pred = lr_logit + cin_logit + dnn_logit  # LR + CIN + DNN
        else:
            y_pred = lr_logit + cin_logit  # only LR + CIN
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
