

import torch
from torch import nn
from .BaseModel import BaseModel

from layer.embedding import EmbeddingLayer
from layer.mlp_layer import MLP_Layer
from layer.shallow import LR_Layer
from layer.attention import SqueezeExcitationLayer
from layer.interaction import BilinearInteractionLayer


class FiBiNet(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="FiBiNET",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",
                 embedding_dim=10,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 reduction_ratio=3,
                 bilinear_type="field_interaction",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(FiBiNet, self).__init__(feature_map,
                                      model_id=model_id,
                                      gpu=gpu,
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        num_fields = feature_map.num_fields
        self.senet_layer = SqueezeExcitationLayer(num_fields, reduction_ratio)
        self.bilinear_interaction = BilinearInteractionLayer(num_fields, embedding_dim, bilinear_type)
        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=False)
        input_dim = num_fields * (num_fields - 1) * embedding_dim
        self.dnn = MLP_Layer(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm,
                             use_bias=True)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)  # [b,f,emb]
        senet_emb = self.senet_layer(feature_emb)  # 挤压激励网络 [b,f,emb]
        bilinear_p = self.bilinear_interaction(feature_emb)  # [b,(f*f-f)/2, emb]
        bilinear_q = self.bilinear_interaction(senet_emb)  # [b,(f*f-f)/2, emb]
        comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
        dnn_out = self.dnn(comb_out)
        y_pred = self.lr_layer(X) + dnn_out
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
