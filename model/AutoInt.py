
import torch
from torch import nn
from .BaseModel import BaseModel

from layer.embedding import EmbeddingLayer
from layer.mlp_layer import MLP_Layer
from layer.shallow import LR_Layer
from layer.attention import MultiHeadSelfAttention


class AutoInt(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="AutoInt",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[64, 64, 64],
                 dnn_activations="ReLU",
                 attention_layers=2,
                 num_heads=2,
                 attention_dim=8,
                 net_dropout=0,
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(AutoInt, self).__init__(feature_map,
                                      model_id=model_id,
                                      gpu=gpu,
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=False) \
            if use_wide else None
        self.dnn = MLP_Layer(input_dim=embedding_dim * feature_map.num_fields,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm,
                             use_bias=True) \
            if dnn_hidden_units else None  # in case no DNN used
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embedding_dim if i == 0 else attention_dim,
                                     attention_dim=attention_dim,
                                     num_heads=num_heads,
                                     dropout_rate=net_dropout,
                                     use_residual=use_residual,
                                     use_scale=use_scale,
                                     layer_norm=layer_norm,
                                     align_to="output")  # output=[b,f,amb]，input=[b,f,emb]
              for i in range(attention_layers)])
        self.fc = nn.Linear(feature_map.num_fields * attention_dim, 1)  # 需要根据align_to调整
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)  # [b,f], [b,1]
        feature_emb = self.embedding_layer(X)  # [b,f,emb]
        attention_out = self.self_attention(feature_emb)  # [b,f,amb]
        attention_out = torch.flatten(attention_out, start_dim=1)  # [b,f*amb]
        y_pred = self.fc(attention_out)
        if self.dnn is not None:
            y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        if self.lr_layer is not None:
            y_pred += self.lr_layer(X)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
