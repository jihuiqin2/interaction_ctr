
import torch
from torch import nn
from .BaseModel import BaseModel
from layer.embedding import EmbeddingLayer
from layer.shallow import LR_Layer
from layer.interaction import InnerProductLayer
from layer.mlp_layer import MLP_Layer


class PNN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="AFM",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 product_type="inner",
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(PNN, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  embedding_regularizer=embedding_regularizer,
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        if product_type != "inner":
            raise NotImplementedError("product_type={} has not been implemented.".format(product_type))
        self.inner_product_layer = InnerProductLayer(feature_map.num_fields, output="inner_product")
        input_dim = int(feature_map.num_fields * (feature_map.num_fields - 1) / 2) \
                    + feature_map.num_fields * embedding_dim
        self.dnn = MLP_Layer(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.get_output_activation(task),
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm,
                             use_bias=True)

        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
                Inputs: [X, y]
                """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        inner_product_vec = self.inner_product_layer(feature_emb)  # [b,(f*f-f)/2]
        dense_input = torch.cat([feature_emb.flatten(start_dim=1), inner_product_vec], dim=1)
        y_pred = self.dnn(dense_input)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
