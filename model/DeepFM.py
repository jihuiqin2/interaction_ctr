

from torch import nn
from .BaseModel import BaseModel
from layer.shallow import FM_Layer
from layer.mlp_layer import MLP_Layer
from layer.embedding import EmbeddingLayer


class DeepFM(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DeepFM",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DeepFM, self).__init__(feature_map,
                                     model_id=model_id,
                                     gpu=gpu,
                                     embedding_regularizer=embedding_regularizer,
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.fm_layer = FM_Layer(feature_map, output_activation=None, use_bias=False)
        self.dnn = MLP_Layer(input_dim=embedding_dim * feature_map.num_fields,
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
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred = self.fm_layer(X, feature_emb)
        y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
