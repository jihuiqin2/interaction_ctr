
from torch import nn
from .BaseModel import BaseModel
from layer.mlp_layer import MLP_Layer
from layer.embedding import EmbeddingLayer


class DNN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DNN",
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
        super(DNN, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  embedding_regularizer=embedding_regularizer,
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.dnn = MLP_Layer(input_dim=embedding_dim * feature_map.num_fields,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.get_output_activation(task),
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred = self.dnn(feature_emb.flatten(start_dim=1))  # tensor[64, 140]
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
