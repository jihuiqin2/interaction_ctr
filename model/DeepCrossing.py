
import torch
from torch import nn
from .BaseModel import BaseModel
from layer.embedding import EmbeddingLayer
from layer.shallow import ResidualBlock


class DeepCrossing(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DeepCrossing",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 residual_blocks=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 use_residual=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DeepCrossing, self).__init__(feature_map,
                                           model_id=model_id,
                                           gpu=gpu,
                                           embedding_regularizer=embedding_regularizer,
                                           net_regularizer=net_regularizer,
                                           **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(residual_blocks)
        layers = []
        input_dim = feature_map.num_fields * embedding_dim
        for hidden_dim, hidden_activation in zip(residual_blocks, hidden_activations):
            layers.append(ResidualBlock(input_dim,
                                        hidden_dim,
                                        hidden_activation,
                                        net_dropout,
                                        use_residual,
                                        batch_norm))
        layers.append(nn.Linear(input_dim, 1))
        self.crossing_layer = nn.Sequential(*layers)  # * used to unpack list

        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)  # [b,f,emb]
        y_pred = self.crossing_layer(feature_emb.flatten(start_dim=1))  # [b,f*emb]，再经过残差层
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
