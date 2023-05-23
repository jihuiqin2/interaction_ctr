
import torch
from itertools import combinations
from torch import nn
from .BaseModel import BaseModel
from layer.embedding import EmbeddingLayer
from layer.shallow import LR_Layer
from layer.interaction import InnerProductLayer


class HOFM(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="HOFM",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 order=3,
                 embedding_dim=10,
                 reuse_embedding=False,
                 output_type=0,
                 regularizer=None,
                 **kwargs):
        super(HOFM, self).__init__(feature_map,
                                   model_id=model_id,
                                   gpu=gpu,
                                   embedding_regularizer=regularizer,
                                   net_regularizer=regularizer,
                                   **kwargs)
        self.order = order
        assert order >= 2, "order >= 2 is required in HOFM!"
        self.reuse_embedding = reuse_embedding

        if reuse_embedding:
            assert isinstance(embedding_dim, int), "embedding_dim should be an integer when reuse_embedding=True."
            self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        else:
            if not isinstance(embedding_dim, list):
                embedding_dim = [embedding_dim] * (order - 1)
            self.embedding_layers = nn.ModuleList([EmbeddingLayer(feature_map, embedding_dim[i]) \
                                                   for i in range(order - 1)])

        self.inner_product_layer = InnerProductLayer(feature_map.num_fields, output=output_type)

        self.lr_layer = LR_Layer(feature_map, use_bias=True)

        self.output_activation = self.get_output_activation(task)
        self.field_conjunction_dict = dict()
        for order_i in range(3, self.order + 1):  # 3阶交叉，4阶交叉...
            # # [(order_i个数),(),()],combinations迭代器
            order_i_conjunction = zip(*list(combinations(range(feature_map.num_fields), order_i)))
            for k, field_index in enumerate(order_i_conjunction):
                self.field_conjunction_dict[(order_i, k)] = torch.LongTensor(field_index)

        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        y_pred = self.lr_layer(X)  # [b,1]
        if self.reuse_embedding:
            feature_emb = self.embedding_layer(X)
        for i in range(2, self.order + 1):
            order_i_out = self.high_order_interaction(feature_emb if self.reuse_embedding \
                                                          else self.embedding_layers[i - 2](X), order_i=i)  # [b,1]
            y_pred += order_i_out
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

    def high_order_interaction(self, feature_emb, order_i):  # [b,f,emb], order_i
        if order_i == 2:
            interaction_out = self.inner_product_layer(feature_emb)  # [b,1]
        elif order_i > 2:
            index = self.field_conjunction_dict[(order_i, 0)].to(self.device)  # [2b]
            hadamard_product = torch.index_select(feature_emb, 1, index)  # [b,2b,emb]
            for k in range(1, order_i):
                index = self.field_conjunction_dict[(order_i, k)].to(self.device)  # [2b]
                hadamard_product = hadamard_product * torch.index_select(feature_emb, 1, index)  # [b,2b,emb]
            interaction_out = hadamard_product.sum((1, 2)).view(-1, 1)  # [b,1]
        return interaction_out
