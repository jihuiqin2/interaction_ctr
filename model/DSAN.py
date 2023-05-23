import torch
from torch import nn
from .BaseModel import BaseModel
from layer.embedding import EmbeddingLayer
from layer.mlp_layer import MLP_Layer
from layer.shallow import FeedForwardNetwork, ResidualBlock, LR_Layer
from layer.interaction import CrossNet, AttentionalAggregation, CrossInteractionLayer
from layer.attention import DisentangledSelfAttention


class DSAN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="CINFM2",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 use_residual=True,
                 relu_before_att=False,
                 batch_norm=False,
                 use_scale=False,
                 net_dropout=0.1,
                 att_dropout=0.1,
                 hidden_activations="relu",
                 attention_dim=16,
                 num_heads=2,
                 layer_norm=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 temperature=1,
                 bridge_type="hadamard_product",
                 num_cross_layers=3,
                 **kwargs):
        super(DSAN, self).__init__(feature_map,
                                     model_id=model_id,
                                     gpu=gpu,
                                     embedding_regularizer=embedding_regularizer,
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.embedding_dim = embedding_dim
        self.use_residual = use_residual
        self.num_cross_layers = num_cross_layers
        hidden_dim = feature_map.num_fields * embedding_dim

        # dnn层
        self.dense_layers = nn.ModuleList([MLP_Layer(input_dim=hidden_dim,  # MLP = linear + activation + dropout
                                                     output_dim=None,
                                                     hidden_units=[hidden_dim],
                                                     hidden_activations=hidden_activations,
                                                     output_activation=None,
                                                     dropout_rates=net_dropout,
                                                     batch_norm=False,
                                                     use_bias=True) \
                                           for _ in range(num_cross_layers)])

        self.cross_layers = nn.ModuleList([CrossInteractionLayer(hidden_dim) for _ in range(num_cross_layers)])
        self.bridge_modules = nn.ModuleList(
            [BridgeModule(embedding_dim, attention_dim, bridge_type) for _ in range(num_cross_layers)])
        self.regulation_modules = nn.ModuleList([RegulationModule(feature_map.num_fields,
                                                                  embedding_dim,
                                                                  tau=temperature,
                                                                  use_bn=batch_norm) \
                                                 for _ in range(num_cross_layers)])
        self.fc = nn.Linear(hidden_dim, 1)

        # 多头注意力
        self.dis_multi_head_attention = DisentangledSelfAttention(embedding_dim,
                                                                  attention_dim,
                                                                  num_heads,
                                                                  att_dropout,
                                                                  use_residual,
                                                                  use_scale,
                                                                  layer_norm,
                                                                  relu_before_att,
                                                                  align_to="input")
        self.aggregation_layers = nn.ModuleList([AttentionalAggregation(embedding_dim, hidden_dim)
                                                 for _ in range(num_cross_layers)])

        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)  # X = [b,f]，y = [b,1]
        feature_emb = self.embedding_layer(X)  # [b,f,emb]
        batch_size = feature_emb.shape[0]

        # 多头注意力部分
        dis_attention = self.dis_multi_head_attention(feature_emb, feature_emb, feature_emb)  # [b,f,emb]
        if self.num_cross_layers == 0:
            y_pred = self.fc(dis_attention.view(batch_size, -1))
        else:
            cross_i, deep_i = self.regulation_modules[0](dis_attention)  # [b,f,emb]
            cross_0 = cross_i
            for i in range(self.num_cross_layers):
                # [b,f,emb] = [b,1,emb] * [b,f,emb] + [b,f,emb]
                cross_i = self.aggregation_layers[i](cross_i).unsqueeze(1) * cross_0 + cross_i
                deep_i = self.dense_layers[i](deep_i.flatten(start_dim=1)).view(batch_size, -1,
                                                                                self.embedding_dim)  # [b,f,emb]
                bridge_i = self.bridge_modules[i](cross_i, deep_i)  # [b,f,emb]
                if i + 1 < len(self.cross_layers):
                    cross_i, deep_i = self.regulation_modules[i + 1](bridge_i)  # [b,f,emb]
             
            y_pred = self.fc((cross_i + deep_i + bridge_i).view(batch_size, -1))
            
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


class BridgeModule(nn.Module):
    def __init__(self, embedding_dim, attention_dim, bridge_type="attention_pooling"):
        super(BridgeModule, self).__init__()
        assert bridge_type in ["hadamard_product", "pointwise_addition", "concatenation", "attention_pooling"], \
            "bridge_type={} is not supported.".format(bridge_type)
        self.bridge_type = bridge_type
        if bridge_type == "concatenation":
            self.concat_pooling = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim),
                                                nn.ReLU())
        elif bridge_type == "attention_pooling":
            self.attention1 = nn.Sequential(nn.Linear(embedding_dim, attention_dim),
                                            nn.ReLU(),
                                            nn.Linear(attention_dim, 1, bias=False),
                                            nn.Softmax(dim=-1))
            self.attention2 = nn.Sequential(nn.Linear(embedding_dim, attention_dim),
                                            nn.ReLU(),
                                            nn.Linear(attention_dim, 1, bias=False),
                                            nn.Softmax(dim=-1))

    def forward(self, X1, X2):  # [b,f,emb]
        out = None
        if self.bridge_type == "hadamard_product":
            out = X1 * X2  # [b,f,emb]
        elif self.bridge_type == "pointwise_addition":
            out = X1 + X2
        elif self.bridge_type == "concatenation":
            out = self.concat_pooling(torch.cat([X1, X2], dim=-1))
        elif self.bridge_type == "attention_pooling":
            out = self.attention1(X1) * X1 + self.attention1(X2) * X2
        return out


class RegulationModule(nn.Module):
    def __init__(self, num_fields, embedding_dim, tau=1, use_bn=False):
        super(RegulationModule, self).__init__()
        self.tau = tau
        self.embedding_dim = embedding_dim
        self.use_bn = use_bn
        self.g1 = nn.Parameter(torch.ones(num_fields))
        self.g2 = nn.Parameter(torch.ones(num_fields))
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(num_fields * embedding_dim)
            self.bn2 = nn.BatchNorm1d(num_fields * embedding_dim)

    def forward(self, X):  # [b,f,emb]
        g1 = (self.g1 / self.tau).softmax(dim=-1).unsqueeze(-1).repeat(1, self.embedding_dim).unsqueeze(0)  # [1,f,emb]
        g2 = (self.g2 / self.tau).softmax(dim=-1).unsqueeze(-1).repeat(1, self.embedding_dim).unsqueeze(0)
        out1, out2 = g1 * X, g2 * X  # [b,f,emb]
        if self.use_bn:
            out1, out2 = self.bn1(out1), self.bn2(out2)
        return out1, out2
