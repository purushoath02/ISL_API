import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class PositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, :seq_length]

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = x + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Transformer(nn.Module):
    def __init__(self, config, n_classes=50):
        super().__init__()
        self.l1 = nn.Linear(
            in_features=config.input_size, out_features=config.hidden_size
        )
        self.embedding = PositionEmbedding(config)
        self.layers = nn.ModuleList(
            [
                transformers.BertLayer(config.model_config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.l2 = nn.Linear(in_features=config.hidden_size, out_features=n_classes)

    def forward(self, x):
        x = self.l1(x)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)[0]

        x = torch.max(x, dim=1).values
        x = F.dropout(x, p=0.2)
        x = self.l2(x)
        return x



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import transformers

# class MultiHeadAttention(nn.Module):
#     def __init__(self, config):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = config.num_attention_heads
#         self.head_dim = config.hidden_size // self.num_heads

#         self.query = nn.Linear(config.hidden_size, config.hidden_size)
#         self.key = nn.Linear(config.hidden_size, config.hidden_size)
#         self.value = nn.Linear(config.hidden_size, config.hidden_size)
#         self.fc_out = nn.Linear(config.hidden_size, config.hidden_size)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()
#         query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

#         scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.head_dim**0.5
#         attention_weights = F.softmax(scores, dim=-1)

#         attended_values = torch.matmul(attention_weights, value)
#         attended_values = attended_values.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
#         output = self.fc_out(attended_values)
#         return output

# class Transformer(nn.Module):
#     def __init__(self, config, n_classes=50):
#         super(Transformer, self).__init__()
#         self.l1 = nn.Linear(config.input_size, config.hidden_size)
#         self.embedding = PositionEmbedding(config)
        
#         self.attention_layers = nn.ModuleList([
#             MultiHeadAttention(config) for _ in range(config.num_attention_heads)
#         ])

#         self.feedforward_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(config.hidden_size, config.intermediate_size),
#                 nn.ReLU(),
#                 nn.Linear(config.intermediate_size, config.hidden_size),
#             ) for _ in range(config.num_hidden_layers)
#         ])

#         self.l2 = nn.Linear(config.hidden_size, n_classes)

#     def forward(self, x):
#         x = self.l1(x)
#         x = self.embedding(x)

#         for attention_layer in self.attention_layers:
#             x = x + attention_layer(x)

#         for feedforward_layer in self.feedforward_layers:
#             x = x + feedforward_layer(x)

#         x = torch.max(x, dim=1).values
#         x = F.dropout(x, p=0.2)
#         x = self.l2(x)
#         return x

# # Example Configuration
# # class ModelConfig:
# #     def __init__(self):
# #         self.input_size = 768  # Input size of the model
# #         self.hidden_size = 768  # Hidden size of the model
# #         self.intermediate_size = 3072  # Intermediate size in the feedforward layer
# #         self.num_hidden_layers = 6  # Number of transformer layers
# #         self.num_attention_heads = 12  # Number of attention heads
# #         self.max_position_embeddings = 512  # Maximum position embeddings
# #         self.layer_norm_eps = 1e-12  # Layer normalization epsilon

# # # Create an instance of the model
# # config = ModelConfig()
# # model = Transformer(config)
# # print(model)
