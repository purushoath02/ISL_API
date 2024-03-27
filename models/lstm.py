from dataclasses import asdict
from torchvision.models import mobilenet_v2
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict

# class LSTM(nn.Module):
#     def __init__(self, config, n_classes=50):
#         super().__init__()
#         config_dict = asdict(config)
#         self.lstm = nn.LSTM(**config_dict)
#         in_features = (
#             config.hidden_size * 2 if config.bidirectional else config.hidden_size
#         )
#         self.l1 = nn.Linear(in_features=in_features, out_features=n_classes)

#     def forward(self, x):
#         x, (_, _) = self.lstm(x)
#         x = torch.max(x, dim=1).values
#         x = F.dropout(x, p=0.3)
#         x = self.l1(x)
#         return x
# # @dataclass
# class LSTMConfig:
#     input_size: int
#     hidden_size: int
#     num_layers: int
#     bidirectional: bool

class LSTM(nn.Module):
    def __init__(self, config, n_classes=50):
        super(LSTM, self).__init__()

        config_dict = asdict(config)
        self.lstm = nn.LSTM(**config_dict)

        in_features = config.hidden_size * 2 if config.bidirectional else config.hidden_size

        # Additional LSTM layer
        self.lstm2 = nn.LSTM(in_features, config.hidden_size, batch_first=True, bidirectional=config.bidirectional)

        # Linear layers
        self.linear1 = nn.Linear(in_features, in_features)
        self.linear2 = nn.Linear(in_features, n_classes)

    def forward(self, x):
        # LSTM layers
        x, (_, _) = self.lstm(x)
        x, (_, _) = self.lstm2(x)

        # Global max pooling over the sequence dimension
        x = F.adaptive_max_pool1d(x.permute(0, 2, 1), (1,)).view(x.size(0), -1)

        # Linear layers
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=0.3)
        x = self.linear2(x)

        return x