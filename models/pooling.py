import transformers
import torch.nn as nn
import torch

class AttentionPooling(nn.Module):
    def __init__():
        nn.TransformerEncoder(nn.TransformerEncoderLayer())

class LSTMPooling(nn.Module):
    def __init__(self, model_config, 
                 hidden_size_lstm = 1024, 
                 num_layers = None,
                 dropout = 0.1,
                 bidirectional = False):
        super(LSTMPooling, self).__init__()

        if num_layers is None:
            num_layers = model_config.num_hidden_layers,

        self.lstm = nn.LSTM(model_config.hidden_size,
                            hidden_size_lstm,
                            num_layers = num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional)
    
    def forward(self, input):

        out, (hidden, cn) = self.lstm(input)

        return out[:, -1]