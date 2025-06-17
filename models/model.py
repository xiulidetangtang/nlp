import transformers
import torch.nn as nn
import torch

from models.pooling import LSTMPooling

class BaseModel(nn.Module):
    def __init__(self, pretrained = None, config = None):
        super(BaseModel, self).__init__()

        if pretrained is None:
            pretrained = 'microsoft/deberta-v3-base'
        
        hidden_size_lstm = 128

        self.config = transformers.AutoConfig.from_pretrained(pretrained)
        self.deberta  = transformers.AutoModel. from_pretrained(pretrained ,config = self.config)

        self.pooling  = LSTMPooling(self.config,hidden_size_lstm = hidden_size_lstm, num_layers=12, dropout=0.6, bidirectional=True)
        self.linear  = nn.Linear(hidden_size_lstm*2, hidden_size_lstm)
        self.linear2 = nn.Linear(hidden_size_lstm, 6)
        self._init_weights(self.linear)
        self._init_weights(self.linear2)
        self.layers = nn.ModuleList([
            self.linear,
            nn.ReLU(),
            self.linear2
        ])

    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=0.1)

    def forward(self, tokens, mask):
        embedding = self.deberta(tokens, mask).last_hidden_state
        doc_embd  = self.pooling(embedding)
        output = doc_embd
        for layer in self.layers:
            output = layer(output)
        return output
    
class ModelSmall(nn.Module):
    """
    A BaseModel without deberta layer, to test the structure 
    """
    def __init__(self, pretrained = None, config = None):
        super(ModelSmall, self).__init__()

        if pretrained is None:
            pretrained = 'microsoft/deberta-v3-base'
        
        hidden_size_lstm = 128

        self.config = transformers.AutoConfig.from_pretrained(pretrained)
        self.pooling  = LSTMPooling(self.config,hidden_size_lstm = hidden_size_lstm, num_layers=12, dropout=0.3, bidirectional=True)
        

        linear1 = nn.Linear(2*hidden_size_lstm, hidden_size_lstm)
        linear2 = nn.Linear(hidden_size_lstm, 32)
        linear3 = nn.Linear(32, 6)
        self._init_weights(linear1)
        self._init_weights(linear2)
        self._init_weights(linear3)

        self.layers = nn.ModuleList([
            linear1,
            nn.Dropout(0.1),
            nn.ReLU(),
            linear2,
            nn.Dropout(0.1),
            nn.ReLU(),
            linear3,
        ])

    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=0.01)

    def forward(self, embedding):
        doc_embd  = self.pooling(embedding)
        output = doc_embd

        for layer in self.layers:
            output = layer(output)

        return output