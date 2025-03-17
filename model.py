import torch
import torch.nn as nn
import numpy as np

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, num_layers=6, d_ff=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(d_model, 5000)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:x.size(0), :]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x

    def positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
