import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=150): #max_length needs to be check
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        size = (x.size(0))
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp

        self.bn1 = nn.BatchNorm1d(100 * 150)

        # self.decoder = nn.Linear(10 * 100 * ninp, 10 * ntoken) #previous
        self.decoder = nn.Linear(10 * 100 * ninp, 10*ntoken) # changed to add batch_normalization

        # self.dense = nn.Linear(ntoken, ntoken)
        # self.dropout = nn.Dropout(0.2)
        # self.out_proj = nn.Linear(ntoken, 1)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = src * math.sqrt(self.ninp) #self attention mechanism
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        output = output.view(10, -1)

        output = self.bn1(output)

        output = output.view(-1)
        output = self.decoder(output)



        # output = self.dropout(output)
        # output = self.dense(output)
        # output = torch.tanh(output)
        # output = self.dropout(output)
        # output = self.out_proj(output)
        return output