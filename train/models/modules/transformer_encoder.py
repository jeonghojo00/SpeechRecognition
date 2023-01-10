import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from models.modules.input_encoding import InputEncoder
from models.modules.position_encoding import PositionalEncoding
from models.modules.attention import MultiHeadAttention, get_attn_pad_mask

class TransformerEncoder(nn.Module):
    def __init__(self, n_feats = 128, max_seq_len=1200, n_layers=4, embed_dim=256, expansion_factor=4, n_heads=4, dropout_rate=0.1, device='cpu'):
        super(TransformerEncoder, self).__init__()
        
        self.device = device
        self.input_enc = InputEncoder(n_feats, embed_dim)
        self.position_enc = PositionalEncoding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([EncoderBlock(embed_dim = embed_dim, 
                                                  expansion_factor = expansion_factor,
                                                  n_heads = n_heads, 
                                                  dropout_rate = dropout_rate) for _ in range(n_layers)])
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, x_lens: Tensor) -> Tuple[Tensor, Tensor]:
        conv_output = self.input_enc(x) # Input Encoding
        seq_len = conv_output.size(1)
        output_lens = torch.LongTensor(x_lens)
        output_lens = (output_lens >> 2).int() # Convert the src length the same as output of conv encoder
        
        self_attn_mask = get_attn_pad_mask(conv_output, output_lens, seq_len)

        output = self.position_enc(conv_output) # Positional Encoding

        for layer in self.layers: # Transformer encoder for Ne times
            output = layer(output, self_attn_mask)
        output = self.layernorm(output) # Layer Normalization before fed into Transformer Encoder

        return output, output_lens

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=256, expansion_factor=4, n_heads=4, dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        """
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadAttention(embed_dim, n_heads)
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, x, self_attn_mask):
        residual = x.clone().detach()
        output = self.layernorm_1(x)
        output = self.self_attention(key = output, 
                                     value = output, 
                                     query = output,
                                     mask = self_attn_mask)
        output = self.dropout_1(output)
        output = output + residual

        residual = output.clone().detach()
        output = self.layernorm_2(output)
        output = self.feedforward(output)
        output = self.dropout_2(output)
        output = output + residual
        
        return output