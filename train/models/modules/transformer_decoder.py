import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.modules.position_encoding import PositionalEncoding
from models.modules.attention import MultiHeadAttention, get_attn_pad_mask, get_decoder_self_attn_mask


class TransformerDecoder(nn.Module):
    def __init__(self, n_class, embed_dim, max_seq_len, n_layers=4, expansion_factor=4, n_heads=4, dropout_rate=0.1,
                 pad_id = 0, device='cpu'):
        super(TransformerDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.device = device

        self.char_embedding = nn.Embedding(n_class, embed_dim)
        self.position_enc = PositionalEncoding(max_seq_len=max_seq_len, embed_dim=embed_dim)
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, expansion_factor, n_heads, dropout_rate) for _ in range(n_layers)])

        self.layernorm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(in_features = embed_dim,
                                out_features = n_class)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, target, enc_output, enc_output_lens):
        batch_size = target.size(0)
        if len(target.size()) == 1:  # validate, evaluation
            target = target.unsqueeze(1)
        #else:  # train
        #    target = target[target != self.eos_id].view(batch_size, -1)

        target_lens = target.size(1)
        self_attn_mask = get_decoder_self_attn_mask(target = target, pad_id = self.pad_id)
        enc_dec_attn_mask = get_attn_pad_mask(enc_output, enc_output_lens, target_lens)
        
        output = self.char_embedding(target)
        output = self.position_enc(output)
        output = self.dropout(output)

        for layer in self.layers:
            output = layer(output, enc_output, self_attn_mask, enc_dec_attn_mask)
        output = F.softmax(self.linear(self.layernorm(output)))
        return output
    
    @torch.no_grad()
    def decode(self, enc_output: Tensor, enc_output_lens: Tensor) -> Tensor:
        batch = enc_output.size(0)
        y_hat = list()

        inputs = torch.LongTensor([self.pad_id] * batch)
        inputs = inputs.to(self.device)

        for i in range(0, self.max_seq_len):
            dec_output_prob = self.forward(inputs, enc_output, enc_output_lens)
            dec_output_prob = dec_output_prob.squeeze(1)
            y_hat.append(dec_output_prob)

        y_hat = torch.stack(y_hat, dim=1)

        return y_hat # (batch, time, character size)

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor, n_heads, dropout_rate=0.2):
        super(DecoderBlock, self).__init__()
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
        self.enc_dec_attention = MultiHeadAttention(embed_dim, n_heads)
        self.dropout_2 = nn.Dropout(dropout_rate)
        
        self.layernorm_3 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )
        self.dropout_3 = nn.Dropout(dropout_rate)

    def forward(self, query, enc_out, self_attn_mask, enc_dec_attn_mask):
        """In "encoder-decoder attention" layers, the queries come from the previous decoder layer,
        and the memory keys and values come from the output of the encoder.
        """
        residual = query.clone().detach()
        output = self.layernorm_1(query)
        output = self.self_attention(key = output, value = output, query = output, 
                                     mask = self_attn_mask)
        output = self.dropout_1(output)
        output = output + residual

        residual = output.clone().detach()
        output = self.layernorm_2(output)
        output = self.enc_dec_attention(key = enc_out, value = enc_out, query = output, 
                                        mask = enc_dec_attn_mask)
        output = self.dropout_2(output)
        output = output + residual

        residual = output.clone().detach()
        output = self.layernorm_3(output)
        output = self.feedforward(output)
        output = self.dropout_3(output)
        output = output + residual
        
        return output