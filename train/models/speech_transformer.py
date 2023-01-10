import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor

from models.modules.transformer_encoder import TransformerEncoder
from models.modules.transformer_decoder import TransformerDecoder


class SpeechTransformer(nn.Module):
    def __init__(self, src_n_feats = 80, src_seq_len = 500, enc_layers = 4, embed_dim = 256, expansion_factor = 4, n_heads = 4,
                n_class=31, trg_seq_len=500, dec_layers=4, dropout_rate=0.1, device='cpu'):
        super(SpeechTransformer, self).__init__()

        self.device = device

        self.encoder = TransformerEncoder(n_feats = src_n_feats,
                                        max_seq_len = src_seq_len,
                                        n_layers = enc_layers,
                                        embed_dim = embed_dim,
                                        expansion_factor = expansion_factor,
                                        n_heads = n_heads,
                                        dropout_rate = dropout_rate,
                                        device = device)
        
        self.decoder = TransformerDecoder(n_class = n_class,
                                        embed_dim = embed_dim,
                                        max_seq_len = trg_seq_len,
                                        n_layers = dec_layers,
                                        expansion_factor = expansion_factor,
                                        n_heads = n_heads,
                                        dropout_rate = dropout_rate, 
                                        pad_id = 0,
                                        device = device)
    
        self.encoder_fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, n_class, bias=False),
        )

    def forward(self, src: Tensor, src_lens: Tensor, trg: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        enc_output, enc_output_lens = self.encoder(src, src_lens)
        enc_log_probs = self.encoder_fc(enc_output).log_softmax(dim=2).transpose(0, 1) # (time, batch, trg_char_size)
        dec_output = self.decoder(trg, enc_output, enc_output_lens)
        return dec_output, enc_log_probs, enc_output_lens

    @torch.no_grad()
    def recognize(self, src: Tensor, src_lens: Tensor) -> Tensor:
        enc_output, enc_output_lens = self.encoder(src, src_lens)
        enc_log_probs = self.encoder_fc(enc_output).log_softmax(dim=2).transpose(0, 1) # (time, batch, trg_char_size)

        return self.decoder.decode(enc_output, enc_output_lens), enc_log_probs, enc_output_lens