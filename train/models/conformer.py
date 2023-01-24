import torch
import torch.nn as nn
from torch import Tensor
from models.modules.input_encoding import InputEncoder
from models.modules.half_feedforward import HalfFeedForward
from models.modules.attention import Conformer_MultiHeadAttention
from models.modules.convolution import Convolution

class Conformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 29,
        n_feats: int = 80,
        conv_sampling: str = 'vgg',
        n_encoders: int = 16,
        embed_dim: int = 256,
        ff_expansion_factor: int = 4,
        mha_heads: int = 4,
        conv_kernel_size: int = 31,
        dropout_p: float = 0.1,
        lstm_layers: int = 1
    ) -> None:
        super().__init__()
        
        self.encoder = ConformerEncoder(n_feats, 
                                        conv_sampling,
                                        n_encoders,
                                        embed_dim,
                                        ff_expansion_factor,
                                        mha_heads,
                                        conv_kernel_size,
                                        dropout_p)
        self.decoder = nn.LSTM(input_size = embed_dim,
                               hidden_size = embed_dim,
                               num_layers = lstm_layers,
                               batch_first = True)
        self.fc_out = nn.Linear(embed_dim, num_classes)
    
    def forward(self, inputs, input_lens):
        enc_outputs, output_lens = self.encoder(inputs, input_lens)
        dec_outputs, _ = self.decoder(enc_outputs)
        fc_outputs = self.fc_out(dec_outputs).log_softmax(dim=-1)
        
        return fc_outputs, output_lens
    
    @torch.no_grad()
    def decode(self, inputs: Tensor, max_len: int = None) -> Tensor:
        return inputs.max(-1)[1]
    
    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lens: Tensor) -> Tensor:
        enc_outputs, _ = self.encoder(inputs, input_lens)
        dec_outputs, _ = self.decoder(enc_outputs)
        fc_outputs = self.fc_out(dec_outputs).log_softmax(dim=-1)
        
        return self.decode(fc_outputs)
    
class ConformerEncoder(nn.Module):
    def __init__(
        self,
        n_feats: int = 80,
        conv_sampling: str = 'vgg',
        n_encoders: int = 16,
        embed_dim: int = 256,
        ff_expansion_factor: int = 4,
        mha_heads: int = 4,
        conv_kernel_size: int = 31,
        dropout_p: float = 0.1) -> None:
        super().__init__()
        
        self.input_enc = InputEncoder(conv_subsampling=conv_sampling, n_feats=n_feats, embed_dim = embed_dim, dropout_p=dropout_p)
        self.conformer_blocks = nn.ModuleList([ConformerBlock(
            embed_dim = embed_dim,
            ff_expansion_factor = ff_expansion_factor,
            mha_heads = mha_heads,
            conv_kernel_size = conv_kernel_size,
            dropout_p = dropout_p
        ) for _ in range(n_encoders)])
        
    def forward(self, inputs: Tensor, input_lens: Tensor) -> Tensor:
        outputs, output_lens = self.input_enc(inputs, input_lens)
        for block in self.conformer_blocks:
            outputs = block(outputs)
        return outputs, output_lens
    
class ConformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        ff_expansion_factor: int = 4,
        mha_heads: int = 4,
        conv_kernel_size: int = 31,
        dropout_p: float = 0.1) -> None:
        super().__init__()
        
        self.sequential = nn.Sequential(
            ResidualConnection(module=HalfFeedForward(embed_dim = embed_dim, 
                                                      expansion_factor= ff_expansion_factor,
                                                      dropout_p = dropout_p),
                               module_factor = 0.5),
            ResidualConnection(module=Conformer_MultiHeadAttention(embed_dim = embed_dim, 
                                                               n_heads = mha_heads, 
                                                               dropout_p = dropout_p)),
            ResidualConnection(module=Convolution(in_channels = embed_dim,
                                                  kernel_size = conv_kernel_size,
                                                  dropout_p = dropout_p)),
            ResidualConnection(module=HalfFeedForward(embed_dim = embed_dim, 
                                                      expansion_factor= ff_expansion_factor,
                                                      dropout_p = dropout_p),
                               module_factor = 0.5), 
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)  
        
class ResidualConnection(nn.Module):
    """
    Residual Connection to control half-step or regular residual connections
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0) -> None:
        super().__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor
        
    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)