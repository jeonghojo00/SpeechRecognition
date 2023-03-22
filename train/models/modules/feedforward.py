import torch
import torch.nn as nn
import torch.nn.init as init
from models.modules.activations import Swish

class FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, dropout_p = 0.1):
        super(FeedForward, self).__init__()
        expanded = int(embed_dim * expansion_factor)
        
        self.sequential = nn.Sequential(
            nn.LayerNorm(embed_dim),
            Linear(embed_dim, expanded, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(expanded, embed_dim, bias=True),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return self.sequential(x)        
    

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=True)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x)