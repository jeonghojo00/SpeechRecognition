import torch
import torch.nn as nn
from models.modules.activations import Swish

class HalfFeedForward(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, dropout_p = 0.1):
        super(HalfFeedForward, self).__init__()
        expanded = int(embed_dim * expansion_factor)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear_1 = nn.Linear(embed_dim, expanded)
        self.swish = Swish()
        self.dropout_1 = nn.Dropout(p=dropout_p)
        self.linear_2 = nn.Linear(expanded, embed_dim)
        self.dropout_2 = nn.Dropout(p=dropout_p)
    def forward(self, x):
        output = self.layer_norm(x)
        output = self.linear_1(output)
        output = self.swish(output)
        output = self.dropout_1(output)
        output = self.linear_2(output)
        output = self.dropout_2(output)
        return output        
    