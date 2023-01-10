import math
import torch
import torch.nn as nn

"""
Positional Encoding
- embed_dim must be EVEN number since the second FOR-loop updates for every two
- max_seq_len must be greater than equal to the the maximum time after InputEncoder
- pe is updated in a way that it is updated by row 
  and for every two columns, odd columns are updated with sine function 
  and even columns are updated with cos function.
- Since current Librispeech dataset has the maximum sequence length of 1963 and it can be reduced into 491 through InputEncoder, 
  the max_seq_len can be any number greater than or equal to 491 for this training.
- referred from https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
"""
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len=500, embed_dim=512):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        
        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x