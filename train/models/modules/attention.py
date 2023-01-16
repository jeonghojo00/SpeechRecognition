import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np

def _get_pad_mask(inputs: Tensor, inputs_lens: Tensor):
    assert len(inputs.size()) == 3

    batch = inputs.size(0)

    pad_attn_mask = inputs.new_zeros(inputs.size()[: -1])

    for idx in range(batch):
        pad_attn_mask[idx, inputs_lens[idx]:] = 1

    return pad_attn_mask.bool()


def get_attn_pad_mask(inputs: Tensor, inputs_lens: Tensor, expand_lens):
    pad_attn_mask = _get_pad_mask(inputs, inputs_lens)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).repeat(1, expand_lens, 1)  # (batch, dec_T, enc_T)

    return pad_attn_mask


def _get_attn_key_pad_mask(target: Tensor, pad_id: int):
    target_lens = target.size(1)
    padding_mask = target.eq(pad_id)
    padding_mask = padding_mask.unsqueeze(1).repeat(1, target_lens, 1)

    return padding_mask


def _get_subsequent_mask(target: Tensor):
    batch, target_lens = target.size()
    subsequent_mask = torch.triu(torch.ones((target_lens, target_lens), device=target.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch, 1, 1)

    return subsequent_mask


def get_decoder_self_attn_mask(target: Tensor, pad_id: int = 0):
    padding_mask = _get_attn_key_pad_mask(target, pad_id)
    subsequent_mask = _get_subsequent_mask(target)

    decoder_self_attn_mask = (padding_mask + subsequent_mask).bool()

    return decoder_self_attn_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int=256, n_heads: int=4) -> None:
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim // self.n_heads) # Since Embedding_size(embed_dim) = Query_size * n_heads
        
        #key,query and value matrixes   
        self.q_matrix = nn.Linear(self.single_head_dim , self.single_head_dim, bias=False) # single head matrix of all n_heads matrices
        self.k_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.v_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)  

        self.out = nn.Linear(int(self.n_heads*self.single_head_dim) ,self.embed_dim) 

    def forward(self, key: Tensor, value:Tensor, query: Tensor, mask: Tensor = None) -> Tensor:
        '''
        input: key and value: (batch_size, seq_len_k, embed_dim)
                    query: (batch_size, seq_len_q, embed_dim)
        '''
        # key : (batch, seq_len, embed_dim)
        batch_size = key.size(0)
        seq_len_q = query.size(1)
        # because query sequence length can be different in decoder, define another seq_len for query
        seq_len_k = key.size(1) # key and value have the same sequence length
        dim_k = key.size(-1)
        
        # 1. Divide embed_dim by number of heads: Each head will have a dimension of single_head_dim (embed_dim = n_heads * single_head_dim)
        query = query.view(batch_size, seq_len_q, self.n_heads, self.single_head_dim) # (batch_size, seq_len_q, n_heads, single_head_dim)
        key   =   key.view(batch_size, seq_len_k, self.n_heads, self.single_head_dim) # (batch_size, seq_len_k, n_heads, single_head_dim)
        value = value.view(batch_size, seq_len_k, self.n_heads, self.single_head_dim) # (batch_size, seq_len_k, n_heads, single_head_dim)
        
        # 2. Linear Projections
        q = self.q_matrix(query) # (batch_size, seq_len_q, n_heads, single_head_dim)
        k = self.k_matrix(key)   # (batch_size, seq_len_k, n_heads, single_head_dim)
        v = self.v_matrix(value) # (batch_size, seq_len_k, n_heads, single_head_dim)
        
        # Reshape to (batch_size, n_heads, seq_len, single_head_dim)
        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len_q, single_head_dim)  
        k = k.transpose(1, 2)  # (batch_size, n_heads, seq_len_k, single_head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_heads, seq_len_k, single_head_dim)        
        
        # 3. Scaled-dot Product
        scores = torch.matmul(q, k.transpose(-1,-2))  
        ## query (batch_size, n_heads, seq_len_q, single_head_dim) x k.T (batch_size, n_heads, single_head_dim, seq_len_k)
        ## = scores (batch_size, n_heads, seq_len_q, seq_len_k)
      
        ## Masking before feed into Softmax
        ## fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # (batch_size, seq_len, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        
        ## Division by square root of dim_k and Apply Softmax
        scores = nn.Softmax(dim=-1)(scores/(np.sqrt(dim_k))) # (batch_size, n_ehads, seq_len_q, seq_len_k)
 
        ## Matrix multiplication with value
        scores = torch.matmul(scores, v)  
        # scores (batch_size, n_ehads, seq_len_q, seq_len_k) x value (batch_size, n_heads, seq_len_k, single_head_dim)
        # = scores (batch_size, n_heads, seq_len_q, single_head_dim)

        # 4. Reshape to Concatenate
        scores = scores.reshape(batch_size, seq_len_q, self.n_heads, self.single_head_dim)
        scores = scores.reshape(batch_size, seq_len_q, self.n_heads * self.single_head_dim)
        
        output = self.out(scores)  # (batch_size, seq_len_q, embed_dim)
       
        return output