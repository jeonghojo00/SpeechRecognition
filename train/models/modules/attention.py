import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

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
    def __init__(self, embed_dim, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads) # Since Embedding_size(embed_dim) = Query_size * n_heads
        
        #key,query and value matrixes   
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False) # single head matrix of all n_heads matrices
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)  

        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

    def forward(self, key, value, query, mask=None):
        # key : (batch, seq_len, embed_dim)
        batch_size = key.size(0)
        seq_len = key.size(1)
        # because query sequence length can be different in decoder, define another seq_len for query
        seq_len_query = query.size(1)
        
        # Convert embed_dim -> n_heads * single_head_dim
        key   = key.view(batch_size, seq_len, self.n_heads, self.single_head_dim)   # (batch_size, seq_len, n_heads, embed_dim/n_heads)
        value = value.view(batch_size, seq_len, self.n_heads, self.single_head_dim) # (batch_size, seq_len, n_heads, embed_dim/n_heads)
        query = query.view(batch_size, seq_len_query, self.n_heads, self.single_head_dim) # (batch_size, seq_len_query, n_heads, embed_dim/n_heads)

        k = self.key_matrix(key)     # (batch_size, seq_len,       n_heads, embed_dim/n_heads)
        v = self.value_matrix(value) # (batch_size, seq_len,       n_heads, embed_dim/n_heads)
        q = self.query_matrix(query) # (batch_size, seq_len_query, n_heads, embed_dim/n_heads)

        k = k.transpose(1, 2)  # (batch_size, n_heads, seq_len,       single_head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_heads, seq_len,       single_head_dim)        
        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len_query, single_head_dim)  

        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_len) 
        product = torch.matmul(q, k_adjusted)  
        ## (batch_size, n_heads, seq_len_query, single_head_dim) x (batch_size, n_heads, single_head_dim, eq_len) = (batch_size, n_heads, seq_len, seq_len)

        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # (batch_size, seq_len, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
            product = product.masked_fill(mask == 0, float("-1e20"))
        
        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product, dim=-1)
 
        #mutiply with value matrix
        scores = torch.matmul(scores, v)  
        ## (batch_size, n_heads, seq_len, seq_len) x (batch_size, n_heads, seq_len, single_head_dim) = (batch_size, n_heads, seq_len, single_head_dim) 
        
        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_len_query, self.single_head_dim*self.n_heads)  
        ## (batch_size, n_heads, seq_len, single_head_dim) -> (batch_size, seq_len, n_heads, single_head_dim)  
        ## -> (batch_size, seq_len, n_heads * single_head_dim)
        
        output = self.out(concat) #(batch_size, seq_len, n_heads * single_head_dim) -> (batch_size, seq_len, d_dim)
       
        return output