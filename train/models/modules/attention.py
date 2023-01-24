import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np
from typing import Optional

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

# Original Multi-Headed Attention based on "Attention is all you need"
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


# Multi-Headed Attention for Conformer that uses relative positioning encoding
class Conformer_MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, n_heads: int = 8, dropout_p = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.attention = MultiHeadAttention_Relative(embed_dim, n_heads)
        self.dropout = nn.Dropout(p=dropout_p)
    
    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_len, _ = inputs.size()
        pos_embed = self.positional_encoding(seq_len)
        pos_embed = pos_embed.repeat(batch_size, 1, 1)
        
        outputs = self.layer_norm(inputs)
        outputs = self.attention(key=outputs, value=outputs, query=outputs,
                                 pos_embed = pos_embed, mask=mask)
        outputs = self.dropout(outputs)
        return outputs
    
class MultiHeadAttention_Relative(nn.Module):
    def __init__(self, embed_dim: int = 512, n_heads: int = 8):
        super().__init__()
        # Check validity of inputs
        assert math.sqrt(embed_dim).is_integer(), "Embed_dim must be a square number for Scaling"
        assert embed_dim % n_heads == 0, "Embed_dim must be a multiple of n_heads"
        
        self.embed_dim = embed_dim
        self.n_heads   = n_heads
        self.head_dim  = embed_dim // n_heads
        
        self.key_proj   = nn.Linear(self.head_dim, self.head_dim)
        self.value_proj = nn.Linear(self.head_dim, self.head_dim)
        self.query_proj = nn.Linear(self.head_dim, self.head_dim)
        self.pos_proj   = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.u_bias = nn.Parameter(torch.Tensor(self.n_heads, self.head_dim))
        self.v_bias = nn.Parameter(torch.Tensor(self.n_heads, self.head_dim))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, key: Tensor, value:Tensor, query: Tensor, pos_embed: Tensor, mask: Tensor = None) -> Tensor:
        '''
        input: key & value: (batch_size, seq_len_k, embed_dim)
                     query: (batch_size, seq_len_q, embed_dim)
        '''
        batch_size, seq_len_k, _ = key.size()
        _, seq_len_q, _ = query.size()
        
        # 1. Divide embed_dim by number of heads: Each head will have a dimension of single_head_dim (embed_dim = n_heads * head_dim)
        key       =       key.view(batch_size, -1, self.n_heads, self.head_dim) # (batch_size, seq_len_k, n_heads, head_dim)
        value     =     value.view(batch_size, -1, self.n_heads, self.head_dim) # (batch_size, seq_len_k, n_heads, head_dim)
        query     =     query.view(batch_size, -1, self.n_heads, self.head_dim) # (batch_size, seq_len_q, n_heads, head_dim)
        pos_embed = pos_embed.view(batch_size, -1, self.n_heads, self.head_dim) # (batch_size, seq_len_q, n_heads, head_dim)
        
        # 2. Linear Projections
        key       = self.key_proj(key)           # (batch_size, seq_len_k, n_heads, head_dim)
        value     = self.value_proj(value)       # (batch_size, seq_len_k, n_heads, head_dim)
        query     = self.query_proj(query)       # (batch_size, seq_len_q, n_heads, head_dim)
        pos_embed = self.pos_proj(pos_embed)     # (batch_size, seq_len_q, n_heads, head_dim)
        
        # 3. Calculate Attention Score
        # query + u_bias & query + v_bias : (batch_size, seq_len_q, n_heads, head_dim)
        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.permute(0, 2, 3, 1))        # (batch_size, n_heads, seq_len_q, seq_len_k)
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embed.permute(0, 2, 3, 1))      # (batch_size, n_heads, seq_len_q, seq_len_q)
        pos_score = self._relative_shift(pos_score)                                                         # (batch_size, n_heads, seq_len_q, seq_len_q)
        
        score = (content_score + pos_score) / np.sqrt(self.embed_dim)
        
        ## Masking before feed into Softmax
        ## fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # (batch_size, seq_len, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
            score = score.masked_fill(mask == 0, float("-1e20"))
        
        ## Division by square root of dim_k and Apply Softmax
        attn_score = nn.Softmax(dim=-1)(score) # (batch_size, n_heads, seq_len_q, seq_len_k)
        
        context = torch.matmul(attn_score, value.transpose(1, 2)).transpose(1, 2) # (batch_size, seq_len_q, n_heads, dim_head)
        
        # 4. Reshape to Concatenate
        context = context.contiguous().view(batch_size, -1, self.embed_dim)       # (batch_size, seq_len, embed_dim)
        
        # 5. Linear Layer
        context = self.out(context) # (batch_size, seq_len, embed_dim)
        
        return context

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score

class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.
    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    Max_Len can be any huge number because we will use only required length(embed_dim) from the whole.
    """
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]