import torch
import torch.nn as nn


class AdaptiveEmbedding()


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super(PositionalEmbedding, self).__init__()
        self.emb_dim = emb_dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, emb_dim, 2.0) / emb_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_pos):
        '''
        seq_pos: reversed positions for input sequence with shape [seq_len].
        '''
        sin_inp = torch.ger(seq_pos, self.inv_freq)
        pos_emb = torch.cat([sin_inp.sin(), sin_inp.cos()], dim=-1)
        return pos_emb


class RelMultiHeadAttn(nn.Module):

    def __init__(self, q_in_features, k_in_features, v_in_features, num_heads, d_model):
        super(RelMultiHeadAttn, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(q_in_features, d_model)
        self.wk = nn.Linear(k_in_features, d_model)
        self.wv = nn.Linear(v_in_features, d_model)
        

    def forward(self, q, k, v, g_content_bias, g_pos_bias):
        batch_size = q.size(0)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        return q, k, v
    
    def split_heads(self, x, batch_size):
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x.permute(0, 2, 1, 3)

emb_dim = 100
pos_emb_func = PositionalEmbedding(emb_dim)
seq_pos = torch.arange(10-1, -1, -1.0, dtype=torch.float32)
pos_emb = pos_emb_func(seq_pos)

num_heads = 3
d_model = 24
depth = d_model // num_heads
q = torch.randn(5, 6, 10)
k = torch.randn(5, 4, 12)
v = k

g_content_bias = nn.Parameter(torch.Tensor(num_heads, depth))
nn.init.xavier_normal_(g_content_bias)
g_pos_bias = nn.Parameter(torch.Tensor(num_heads, depth))
nn.init.xavier_normal_(g_pos_bias)

rel = RelMultiHeadAttn(q.size(-1), k.size(-1), k.size(-1), num_heads, d_model)

q1, k1, v1 = rel(q, k, v, None, None)