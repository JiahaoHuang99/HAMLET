import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np
from einops import rearrange

"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func



"""
    Single Attention Head
"""


# In the same graph
class CrossFormerAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias=False):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
    
    def forward(self, g, h, g_add=None, h_add=None):
        
        Q_h = self.Q(h_add)
        K_h = self.K(h)
        V_h = self.V(h)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)

        # FIXME: Check here
        # g.ndata['V_h_add'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        # get number of nodes of the first graph in a batch of graphs
        head_out = g.ndata['wV'] / g.batch_num_nodes()[0]
        
        return head_out


class CrossFormerAttentionM2(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias=False):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def forward(self, h, h_add, batch_size):
        Q_h = self.Q(h_add)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [batch_sizel, num_heads, num_nodes, feat_dim] to get projections for multi-head attention
        Q_h = rearrange(Q_h, '(b nq) (h d) -> b h nq d', b=batch_size, h=self.num_heads, d=self.out_dim)  # nq: node of a query graph
        K_h = rearrange(K_h, '(b n) (h d) -> b h n d', b=batch_size, h=self.num_heads, d=self.out_dim)  # n: node of a graph
        V_h = rearrange(V_h, '(b n) (h d) -> b h n d', b=batch_size, h=self.num_heads, d=self.out_dim)  # n: node of a graph

        attn = torch.matmul(Q_h, K_h.transpose(-2, -1))  # Compute attention weights  # (b h nq d) * (b h d n) = (b h nq n)

        attn = F.softmax(attn, dim=-1)  # Apply softmax to get attention probabilities (b h nq n)

        attn_out = torch.matmul(attn, V_h)  # (b h nq n) * (b h n d) = (b h nq d)

        attn_out = attn_out.view(batch_size, -1, self.out_dim * self.num_heads)

        return attn_out  # (b nq d)


class CrossFormerAttentionM2NoSoft(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias=False):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def forward(self, h, h_add, batch_size):
        Q_h = self.Q(h_add)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [batch_sizel, num_heads, num_nodes, feat_dim] to get projections for multi-head attention
        Q_h = rearrange(Q_h, '(b nq) (h d) -> b h nq d', b=batch_size, h=self.num_heads, d=self.out_dim)  # nq: node of a query graph
        K_h = rearrange(K_h, '(b n) (h d) -> b h n d', b=batch_size, h=self.num_heads, d=self.out_dim)  # n: node of a graph
        V_h = rearrange(V_h, '(b n) (h d) -> b h n d', b=batch_size, h=self.num_heads, d=self.out_dim)  # n: node of a graph

        attn = torch.matmul(Q_h, K_h.transpose(-2, -1))  # Compute attention weights  # (b h nq d) * (b h d n) = (b h nq n)

        # attn = F.softmax(attn, dim=-1)  # Apply softmax to get attention probabilities (b h nq n)

        attn_out = torch.matmul(attn, V_h)  # (b h nq n) * (b h n d) = (b h nq d)

        attn_out = attn_out.view(batch_size, -1, self.out_dim * self.num_heads)

        return attn_out  # (b nq d)


class CrossFormerAttentionM3(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias=False):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def forward(self, h, h_add, batch_size):
        Q_h = self.Q(h_add)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [batch_sizel, num_heads, num_nodes, feat_dim] to get projections for multi-head attention
        Q_h = rearrange(Q_h, '(b nq) (h d) -> b h nq d', b=batch_size, h=self.num_heads, d=self.out_dim)  # nq: node of a query graph
        K_h = rearrange(K_h, '(b n) (h d) -> b h n d', b=batch_size, h=self.num_heads, d=self.out_dim)  # n: node of a graph
        V_h = rearrange(V_h, '(b n) (h d) -> b h n d', b=batch_size, h=self.num_heads, d=self.out_dim)  # n: node of a graph

        dots = torch.matmul(K_h.transpose(-2, -1), V_h)  # Compute attention weights  # (b h d n) * (b h n d) = (b h d d)

        attn_out = torch.matmul(Q_h, dots) * (1./Q_h.shape[2])  # (b h nq d) * (b h d d) = (b h nq d)

        attn_out = attn_out.view(batch_size, -1, self.out_dim * self.num_heads)

        return attn_out  # (b nq d)


class CrossFormerAttentionM3NoScale(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias=False):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def forward(self, h, h_add, batch_size):
        Q_h = self.Q(h_add)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [batch_sizel, num_heads, num_nodes, feat_dim] to get projections for multi-head attention
        Q_h = rearrange(Q_h, '(b nq) (h d) -> b h nq d', b=batch_size, h=self.num_heads, d=self.out_dim)  # nq: node of a query graph
        K_h = rearrange(K_h, '(b n) (h d) -> b h n d', b=batch_size, h=self.num_heads, d=self.out_dim)  # n: node of a graph
        V_h = rearrange(V_h, '(b n) (h d) -> b h n d', b=batch_size, h=self.num_heads, d=self.out_dim)  # n: node of a graph

        dots = torch.matmul(K_h.transpose(-2, -1), V_h)  # Compute attention weights  # (b h d n) * (b h n d) = (b h d d)

        attn_out = torch.matmul(Q_h, dots)  # (b h nq d) * (b h d d) = (b h nq d)

        attn_out = attn_out.view(batch_size, -1, self.out_dim * self.num_heads)

        return attn_out  # (b nq d)

########### From OFormer ###########

from layers.attention_module import CrossLinearAttention, FeedForward

class CrossFormer(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,
                 heads,
                 dim_head,
                 mlp_dim,
                 residual=True,
                 use_ffn=True,
                 use_ln=False,
                 relative_emb=False,
                 scale=1.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 dropout=0.,
                 cat_pos=False,
                 ):
        super().__init__()

        self.cross_attn_module = CrossLinearAttention(dim,
                                                      attn_type,
                                                      heads=heads,
                                                      dim_head=dim_head,
                                                      dropout=dropout,
                                                      relative_emb=relative_emb,
                                                      scale=scale,
                                                      relative_emb_dim=relative_emb_dim,
                                                      min_freq=min_freq,
                                                      init_method='orthogonal',
                                                      cat_pos=cat_pos,
                                                      pos_dim=relative_emb_dim,
                                                      )
        self.use_ln = use_ln
        self.residual = residual
        self.use_ffn = use_ffn

        if self.use_ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        if self.use_ffn:
            self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, z, x_pos=None, z_pos=None):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        if self.use_ln:
            z = self.ln1(z)
            if self.residual:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos)) + x
            else:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos))
        else:
            if self.residual:
                x = self.cross_attn_module(x, z, x_pos, z_pos) + x
            else:
                x = self.cross_attn_module(x, z, x_pos, z_pos)

        if self.use_ffn:
            x = self.ffn(x) + x

        return x
