import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np
from utils.gcn_lib import *
from einops import rearrange, repeat, reduce
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

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_heads,
                 use_bias,
                 k=9,
                 relative_emb=False,
                 relative_emb_dim=None,
                 min_freq=None,
                 cat_pos=False,
                 scale=1.):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.k = k
        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        self.min_freq = min_freq
        self.cat_pos = cat_pos
        self.scale = scale

        self.dilated_knn_graph = DenseDilatedKnnGraph(k=k,
                                                      dilation=1,
                                                      stochastic=False,
                                                      epsilon=0.0)

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        if self.relative_emb:
            assert not self.cat_pos
            self.emb_module = RotaryEmbedding(out_dim // relative_emb_dim, min_freq=min_freq, scale=scale)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, h, pos=None):

        bs, N, C = h.shape
        device = h.device
        assert C == self.in_dim

        # Construct graph
        edge_index = self.dilated_knn_graph(h.view(bs, C, N, 1)).view(2, bs, -1)  # (2, B, N*K)
        self.edge_index = edge_index.detach()

        graphs = []
        for b in range(bs):
            g = dgl.graph(([], [])).to(device)
            g.add_nodes(N)
            g.ndata['feat'] = h[b]
            g.add_edges(edge_index[0, b], edge_index[1, b])
            graphs.append(g)

        # Batch the graphs
        g = dgl.batch(graphs)

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        Q_h = rearrange(Q_h, 'b n (h d) -> b h n d', h=self.num_heads, d=self.out_dim)
        K_h = rearrange(K_h, 'b n (h d) -> b h n d', h=self.num_heads, d=self.out_dim)
        V_h = rearrange(V_h, 'b n (h d) -> b h n d', h=self.num_heads, d=self.out_dim)

        if self.relative_emb:
            if self.relative_emb_dim == 2:
                freqs_x = self.emb_module.forward(pos[..., 0], h.device)
                freqs_y = self.emb_module.forward(pos[..., 1], h.device)
                freqs_x = repeat(freqs_x, 'b n d -> b h n d', h=Q_h.shape[1])
                freqs_y = repeat(freqs_y, 'b n d -> b h n d', h=Q_h.shape[1])
                Q_h = apply_2d_rotary_pos_emb(Q_h, freqs_x, freqs_y)
                K_h = apply_2d_rotary_pos_emb(K_h, freqs_x, freqs_y)
            elif self.relative_emb_dim == 1:
                assert pos.shape[-1] == 1
                freqs = self.emb_module.forward(pos[..., 0], h.device)
                freqs = repeat(freqs, 'b n d -> b h n d', h=Q_h.shape[1])
                Q_h = apply_rotary_pos_emb(Q_h, freqs)
                K_h = apply_rotary_pos_emb(K_h, freqs)
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')

        elif self.cat_pos:
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.num_heads , 1, 1])
            Q_h, K_h, V_h = [torch.cat([pos, x], dim=-1) for x in (Q_h, K_h, V_h)]

        else:
            pass

        Q_h = rearrange(Q_h, 'b h n d -> (n b) h d')
        K_h = rearrange(K_h, 'b h n d -> (n b) h d')
        V_h = rearrange(V_h, 'b h n d -> (n b) h d')

        g.ndata['Q_h'] = Q_h  # bs*N, num_heads, out_dim (out_dim+2)
        g.ndata['K_h'] = K_h  # bs*N, num_heads, out_dim (out_dim+2)
        g.ndata['V_h'] = V_h  # bs*N, num_heads, out_dim (out_dim+2)

        self.propagate_attention(g)

        attn_out = g.ndata['wV'] / g.ndata['z']

        attn_out = rearrange(attn_out, '(n b) h d -> b n (h d)', n=N, b=bs, h=self.num_heads, d=self.out_dim)

        return attn_out


class InputEncoderLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_heads,
                 dropout=0.0,
                 use_ball_connectivity=False,
                 r=0.1,
                 use_knn=True,
                 k=101,
                 norm_type='batch',
                 residual=True,
                 use_bias=False,
                 relative_emb=False,
                 relative_emb_dim=None,
                 min_freq=None,
                 cat_pos=False,
                 scale=1):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.k = k
        self.norm_type = norm_type
        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        self.min_freq = min_freq
        self.cat_pos = cat_pos
        self.residual = residual
        self.scale = scale

        if self.norm_type == 'layer':
            self.norm1 = nn.LayerNorm(out_dim)
            self.norm2 = nn.LayerNorm(out_dim)
        elif self.norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(out_dim)
            self.norm2 = nn.BatchNorm1d(out_dim)
        elif self.norm_type == 'group':
            self.norm1 = nn.GroupNorm(4, out_dim)
            self.norm2 = nn.GroupNorm(4, out_dim)
        elif self.norm_type == 'instance':
            self.norm1 = nn.InstanceNorm1d(out_dim)
            self.norm2 = nn.InstanceNorm1d(out_dim)
        elif self.norm_type == 'no':
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            raise ValueError('Norm type {} not supported'.format(self.norm_type))

        self.attention = MultiHeadAttentionLayer(in_dim=in_dim,
                                                 out_dim=out_dim//num_heads,
                                                 num_heads=num_heads,
                                                 use_bias=use_bias,
                                                 k=k,
                                                 relative_emb=relative_emb,
                                                 relative_emb_dim=relative_emb_dim,
                                                 min_freq=min_freq,
                                                 cat_pos=cat_pos,
                                                 scale=scale)

        if not cat_pos:
            self.O = nn.Linear(out_dim, out_dim)
        else:
            self.O = nn.Linear(out_dim + 2 * num_heads, out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)

        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

    def forward(self, h, pos=None, g=None):
        bs, N, C = h.shape

        h_in1 = h.clone()  # for first residual connection

        # multi-head attention out
        h = self.attention(h, pos=pos)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)

        if self.residual:
            h = h_in1 + h

        if self.norm_type in ['layer', 'no']:
            h = self.norm1(h)  # Bs, N, C
        elif self.norm_type in ['batch', 'instance']:
            h = self.norm1(h.view(bs, C, N)).view(bs, N, C)  # Bs, N, C -> Bs, C, N -> Bs, N, C
        elif self.norm_type in ['group']:
            raise NotImplementedError
        else:
            raise ValueError('Norm type {} not supported'.format(self.norm_type))

        h_in2 = h.clone()  # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h

        if self.norm_type in ['layer', 'no']:
            h = self.norm2(h)  # Bs, N, C
        elif self.norm_type in ['batch', 'instance']:
            h = self.norm2(h.view(bs, C, N)).view(bs, N, C)  # Bs, N, C -> Bs, C, N -> Bs, N, C
        elif self.norm_type in ['group']:
            raise NotImplementedError
        else:
            raise ValueError('Norm type {} not supported'.format(self.norm_type))

        return h.view(bs, -1, self.out_channels)
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels,
                                                                                   self.num_heads,
                                                                                   self.residual)

# New position encoding module
# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)
