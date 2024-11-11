import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np
from einops import rearrange, repeat

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
                 relative_emb=False,
                 relative_emb_dim=None,
                 min_freq=None,
                 cat_pos=False):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        self.min_freq = min_freq
        self.cat_pos = cat_pos

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
            self.emb_module = RotaryEmbedding(out_dim // relative_emb_dim, min_freq=min_freq, scale=1)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h, pos=None):

        bs = g.batch_size
        h = rearrange(h, '(bs n) c -> bs n c', bs=bs)
        bs, N, C = h.shape
        device = h.device
        assert C == self.in_dim

        Q_h = self.Q(h).view(bs, self.num_heads, N, self.out_dim)
        K_h = self.K(h).view(bs, self.num_heads, N, self.out_dim)
        V_h = self.V(h).view(bs, self.num_heads, N, self.out_dim)

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

            Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
            K_h = K_h.view(-1, self.num_heads, self.out_dim)
            V_h = V_h.view(-1, self.num_heads, self.out_dim)

        elif self.cat_pos:
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.num_heads , 1, 1])
            Q_h, K_h, V_h = [torch.cat([pos, x], dim=-1) for x in (Q_h, K_h, V_h)]

            Q_h = Q_h.view(-1, self.num_heads, self.out_dim + 2)
            K_h = K_h.view(-1, self.num_heads, self.out_dim + 2)
            V_h = V_h.view(-1, self.num_heads, self.out_dim + 2)


        else:
            Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
            K_h = K_h.view(-1, self.num_heads, self.out_dim)
            V_h = V_h.view(-1, self.num_heads, self.out_dim)

        g.ndata['Q_h'] = Q_h
        g.ndata['K_h'] = K_h
        g.ndata['V_h'] = V_h

        self.propagate_attention(g)

        attn_out = g.ndata['wV'] / g.ndata['z']

        if not self.cat_pos:
            attn_out = attn_out.view(-1, self.num_heads * self.out_dim)
        else:
            attn_out = attn_out.view(-1, self.num_heads * (self.out_dim + 2))

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
                 norm_type='batch',
                 residual=True,
                 use_bias=False,
                 relative_emb=False,
                 relative_emb_dim=None,
                 min_freq=None,
                 cat_pos=False,):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.norm_type = norm_type
        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        self.min_freq = min_freq
        self.cat_pos = cat_pos
        self.residual = residual

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
        else:
            raise ValueError('Norm type {} not supported'.format(self.norm_type))

        self.attention = MultiHeadAttentionLayer(in_dim=in_dim,
                                                 out_dim=out_dim//num_heads,
                                                 num_heads=num_heads,
                                                 use_bias=use_bias,
                                                 relative_emb=relative_emb,
                                                 relative_emb_dim=relative_emb_dim,
                                                 min_freq=min_freq,
                                                 cat_pos=cat_pos)

        if not cat_pos:
            self.O = nn.Linear(out_dim, out_dim)
        else:
            self.O = nn.Linear(out_dim + 2 * num_heads, out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

    def forward(self, g, h,  pos=None):

        h_in1 = h.clone()  # for first residual connection

        # multi-head attention out
        h = self.attention(g, h, pos=pos)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)

        if self.residual:
            h = h_in1 + h

        h = self.norm1(h)

        h_in2 = h.clone()  # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h

        h = self.norm2(h)

        return h
        
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
