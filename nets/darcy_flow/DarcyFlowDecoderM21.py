import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal_
from layers.cross_transformer_layer import CrossFormer
import dgl
import numpy as np
from einops import rearrange


# code copied from: https://github.com/ndahlquist/pytorch-fourier-feature-networks
# author: Nic Dahlquist
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):

        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class DarcyFlowDecoder(nn.Module):
    """
        Darcy Flow Decoder
    """
    # Should define a recurrent neural network propagating on the graph
    def __init__(self, net_params):
        super().__init__()

        # Generalize to a convolution network on the graph
        self.hidden_dim_dec = net_params['hidden_dim_dec']
        self.out_dim = net_params['out_dim']
        self.num_heads = net_params['num_heads']
        self.decoding_depth = net_params['decoding_depth']
        self.residual = net_params['residual']
        self.relative_emb = net_params['relative_emb']
        self.relative_emb_dim = net_params['relative_emb_dim']
        self.min_freq = net_params['min_freq']
        self.cat_pos = net_params['cat_pos']

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.hidden_dim_dec//2, scale=1),
            # nn.Linear(2, self.hidden_dim_dec, bias=False),
            # nn.GELU(),
            nn.Linear(self.hidden_dim_dec, self.hidden_dim_dec, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_dim_dec, self.hidden_dim_dec, bias=False),
            # nn.Dropout(0.05),
        )

        self.decoding_transformer = CrossFormer(dim=self.hidden_dim_dec,
                                                attn_type='galerkin',
                                                heads=self.num_heads,
                                                dim_head=self.hidden_dim_dec,
                                                mlp_dim=self.hidden_dim_dec,
                                                residual=self.residual,
                                                use_ffn=True,
                                                use_ln=True,
                                                relative_emb=self.relative_emb,
                                                scale=1.,
                                                relative_emb_dim=self.relative_emb_dim,
                                                min_freq=self.min_freq,
                                                dropout=0.,
                                                cat_pos=self.cat_pos,
                                                )

        # self.propagator = nn.ModuleList([
        #        nn.ModuleList([nn.LayerNorm(self.hidden_dim_dec),
        #        nn.Sequential(
        #             nn.Linear(self.hidden_dim_dec + 2, self.hidden_dim_dec, bias=False),
        #             nn.GELU(),
        #             nn.Linear(self.hidden_dim_dec, self.hidden_dim_dec, bias=False),
        #             nn.GELU(),
        #             nn.Linear(self.hidden_dim_dec, self.hidden_dim_dec, bias=False))])
        #     for _ in range(self.decoding_depth)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim_dec),
            nn.Linear(self.hidden_dim_dec, self.hidden_dim_dec // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_dim_dec // 2, self.out_dim, bias=False))

    # def propagate(self, h, propagate_pos):
    #     for layer in self.propagator:
    #         norm_fn, ffn = layer
    #         h = h + ffn(torch.concat((norm_fn(h), propagate_pos), dim=-1))
    #     return h

    def decode(self, h):
        h = self.to_out(h)
        return h

    def forward(self, h, input_pos, propagate_pos):

        # cross attention
        h_add = self.coordinate_projection.forward(propagate_pos)
        h_out = self.decoding_transformer.forward(x=h_add,
                                                  z=h,
                                                  x_pos=propagate_pos,
                                                  z_pos=input_pos)


        # propagate
        # h_out = self.propagate(h_out)

        h_out = self.decode(h_out)
        # h_out = self.decode(torch.cat((h_out, propagate_pos), dim=-1))
        return h_out

    def rollout(self, h, input_pos, propagate_pos, forward_steps):
        history = []

        # cross attention
        h_add = self.coordinate_projection.forward(propagate_pos)
        h = self.decoding_transformer.forward(x=h_add,
                                              z=h,
                                              x_pos=propagate_pos,
                                              z_pos=input_pos)

        for step in range(forward_steps):
            h = self.propagate(h, propagate_pos)
            h_out = self.decode(h)
            history.append(h_out)
        return history

