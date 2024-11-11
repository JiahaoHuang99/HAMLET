import torch
import torch.nn as nn
import torch.nn.functional as F



class DiffusionReaction2DEncoder(nn.Module):
    """
        Diffusion Reaction 2D Encoder
    """

    def __init__(self, net_params):
        super().__init__()

        # Parameter
        self.net_params = net_params

        self.in_dim = net_params['in_dim']
        self.hidden_dim_enc = net_params['hidden_dim_enc']
        self.hidden_dim_enc_out = net_params['hidden_dim_enc_out']
        self.out_dim = net_params['out_dim']
        self.num_heads = net_params['num_heads']
        self.num_layers = net_params['num_layers']
        self.in_feat_dropout = net_params['in_feat_dropout']
        self.dropout = net_params['dropout']
        self.use_ball_connectivity = net_params['use_ball_connectivity']
        self.r = net_params['r']
        self.use_knn = net_params['use_knn']
        self.k = net_params['k']
        assert self.use_ball_connectivity != self.use_knn, "use_ball_connectivity and use_knn cannot be both True"
        self.norm_type = net_params['norm_type']
        self.residual = net_params['residual']
        self.relative_emb = net_params['relative_emb']
        self.relative_emb_dim = net_params['relative_emb_dim']
        self.min_freq = net_params['min_freq']
        self.cat_pos = net_params['cat_pos']
        self.device = net_params['device']

        # Projection
        self.proj_h = nn.Linear(in_features=self.in_dim, out_features=self.hidden_dim_enc)
        self.proj_h_add = nn.Linear(in_features=2, out_features=self.hidden_dim_enc)

        # Dropout
        self.in_feat_dropout = nn.Dropout(self.in_feat_dropout)

        if self.use_knn:
            from layers.input_encoder_layer_dy_graph import InputEncoderLayer
        elif self.use_ball_connectivity:
            from layers.input_encoder_layer_stat_graph import InputEncoderLayer
        else:
            raise NotImplementedError

        # Encoder
        self.input_encoder_layers = nn.ModuleList([InputEncoderLayer(in_dim=self.hidden_dim_enc,
                                                                     out_dim=self.hidden_dim_enc,
                                                                     num_heads=self.num_heads,
                                                                     dropout=self.dropout,
                                                                     use_ball_connectivity=self.use_ball_connectivity,
                                                                     r=self.r,
                                                                     use_knn=self.use_knn,
                                                                     k=self.k,
                                                                     min_freq=self.min_freq,
                                                                     relative_emb=self.relative_emb,
                                                                     relative_emb_dim=self.relative_emb_dim,
                                                                     cat_pos=self.cat_pos,
                                                                     norm_type=self.norm_type,
                                                                     residual=self.residual)
                                                   for _ in range(self.num_layers - 1)])

        self.to_out = nn.Linear(self.hidden_dim_enc, self.hidden_dim_enc_out, bias=False)

    def forward(self, h, input_pos=None, g=None):

        bs = h.shape[0]

        # Embedding
        h = self.proj_h(h)  # (B, N, D) -> (B, N, D')

        # Dropout
        h = self.in_feat_dropout(h)

        # Encoder
        for graph_transformer_layer in self.input_encoder_layers:
            h = graph_transformer_layer(h, pos=input_pos, g=g)

        h_out = self.to_out(h)

        return h_out  # (B, N, D')

