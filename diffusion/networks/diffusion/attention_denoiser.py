import torch
import torch.nn as nn

from networks.diffusion.utils import SinusoidalPosEmb


class CrossAttentionDenoiser(nn.Module):
    """
    Denoiser / velocity network that attends from the noisy grasp to local point features.

    Expects the condition produced by a token embedder's compute_features (PointNet2Local, PTv3Encoder):
    [B, M, 3 + C] with the token positions relative to the current grasp position in the first 3 channels.
    Only the k tokens closest to the grasp take part in the cross-attention, a max pooled summary of all
    tokens is added to the query as global context.
    """

    def __init__(
            self,
            input_dim: int,
            cond_dim: int,
            output_dim: int = None,
            hidden_dim: int = 256,
            num_layers: int = 2,
            num_heads: int = 4,
            k_neighbors: int = 32,
            t_dim: int = 32,
            dropout: float = 0.0,
            device: str = 'cuda',
            **kwargs,
    ):
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        self.k_neighbors = k_neighbors
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(t_dim), nn.Linear(t_dim, 2 * t_dim), nn.Mish(), nn.Linear(2 * t_dim, t_dim))
        self.query_in = nn.Sequential(nn.Linear(input_dim + t_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, hidden_dim))
        self.token_in = nn.Linear(cond_dim, hidden_dim)
        self.pos_in = nn.Sequential(nn.Linear(3, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, hidden_dim))
        self.global_in = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm_q': nn.LayerNorm(hidden_dim),
                'norm_kv': nn.LayerNorm(hidden_dim),
                'attn': nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                'norm_mlp': nn.LayerNorm(hidden_dim),
                'mlp': nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim)),
            }) for _ in range(num_layers)
        ])
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
        self.to(device)

    def forward(self, x, t, condition):
        rel_pos, features = condition[..., :3], condition[..., 3:]
        tokens = self.token_in(features)
        global_feature = tokens.max(dim=1).values

        if self.k_neighbors < tokens.shape[1]:
            idx = torch.topk(torch.linalg.vector_norm(rel_pos, dim=-1), self.k_neighbors, dim=1, largest=False).indices
            tokens = torch.gather(tokens, 1, idx[..., None].expand(-1, -1, tokens.shape[-1]))
            rel_pos = torch.gather(rel_pos, 1, idx[..., None].expand(-1, -1, 3))
        tokens = tokens + self.pos_in(rel_pos)

        q = self.query_in(torch.cat([x, self.time_embedding(t)], dim=-1)) + self.global_in(global_feature)
        q = q[:, None]
        for layer in self.layers:
            kv = layer['norm_kv'](tokens)
            q = q + layer['attn'](layer['norm_q'](q), kv, kv, need_weights=False)[0]
            q = q + layer['mlp'](layer['norm_mlp'](q))
        return self.out(q[:, 0])

    def get_params(self):
        return self.parameters()
