"""
Compact Point Transformer V3 style encoder (Wu et al., CVPR 2024) in plain PyTorch.

Kept from PTv3: serialisation of the cloud along space filling curves (z-order and transposed z-order),
attention inside non-overlapping patches of the serialised sequence, a different curve per block
(shift order, shuffled during training), a convolutional positional encoding in front of every attention
layer (xCPE, here a kNN point convolution instead of a sparse convolution), pre-norm blocks with LayerNorm,
pooling along the serialisation with BatchNorm.
Simplified: dense [B, N, C] tensors with a fixed number of points per cloud, pooling merges `stride`
consecutive points of the z-order sequence, no Hilbert curves, no spconv / flash attention.
Optimisations that need extra dependencies are listed in knowledgebase/research/point-cloud-encoders.md.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.vision_encoder.pointnet2_local import relative_tokens


def morton_code(xyz, grid_size, bits=10):
    """ z-order code of points [B, N, 3] quantised to grid_size (per cloud, origin at its minimum) """
    q = ((xyz - xyz.min(dim=1, keepdim=True).values) / grid_size).long().clamp(0, 2 ** bits - 1)
    code = torch.zeros_like(q[..., 0])
    for bit in range(bits):
        for axis in range(3):
            code |= ((q[..., axis] >> bit) & 1) << (3 * bit + axis)
    return code


def serialization_orders(xyz, grid_size):
    """ Permutations [B, N] that sort the points along z-order and transposed (y, x, z) z-order """
    return [torch.argsort(morton_code(xyz, grid_size), dim=1),
            torch.argsort(morton_code(xyz[..., [1, 0, 2]], grid_size), dim=1)]


def gather(x, idx):
    return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))


class PointConvCPE(nn.Module):
    """ xCPE stand-in: max over the kNN neighbourhood of linear(neighbour feature) + linear(relative position) """

    def __init__(self, channels):
        super().__init__()
        self.feat = nn.Linear(channels, channels)
        self.pos = nn.Linear(3, channels)
        self.out = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels))

    def forward(self, x, xyz, knn):
        B, N, K = knn.shape
        flat = knn.reshape(B, N * K)
        neighbours = gather(self.feat(x), flat).view(B, N, K, -1)
        rel = gather(xyz, flat).view(B, N, K, 3) - xyz[:, :, None]
        return self.out((neighbours + self.pos(rel)).max(dim=2).values)


class SerializedAttention(nn.Module):
    """ Multi-head attention inside patches of patch_size consecutive points of a serialisation order """

    def __init__(self, channels, num_heads, patch_size):
        super().__init__()
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x, order):
        B, N, C = x.shape
        P = min(self.patch_size, N)
        pad = (-N) % P
        idx = torch.cat([order, order[:, :pad]], dim=1) if pad else order  # pad by reusing the first points
        h = gather(x, idx)
        qkv = self.qkv(h).view(B, -1, P, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(3, 0, 1, 4, 2, 5)  # each [B, patches, heads, P, head_dim]
        h = F.scaled_dot_product_attention(q, k, v).permute(0, 1, 3, 2, 4).reshape(B, -1, C)
        out = torch.zeros_like(x).scatter(1, order[..., None].expand(-1, -1, C), h[:, :N])  # undo the ordering
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, channels, num_heads, patch_size, mlp_ratio=4):
        super().__init__()
        self.cpe = PointConvCPE(channels)
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = SerializedAttention(channels, num_heads, patch_size)
        self.norm_mlp = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, mlp_ratio * channels), nn.GELU(), nn.Linear(mlp_ratio * channels, channels))

    def forward(self, x, xyz, knn, order):
        x = x + self.cpe(x, xyz, knn)
        x = x + self.attn(self.norm_attn(x), order)
        return x + self.mlp(self.norm_mlp(x))


class SerializedPooling(nn.Module):
    """ Merges `stride` consecutive points of the z-order sequence: max of features, mean of positions """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x, xyz, order):
        B, N, _ = x.shape
        n = N // self.stride
        order = order[:, :n * self.stride]
        x = gather(self.proj(x), order).view(B, n, self.stride, -1).max(dim=2).values
        xyz = gather(xyz, order).view(B, n, self.stride, 3).mean(dim=2)
        x = F.gelu(self.norm(x.transpose(1, 2)).transpose(1, 2))
        return x, xyz


class PTv3Encoder(nn.Module):
    """
    forward(pc) -> (condition, B): tokens [B, N / prod(strides), 3 + output_dim] if return_tokens,
    otherwise a max pooled global vector [B, output_dim].
    """

    def __init__(self, output_dim=256, in_channels=3, channels=(32, 64, 128, 256), depths=(1, 1, 2, 1),
                 num_heads=(2, 4, 8, 16), strides=(2, 2, 2), patch_size=128, k_neighbors=16, grid_size=0.002,
                 return_tokens=True, **kwargs):
        super().__init__()
        assert len(channels) == len(depths) == len(num_heads) == len(strides) + 1
        self.in_channels = in_channels
        self.k_neighbors = k_neighbors
        self.grid_size = grid_size
        self.return_tokens = return_tokens
        self.embed = nn.Sequential(nn.Linear(in_channels, channels[0]), nn.LayerNorm(channels[0]), nn.GELU())
        self.stages = nn.ModuleList([
            nn.ModuleList([Block(channels[s], num_heads[s], patch_size) for _ in range(depths[s])])
            for s in range(len(channels))
        ])
        self.pools = nn.ModuleList([SerializedPooling(channels[s], channels[s + 1], strides[s]) for s in range(len(strides))])
        self.head = nn.Sequential(nn.LayerNorm(channels[-1]), nn.Linear(channels[-1], output_dim))

    def forward(self, pc):
        xyz = pc[..., :3]
        x = self.embed(pc[..., :self.in_channels])
        for s, blocks in enumerate(self.stages):
            orders = serialization_orders(xyz, self.grid_size * 2 ** s)
            if self.training:  # shuffle order: random curve sequence per forward pass
                orders = [orders[i] for i in torch.randperm(len(orders)).tolist()]
            knn = torch.cdist(xyz, xyz).topk(min(self.k_neighbors, xyz.shape[1]), dim=-1, largest=False).indices
            for b, block in enumerate(blocks):
                x = block(x, xyz, knn, orders[b % len(orders)])
            if s < len(self.pools):
                x, xyz = self.pools[s](x, xyz, orders[0])
        tokens = self.head(x)
        if self.return_tokens:
            return torch.cat([xyz, tokens], dim=-1), pc.shape[0]
        return tokens.max(dim=1).values, pc.shape[0]

    def compute_features(self, condition, t, position):
        return relative_tokens(condition, position) if self.return_tokens else condition
