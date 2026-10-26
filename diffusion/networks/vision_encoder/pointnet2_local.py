import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.vision_encoder.pointnet2 import (PointNetSetAbstractionMsg, farthest_point_sample, index_points,
                                               query_ball_point)


class NormalizedSetAbstractionMsg(PointNetSetAbstractionMsg):
    """
    PointNet++ MSG set abstraction with relative positions divided by the ball radius (PointNeXt recipe),
    so all scales see offsets in [-1, 1] instead of a few millimetres.
    """

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            group_idx = query_ball_point(radius, self.nsample_list[i], xyz, new_xyz)
            grouped_xyz = (index_points(xyz, group_idx) - new_xyz.view(B, S, 1, C)) / radius
            if points is not None:
                grouped_points = torch.cat([index_points(points, group_idx), grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for conv, bn in zip(self.conv_blocks[i], self.bn_blocks[i]):
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points_list.append(torch.max(grouped_points, 2)[0])
        return new_xyz.permute(0, 2, 1), torch.cat(new_points_list, dim=1)


class PointNet2Local(nn.Module):
    """
    PointNet++ encoder that keeps local features.

    Two MSG set abstraction levels (512 -> num_tokens centroids) with radius normalised offsets. The output
    is a set of tokens (centroid position + feature) instead of a single max pooled vector, so a
    CrossAttentionDenoiser can look at the geometry around the current grasp position.

    forward(pc) -> (condition, B):  condition [B, num_tokens, 3 + output_dim] if return_tokens,
                                    otherwise a global vector [B, output_dim] (drop-in for the MLP denoiser).
    """

    def __init__(self, output_dim=256, radius=0.02, num_tokens=128, use_normals=False, return_tokens=True, width=1,
                 **kwargs):
        super().__init__()
        c = lambda channels: [[int(width * ch) for ch in mlp] for mlp in channels]  # width multiplier of all MLPs
        in_channel = 3 if use_normals else 0
        self.use_normals = use_normals
        self.return_tokens = return_tokens
        self.output_dim = output_dim
        mlp1 = c([[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        mlp2 = c([[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa1 = NormalizedSetAbstractionMsg(512, [radius, 2 * radius, 4 * radius], [16, 32, 64], in_channel, mlp1)
        self.sa2 = NormalizedSetAbstractionMsg(num_tokens, [2 * radius, 4 * radius, 8 * radius], [16, 32, 64],
                                               sum(m[-1] for m in mlp1), mlp2)
        self.proj = nn.Sequential(nn.Linear(sum(m[-1] for m in mlp2), output_dim), nn.LayerNorm(output_dim), nn.GELU(),
                                  nn.Linear(output_dim, output_dim))

    def forward(self, pc):
        xyz = pc[..., :3].transpose(1, 2)
        features = pc[..., 3:6].transpose(1, 2) if self.use_normals else None
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        tokens = self.proj(l2_points.transpose(1, 2))  # [B, M, C]
        if self.return_tokens:
            return torch.cat([l2_xyz.transpose(1, 2), tokens], dim=-1), pc.shape[0]
        return tokens.max(dim=1).values, pc.shape[0]

    def compute_features(self, condition, t, position):
        return relative_tokens(condition, position) if self.return_tokens else condition


def relative_tokens(condition, position):
    """ Expresses the token positions relative to the current grasp position [B, 3] """
    return torch.cat([condition[..., :3] - position[:, None, :].to(condition.dtype), condition[..., 3:]], dim=-1)
