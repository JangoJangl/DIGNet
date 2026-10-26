"""
Minimal SO(3) helpers in plain PyTorch (batched, differentiable, no Theseus / PyTorch3D dependency).
Rotation vectors (axis * angle) are the so(3) coordinates used by se3_pseudo_log_map_representation.
"""
import torch


def hat(w):
    """ [..., 3] -> skew symmetric [..., 3, 3] """
    zero = torch.zeros_like(w[..., 0])
    return torch.stack([
        torch.stack([zero, -w[..., 2], w[..., 1]], dim=-1),
        torch.stack([w[..., 2], zero, -w[..., 0]], dim=-1),
        torch.stack([-w[..., 1], w[..., 0], zero], dim=-1),
    ], dim=-2)


def exp_map(w):
    """ Rodrigues formula, rotation vector [..., 3] -> rotation matrix [..., 3, 3] """
    theta = torch.linalg.vector_norm(w, dim=-1, keepdim=True)[..., None]  # [..., 1, 1]
    K = hat(w)
    small = theta < 1e-4
    theta_safe = torch.where(small, torch.ones_like(theta), theta)
    a = torch.where(small, 1 - theta ** 2 / 6, torch.sin(theta_safe) / theta_safe)
    b = torch.where(small, 0.5 - theta ** 2 / 24, (1 - torch.cos(theta_safe)) / theta_safe ** 2)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
    return eye + a * K + b * (K @ K)


def matrix_to_quaternion(R):
    """ Rotation matrix [..., 3, 3] -> unit quaternion (w, x, y, z) [..., 4] with w >= 0, robust for all angles """
    m = R
    t = torch.stack([
        1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2],
        1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
        1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
        1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
    ], dim=-1)
    # candidate quaternions, each well conditioned when its own component is the largest (Shepperd's method)
    candidates = torch.stack([
        torch.stack([t[..., 0], m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1),
        torch.stack([m[..., 2, 1] - m[..., 1, 2], t[..., 1], m[..., 1, 0] + m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0]], -1),
        torch.stack([m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] + m[..., 0, 1], t[..., 2], m[..., 2, 1] + m[..., 1, 2]], -1),
        torch.stack([m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 2, 1] + m[..., 1, 2], t[..., 3]], -1),
    ], dim=-2)
    best = t.argmax(dim=-1)
    q = torch.gather(candidates, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    return torch.where(q[..., :1] < 0, -q, q)


def log_map(R):
    """ Rotation matrix [..., 3, 3] -> rotation vector [..., 3] with angle in [0, pi] """
    q = matrix_to_quaternion(R)
    v = q[..., 1:]
    sin_half = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sin_half, q[..., :1])
    small = sin_half < 1e-6
    scale = torch.where(small, 2.0 / q[..., :1].clamp(min=1e-6), angle / sin_half.clamp(min=1e-12))
    return v * scale


def random_rotation(n, dtype=torch.float32, device='cpu'):
    """ n rotation matrices, uniformly (Haar) distributed """
    q = torch.randn(n, 4, dtype=dtype, device=device)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def matrix_to_rotation_6d(R):
    """ Continuous 6D rotation features (first two columns, Zhou et al. 2019) [..., 3, 3] -> [..., 6] """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)
