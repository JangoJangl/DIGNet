"""
4x4 homogeneous transforms with numpy. Quaternions are (x, y, z, w), the convention of Isaac Lab 3.
"""
import numpy as np


def translation(x=0.0, y=0.0, z=0.0):
    t = np.eye(4)
    t[:3, 3] = (x, y, z)
    return t


def rpy(roll, pitch, yaw, xyz=(0.0, 0.0, 0.0)):
    """ URDF origin: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), then the translation xyz """
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    t = np.eye(4)
    t[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    t[:3, 3] = xyz
    return t


def invert(t):
    t = np.asarray(t, dtype=np.float64)
    inv = np.zeros_like(t)
    rot_t = np.swapaxes(t[..., :3, :3], -1, -2)
    inv[..., :3, :3] = rot_t
    inv[..., :3, 3] = -np.einsum('...ij,...j->...i', rot_t, t[..., :3, 3])
    inv[..., 3, 3] = 1.0
    return inv


def matrix_to_quat_xyzw(rot):
    """ rotation matrices [..., 3, 3] -> unit quaternions [..., 4] (x, y, z, w) with w >= 0 """
    rot = np.asarray(rot, dtype=np.float64)
    m = rot.reshape(-1, 3, 3)
    q = np.empty((len(m), 4))
    for i, r in enumerate(m):
        tr = np.trace(r)
        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            q[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            q[i] = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif r[1, 1] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            q[i] = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            q[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1
    return q.reshape(rot.shape[:-2] + (4,))


def quat_xyzw_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    rot = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)
    return rot.reshape(q.shape[:-1] + (3, 3))


def pose7_from_matrix(t):
    """ [..., 4, 4] -> [..., 7] position + quaternion (x, y, z, w), the root pose layout of Isaac Lab """
    t = np.asarray(t, dtype=np.float64)
    return np.concatenate([t[..., :3, 3], matrix_to_quat_xyzw(t[..., :3, :3])], axis=-1)


def matrix_from_pose7(p):
    p = np.asarray(p, dtype=np.float64)
    t = np.zeros(p.shape[:-1] + (4, 4))
    t[..., :3, :3] = quat_xyzw_to_matrix(p[..., 3:7])
    t[..., :3, 3] = p[..., :3]
    t[..., 3, 3] = 1.0
    return t
