import numpy as np
import torch
from scipy.spatial import ConvexHull


class Augmentation(object):
    """
    Data augmentation for object point clouds and their grasps.

    All methods work on an object centred point cloud [N, 3 + F] (xyz first) and grasp poses [G, 4, 4]
    expressed in the same frame. Every transformation is applied to the points and the grasps together,
    so that the grasp labels stay valid.
    """

    @staticmethod
    def random_rotation(dtype=torch.float64, device='cpu', generator=None):
        """ Uniformly distributed random rotation matrix (via a normalised random quaternion) """
        q = torch.randn(4, dtype=dtype, generator=generator).to(device)
        q = q / torch.linalg.vector_norm(q)
        w, x, y, z = q
        return torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ])

    @staticmethod
    def rotate(points, grasps, rotation):
        """
        Rotates points, normals (if present as features 3:6) and grasps around the origin

        :param points: [N, 3 + F] point cloud
        :param grasps: [G, 4, 4] grasp poses
        :param rotation: [3, 3] rotation matrix
        """
        rotation = rotation.to(device=points.device, dtype=points.dtype)
        points = points.clone()
        points[:, :3] = points[:, :3] @ rotation.T
        if points.shape[1] >= 6:
            points[:, 3:6] = points[:, 3:6] @ rotation.T
        transform = torch.eye(4, device=grasps.device, dtype=grasps.dtype)
        transform[:3, :3] = rotation.to(device=grasps.device, dtype=grasps.dtype)
        grasps = torch.matmul(transform, grasps)
        return points, grasps

    @staticmethod
    def translate(points, grasps, offset):
        """ Shifts points and grasps by offset [3] """
        points = points.clone()
        points[:, :3] = points[:, :3] + offset.to(device=points.device, dtype=points.dtype)
        grasps = grasps.clone()
        grasps[:, :3, 3] = grasps[:, :3, 3] + offset.to(device=grasps.device, dtype=grasps.dtype)
        return points, grasps

    @staticmethod
    def visible_point_mask(xyz, camera, radius_factor=100.0):
        """
        Hidden point removal (Katz et al. 2007, "Direct Visibility of Point Sets").
        Approximates which points a depth camera at position camera would see.

        :param xyz: [N, 3] numpy array
        :param camera: [3] numpy array, camera position in the same frame
        :return: boolean mask [N] of visible points
        """
        p = xyz - camera
        norm = np.linalg.norm(p, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        radius = norm.max() * radius_factor
        flipped = p + 2 * (radius - norm) * p / norm
        hull = ConvexHull(np.vstack([flipped, np.zeros((1, 3))]))
        visible = hull.vertices[hull.vertices < len(xyz)]
        mask = np.zeros(len(xyz), dtype=bool)
        mask[visible] = True
        return mask

    @staticmethod
    def partial_view(points, grasps, num_points, distance_range=(0.4, 0.8), min_elevation=-0.2,
                     recenter=True, generator=None):
        """
        Simulates a single view depth observation of the object: a camera is placed at a random direction
        around the object, only the points visible from it are kept and the result is resampled to num_points.

        :param points: [N, 3 + F] dense object point cloud, should hold more than num_points points
        :param grasps: [G, 4, 4] grasp poses
        :param num_points: number of points of the returned cloud
        :param distance_range: camera distance to the object centre in metre
        :param min_elevation: minimal z component of the viewing direction, -1 allows views from below
        :param recenter: move the frame to the mean of the visible points, as it would be done on a real
                         partial observation where the full object centre is unknown
        """
        rng = np.random.default_rng(None if generator is None else int(generator.integers(2 ** 31)))
        direction = rng.normal(size=3)
        direction[2] = max(direction[2], min_elevation * np.linalg.norm(direction))
        direction = direction / np.linalg.norm(direction)
        camera = direction * rng.uniform(*distance_range)

        xyz = points[:, :3].detach().cpu().numpy()
        mask = Augmentation.visible_point_mask(xyz, camera)
        idcs = np.flatnonzero(mask)
        if len(idcs) == 0:
            idcs = np.arange(len(xyz))
        idcs = rng.choice(idcs, num_points, replace=len(idcs) < num_points)
        points = points[torch.as_tensor(idcs, device=points.device)]

        if recenter:
            points, grasps = Augmentation.translate(points, grasps, -points[:, :3].mean(dim=0))
        return points, grasps

    @staticmethod
    def jitter(points, sigma=0.002, clip=0.01):
        """ Adds clipped gaussian noise to the xyz coordinates, a crude model of depth sensor noise """
        points = points.clone()
        noise = torch.clamp(torch.randn_like(points[:, :3]) * sigma, -clip, clip)
        points[:, :3] = points[:, :3] + noise
        return points
