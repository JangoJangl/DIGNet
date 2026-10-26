import numpy as np
import torch


class DepthRenderer(object):
    """
    Renders single-view depth observations of a mesh by ray casting (Open3D RaycastingScene, Embree) and
    turns them into a noisy partial point cloud, approximating what a depth camera sees.

    Noise model (defaults roughly in the range of active stereo sensors such as a RealSense D4xx at < 1 m):
      - axial noise along the viewing ray, std = sigma_base + sigma_quad * depth^2
      - random pixel dropout
      - dropout of pixels at depth discontinuities (edges, where stereo matching fails most often)
    """

    def __init__(self, width=160, height=120, fov_deg=58.0, distance_range=(0.35, 0.7), min_elevation=-0.2,
                 sigma_base=0.001, sigma_quad=0.002, dropout=0.02, edge_threshold=0.01, edge_dropout=0.5):
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.distance_range = distance_range
        self.min_elevation = min_elevation
        self.sigma_base = sigma_base
        self.sigma_quad = sigma_quad
        self.dropout = dropout
        self.edge_threshold = edge_threshold
        self.edge_dropout = edge_dropout

    def random_camera(self, rng):
        direction = rng.normal(size=3)
        direction[2] = max(direction[2], self.min_elevation * np.linalg.norm(direction))
        direction /= np.linalg.norm(direction)
        eye = direction * rng.uniform(*self.distance_range)
        center = rng.normal(scale=0.01, size=3)  # do not always look exactly at the object centre
        up = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
        return eye, center, up

    def render(self, mesh, rng=None):
        """
        :param mesh: trimesh.Trimesh in the object frame
        :return: [M, 3] numpy array of observed points in the object frame (M varies)
        """
        return self.render_view(mesh, rng)['points']

    @staticmethod
    def camera_pose(eye, center, up):
        """ 4x4 camera-to-object transform, OpenCV convention (x right, y down, z forward) """
        forward = (center - eye) / np.linalg.norm(center - eye)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([right, down, forward], axis=1)
        pose[:3, 3] = eye
        return pose

    def render_view(self, mesh, rng=None):
        """
        Like render, but also returns the camera and the ray casting scene (for visibility queries)

        :return: dict with points [M, 3], camera_pose [4, 4] (camera -> object), eye, scene
        """
        import open3d as o3d
        rng = rng or np.random.default_rng()
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                            o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
        eye, center, up = self.random_camera(rng)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=self.fov_deg, center=center.tolist(), eye=eye.tolist(), up=up.tolist(),
            width_px=self.width, height_px=self.height)
        t_hit = scene.cast_rays(rays)['t_hit'].numpy()  # [H, W], inf where the ray misses
        rays = rays.numpy()
        origins, directions = rays[..., :3], rays[..., 3:]

        forward = (center - eye) / np.linalg.norm(center - eye)
        depth = t_hit * (directions @ forward)
        valid = np.isfinite(depth)

        # edge dropout: large depth jumps to a 4-neighbour (or to the background)
        padded = np.pad(np.where(valid, depth, np.inf), 1, constant_values=np.inf)
        jump = np.zeros_like(depth)
        for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
            neighbour = padded[dy:dy + self.height, dx:dx + self.width]
            with np.errstate(invalid='ignore'):  # inf - inf where both pixels miss the object
                jump = np.maximum(jump, np.nan_to_num(np.abs(neighbour - depth), nan=np.inf, posinf=np.inf))
        edge = valid & (jump > self.edge_threshold)
        valid &= ~(edge & (rng.random(depth.shape) < self.edge_dropout))
        valid &= rng.random(depth.shape) >= self.dropout

        t = t_hit[valid]
        sigma = self.sigma_base + self.sigma_quad * depth[valid] ** 2
        t = t + rng.normal(size=t.shape) * sigma / np.maximum(directions[valid] @ forward, 1e-3)
        points = origins[valid] + directions[valid] * t[:, None]
        pose = self.camera_pose(eye, center, up)
        corners = directions[[0, 0, -1, -1], [0, -1, 0, -1]] @ pose[:3, :3]  # corner rays in the camera frame
        tan_limits = np.abs(corners[:, :2] / corners[:, 2:]).max(axis=0)  # half field of view (x, y) as tangents
        return {'points': points, 'camera_pose': pose, 'eye': eye, 'scene': scene, 'tan_limits': tan_limits}

    @staticmethod
    def in_view(camera_pose, tan_limits, points):
        """ True for points [K, 3] (object frame) inside the camera frustum """
        p = (points - camera_pose[:3, 3]) @ camera_pose[:3, :3]
        return (p[:, 2] > 0) & (np.abs(p[:, 0]) <= tan_limits[0] * p[:, 2]) & (np.abs(p[:, 1]) <= tan_limits[1] * p[:, 2])

    @staticmethod
    def visible(scene, eye, points, tolerance=0.003):
        """
        Line of sight test: True where nothing of the mesh lies between the camera and the point.
        Points on the surface count as visible if the first hit is within tolerance of the point.

        :param points: [K, 3] query points in the object frame
        """
        import open3d as o3d
        directions = points - eye
        distance = np.linalg.norm(directions, axis=1)
        rays = np.concatenate([np.broadcast_to(eye, points.shape), directions / distance[:, None]], axis=1)
        t_hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()
        return t_hit >= distance - tolerance

    def partial_cloud(self, mesh, num_points, device='cpu', dtype=torch.float64, rng=None, min_points=32):
        """ Rendered partial cloud resampled to num_points, or None if the object was (nearly) not visible """
        rng = rng or np.random.default_rng()
        points = self.render(mesh, rng)
        if len(points) < min_points:
            return None
        idcs = rng.choice(len(points), num_points, replace=len(points) < num_points)
        return torch.as_tensor(points[idcs], device=device, dtype=dtype)
