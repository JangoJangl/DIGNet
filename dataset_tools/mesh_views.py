"""
Full and partial point cloud rows for mesh based grasp datasets (ACRONYM, GraspGen): the same code renders
both sources, so full and partial data differ only in the observation, never in grasp handling.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from dataset_tools.common import GRIPPERS, make_row, resample, select_grasps, to_frame
from helper.rendering import DepthRenderer


@dataclass
class MeshRecord:
    source: str
    object_id: str
    category: str
    mesh_file: str
    scale: float
    gripper: str
    mesh: object                     # trimesh.Trimesh, already scaled, in the frame of the grasps
    grasps: np.ndarray               # [K, 4, 4] gripper base poses, GraspGen / ACRONYM convention
    success: np.ndarray              # [K] int8
    quality: Optional[np.ndarray] = None


def raycasting_scene(mesh):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
    return scene


def finger_contacts(scene, grasps, gripper='franka_panda'):
    """
    Closing rays from sample points on both finger pads towards the other finger, the first hit within the
    stroke is the contact.

    :return: [G, 2, 3] contacts in the grasp frame's parent frame, NaN where a finger closes on nothing
    """
    import open3d as o3d
    g = GRIPPERS[gripper]
    pads = np.array([[s * g['pad_x'], y, z, 1.0] for s in (1, -1) for z in g['pad_z'] for y in g['pad_y']])
    r = len(g['pad_z']) * len(g['pad_y'])
    origins = np.einsum('gij,pj->gpi', grasps[:, :3, :], pads)                     # [G, 2R, 3]
    directions = -np.sign(pads[:, 0])[None, :, None] * grasps[:, None, :3, 0]      # towards the other finger
    rays = np.concatenate([origins, directions], axis=-1).reshape(-1, 6).astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy().reshape(len(grasps), 2, r)
    t = np.where(t <= 2 * g['pad_x'], t, np.inf)
    best = t.argmin(axis=-1)
    t_best = np.take_along_axis(t, best[..., None], -1)[..., 0]
    pick = lambda a: np.take_along_axis(a.reshape(len(grasps), 2, r, 3), best[..., None, None].repeat(3, -1), 2)[:, :, 0]
    return pick(origins) + pick(directions) * np.where(np.isfinite(t_best), t_best, np.nan)[..., None]


def subsample(record, max_grasps, rng):
    idx = select_grasps(record.success, max_grasps, rng, record.quality)
    quality = None if record.quality is None else record.quality[idx]
    return record.grasps[idx], record.success[idx], quality


def full_row(record, num_points, max_grasps, rng):
    """ points sampled uniformly on the surface, object (mesh) frame """
    import trimesh
    grasps, success, quality = subsample(record, max_grasps, rng)
    points, _ = trimesh.sample.sample_surface(record.mesh, num_points, seed=int(rng.integers(2 ** 31)))
    contacts = finger_contacts(raycasting_scene(record.mesh), grasps, record.gripper)
    valid = np.isfinite(contacts).all(axis=(1, 2))
    dist = cKDTree(points).query(np.nan_to_num(contacts.reshape(-1, 3)))[0].reshape(-1, 2)
    return make_row(source=record.source, cloud_type='full', gripper=record.gripper, object_id=record.object_id,
                    category=record.category, mesh_file=record.mesh_file, scale=record.scale, view=-1,
                    points=points, camera_pose=np.eye(4), grasps=grasps, success=success, quality=quality,
                    contacts=contacts, contact_visible=np.repeat(valid[:, None], 2, 1),
                    contact_cloud_distance=np.where(valid[:, None], dist, np.nan),
                    visibility_class=np.where(valid, 2, -1))


def partial_rows(record, views, num_points, max_grasps, rng, renderer=None, tolerance=0.003):
    """ rendered single views with sensor noise (helper/rendering.py), camera frame """
    renderer = renderer or DepthRenderer()
    grasps, success, quality = subsample(record, max_grasps, rng)
    # the renderer aims at the origin, many Objaverse meshes lie far from it: render around the bounding box centre
    shift = np.eye(4)
    shift[:3, 3] = record.mesh.bounds.mean(axis=0)
    mesh = record.mesh.copy().apply_translation(-shift[:3, 3])
    grasps_centred = np.linalg.inv(shift)[None] @ grasps
    contacts, rows = None, []
    for view in range(views):
        obs = renderer.render_view(mesh, rng)
        if len(obs['points']) < 32:
            continue
        if contacts is None:
            contacts = finger_contacts(obs['scene'], grasps_centred, record.gripper)
        valid = np.isfinite(contacts).all(axis=(1, 2))
        flat = np.nan_to_num(contacts.reshape(-1, 3))
        visible = renderer.visible(obs['scene'], obs['eye'], flat, tolerance) \
            & DepthRenderer.in_view(obs['camera_pose'], obs['tan_limits'], flat)
        visible = visible.reshape(-1, 2) & valid[:, None]
        pose = obs['camera_pose']  # camera in the centred frame, camera_pose below is in the object frame
        rows.append(make_row(
            source=record.source, cloud_type='partial', gripper=record.gripper, object_id=record.object_id,
            category=record.category, mesh_file=record.mesh_file, scale=record.scale, view=view,
            points=to_frame(pose, resample(obs['points'], num_points, rng)), camera_pose=shift @ pose,
            grasps=np.linalg.inv(pose)[None] @ grasps_centred, success=success, quality=quality,
            contacts=np.where(valid[:, None, None], to_frame(pose, flat).reshape(-1, 2, 3), np.nan),
            contact_visible=visible,
            contact_cloud_distance=np.where(valid[:, None], cKDTree(obs['points']).query(flat)[0].reshape(-1, 2), np.nan),
            visibility_class=np.where(valid, visible.sum(axis=1), -1)))
    return rows
