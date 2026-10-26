"""
Unified grasp dataset format shared by all converters (ACRONYM, GraspGen, GraspNet-1Billion).

One row = one observation of one object:
  cloud_type 'full'    -> points sampled on the whole object surface, object frame (mesh frame of the source)
  cloud_type 'partial' -> points seen by one (simulated or real) depth camera, camera frame (OpenCV: x right,
                          y down, z forward), camera_pose maps camera -> object frame

Grasps are 4x4 poses in the same frame as the points, in the GraspGen / ACRONYM gripper convention:
origin at the gripper base link, approach along +z, fingers close along x. `tcp_depth` is the distance from the
base link to the point between the finger tips, so `grasp @ translation(0, 0, tcp_depth)` is the finger centre
that DIGNet's processor works with.
"""
import hashlib

import numpy as np

# Gripper geometry in the GraspGen / ACRONYM convention.
# pad_x: inner pad surface at full opening, pad_z / pad_y: sample points on the pad used to find contacts
GRIPPERS = {
    'franka_panda': dict(tcp_depth=0.11217, max_width=0.08, pad_x=0.0399, pad_z=(0.1034, 0.1078, 0.112204),
                         pad_y=(-0.008, 0.0, 0.008)),
    'robotiq_2f_140': dict(tcp_depth=0.195, max_width=0.136, pad_x=0.068, pad_z=(0.18, 0.1875, 0.195),
                           pad_y=(-0.01, 0.0, 0.01)),
}

SPLITS = {'train': 0.90, 'validation': 0.05, 'test': 0.05}


def features():
    from datasets import Array2D, Array3D, Features, Sequence, Value
    return Features({
        'sample_id': Value('string'),
        'source': Value('string'),          # acronym | graspgen | graspnet1b
        'cloud_type': Value('string'),      # full | partial
        'gripper': Value('string'),
        'object_id': Value('string'),
        'category': Value('string'),
        'mesh_file': Value('string'),
        'scale': Value('float32'),
        'view': Value('int32'),             # -1 for full clouds
        'split': Value('string'),
        'tcp_depth': Value('float32'),
        'points': Array2D(shape=(None, 3), dtype='float32'),
        'camera_pose': Array2D(shape=(4, 4), dtype='float32'),
        'grasps': Array3D(shape=(None, 4, 4), dtype='float32'),
        'success': Sequence(Value('int8')),
        'quality': Sequence(Value('float32')),     # source specific score, NaN if the source has none
        'width': Sequence(Value('float32')),       # opening width, NaN if unknown
        'contacts': Array3D(shape=(None, 2, 3), dtype='float32'),        # NaN if the fingers hit nothing
        'contact_visible': Array2D(shape=(None, 2), dtype='int8'),
        'contact_cloud_distance': Array2D(shape=(None, 2), dtype='float32'),
        'visibility_class': Sequence(Value('int8')),  # 2 / 1 / 0 contacts visible, -1 no contact found
    })


def split_of(object_id, splits=SPLITS):
    """ deterministic split per object, so all views of an object land in the same split """
    u = int(hashlib.md5(object_id.encode()).hexdigest(), 16) / 16 ** 32
    edges = np.cumsum(list(splits.values()))
    return list(splits)[int(np.searchsorted(edges, u, side='right').clip(max=len(splits) - 1))]


def seed_of(*parts):
    return int(hashlib.md5('/'.join(map(str, parts)).encode()).hexdigest(), 16) % 2 ** 32


def select_grasps(success, max_grasps, rng, quality=None):
    """ subset of at most max_grasps indices, half successful where possible, best quality first if given """
    pos, neg = np.flatnonzero(success == 1), np.flatnonzero(success == 0)
    n_pos = min(len(pos), max(max_grasps // 2, max_grasps - len(neg)))
    n_neg = min(len(neg), max_grasps - n_pos)
    if quality is not None:
        pos = pos[np.argsort(-quality[pos], kind='stable')][:n_pos]
    else:
        pos = rng.choice(pos, n_pos, replace=False)
    return np.sort(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])).astype(int)


def resample(points, num_points, rng):
    return points[rng.choice(len(points), num_points, replace=len(points) < num_points)]


def to_frame(transform, points):
    """ apply the inverse of a 4x4 transform (frame -> world) to points [N, 3], i.e. express them in that frame """
    return (points - transform[:3, 3]) @ transform[:3, :3]


def make_row(*, source, cloud_type, gripper, object_id, category, mesh_file, scale, view, points, camera_pose,
             grasps, success, quality=None, width=None, contacts=None, contact_visible=None,
             contact_cloud_distance=None, visibility_class=None, split=None):
    k = len(grasps)
    nan = lambda *shape: np.full(shape, np.nan, dtype=np.float32)
    return {
        'sample_id': f"{source}/{cloud_type}/{gripper}/{object_id}/{view}",
        'source': source, 'cloud_type': cloud_type, 'gripper': gripper,
        'object_id': object_id, 'category': category, 'mesh_file': mesh_file, 'scale': float(scale),
        'view': int(view), 'split': split or split_of(object_id),
        'tcp_depth': GRIPPERS[gripper]['tcp_depth'] if gripper in GRIPPERS else np.nan,
        'points': np.asarray(points, dtype=np.float32),
        'camera_pose': np.asarray(camera_pose, dtype=np.float32),
        'grasps': np.asarray(grasps, dtype=np.float32).reshape(k, 4, 4),
        'success': np.asarray(success, dtype=np.int8),
        'quality': nan(k) if quality is None else np.asarray(quality, dtype=np.float32),
        'width': nan(k) if width is None else np.asarray(width, dtype=np.float32),
        'contacts': nan(k, 2, 3) if contacts is None else np.asarray(contacts, dtype=np.float32),
        'contact_visible': np.zeros((k, 2), np.int8) if contact_visible is None else np.asarray(contact_visible, np.int8),
        'contact_cloud_distance': nan(k, 2) if contact_cloud_distance is None else np.asarray(contact_cloud_distance, np.float32),
        'visibility_class': np.full(k, -1, np.int8) if visibility_class is None else np.asarray(visibility_class, np.int8),
    }
