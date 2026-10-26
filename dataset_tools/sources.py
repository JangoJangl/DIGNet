"""
Readers for the original dataset layouts. Each yields MeshRecords (mesh based sources) or finished rows
(GraspNet-1Billion scenes, which are real depth images and therefore always partial).

    ACRONYM            meshes + simulated grasps        -> full and partial (rendered)
    GraspGen (NVIDIA)  Objaverse meshes + sim. grasps    -> full and partial (rendered)
    GraspNet-1Billion  real RGB-D scenes (+ 88 models)   -> partial (real scenes); full from the object models
"""
import glob
import json
import os
import tarfile
import xml.etree.ElementTree as ET

import numpy as np

from dataset_tools.common import GRIPPERS, make_row, resample, select_grasps, split_of
from dataset_tools.mesh_views import MeshRecord


def load_mesh(path, scale):
    import trimesh
    # textures are not needed and merging them (multi part Objaverse glbs) can take > 5 GB of RAM
    mesh = trimesh.load(path, force='mesh', skip_materials=True)
    mesh.apply_scale(scale)
    return mesh


# ---------------------------------------------------------------------------------------------------- ACRONYM

def acronym_files(root, max_objects=None):
    return sorted(glob.glob(os.path.join(root, 'grasps', '*.h5')))[:max_objects]


def acronym_mesh_path(root, mesh_file):
    """ <root>/meshes/<Category>/<id>.obj (ACRONYM layout) or the unpacked ShapeNetSem archive <root>/models/<id>.obj """
    candidates = [os.path.join(root, mesh_file)] + [os.path.join(root, d, os.path.basename(mesh_file))
                                                    for d in ('models', os.path.join('models-OBJ', 'models'))]
    return next((c for c in candidates if os.path.exists(c)), candidates[0])


def acronym_record(root, h5_path):
    """ ACRONYM layout: <root>/grasps/<Category>_<id>_<scale>.h5, mesh path inside the file relative to <root> """
    import h5py
    with h5py.File(h5_path, 'r') as f:
        mesh_file = f['object/file'][()].decode('utf-8')
        scale = float(f['object/scale'][()])
        grasps = np.array(f['grasps/transforms'])
        success = np.array(f['grasps/qualities/flex/object_in_gripper']).astype(np.int8)
    object_id = os.path.splitext(os.path.basename(h5_path))[0]
    return MeshRecord(source='acronym', object_id=object_id, category=object_id.split('_')[0], mesh_file=mesh_file,
                      scale=scale, gripper='franka_panda', mesh=load_mesh(acronym_mesh_path(root, mesh_file), scale),
                      grasps=grasps, success=success)


# --------------------------------------------------------------------------------------------------- GraspGen

def graspgen_shards(root, gripper):
    """ HF repo nvidia/PhysicalAI-Robotics-GraspGen: grasp_data/<gripper>/shard_XXX.tar (WebDataset) """
    return sorted(glob.glob(os.path.join(root, 'grasp_data', gripper, 'shard_*.tar')))


def graspgen_splits(root, gripper):
    """ official object splits: train.txt -> train (5 % of it -> validation by hash), valid.txt -> test """
    split = {}
    for name, target in (('train', 'train'), ('valid', 'test')):
        path = os.path.join(root, 'splits', gripper, f'{name}.txt')
        if os.path.exists(path):
            for uuid in open(path).read().split():
                split[uuid] = target
    for uuid, s in split.items():
        if s == 'train' and split_of(uuid, {'train': 0.95, 'validation': 0.05}) == 'validation':
            split[uuid] = 'validation'
    return split


def graspgen_records(shard, object_root, gripper, max_objects=None):
    """
    One WebDataset sample per object: member '<uuid>.grasps.json' with
    {'object': {'file', 'scale'}, 'grasps': {'transforms' [K, 4, 4], 'object_in_gripper' [K]}}.
    Meshes come from Objaverse (download_objaverse.py of the dataset repo writes map_uuid_to_path.json).
    """
    map_path = os.path.join(object_root, 'map_uuid_to_path.json')
    uuid_to_path = json.load(open(map_path)) if os.path.exists(map_path) else {}
    count = 0
    with tarfile.open(shard) as tar:
        for member in tar:
            if not member.name.endswith('.json'):
                continue
            uuid = os.path.basename(member.name).split('.')[0]
            data = json.load(tar.extractfile(member))
            mesh_path = uuid_to_path.get(uuid) or os.path.join(object_root, data['object']['file'])
            if not os.path.exists(mesh_path):
                print(f"skip {uuid}: mesh {mesh_path} not found")
                continue
            scale = float(data['object'].get('scale', 1.0))
            yield MeshRecord(source='graspgen', object_id=uuid, category='', mesh_file=os.path.relpath(mesh_path, object_root),
                             scale=scale, gripper=gripper, mesh=load_mesh(mesh_path, scale),
                             grasps=np.asarray(data['grasps']['transforms'], dtype=np.float64),
                             success=np.asarray(data['grasps']['object_in_gripper']).reshape(-1).astype(np.int8))
            count += 1
            if max_objects and count >= max_objects:
                return


# ------------------------------------------------------------------------------------------ GraspNet-1Billion

# GraspNet gripper frame: x = approach, y = closing direction, origin at the target point on the surface, the
# finger tips are `depth` further along x. Our convention: z = approach, x = closing, origin at the base link.
GRASPNET_TO_OURS = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])  # columns: our x, y, z in GraspNet axes
NUM_VIEWS, NUM_ANGLES, NUM_DEPTHS = 300, 12, 4


def graspnet_views(n=NUM_VIEWS, phi=(np.sqrt(5) - 1) / 2):
    """ Fibonacci lattice approach views, identical to graspnetAPI.utils.utils.generate_views """
    i = np.arange(n, dtype=np.float32)
    z = (2 * i + 1) / n - 1
    return np.stack([np.sqrt(1 - z ** 2) * np.cos(2 * i * np.pi * phi), np.sqrt(1 - z ** 2) * np.sin(2 * i * np.pi * phi), z], 1)


def viewpoint_to_matrix(towards, angle):
    """ graspnetAPI.utils.rotation.batch_viewpoint_params_to_matrix """
    x = towards / np.linalg.norm(towards, axis=-1, keepdims=True)
    y = np.stack([-towards[:, 1], towards[:, 0], np.zeros(len(towards))], -1)
    y[np.linalg.norm(y, axis=-1) == 0] = [0, 1, 0]
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    c, s = np.cos(angle), np.sin(angle)
    r1 = np.zeros((len(angle), 3, 3))
    r1[:, 0, 0], r1[:, 1, 1], r1[:, 1, 2], r1[:, 2, 1], r1[:, 2, 2] = 1, c, -s, s, c
    return np.stack([x, y, z], -1) @ r1


def graspnet_to_ours(rotation, target, depth, gripper='franka_panda'):
    """ GraspNet (R, target point, depth) -> 4x4 base link poses in the GraspGen / ACRONYM convention """
    approach = rotation[:, :, 0]
    tips = target + depth[:, None] * approach                     # centre between the finger tips
    poses = np.tile(np.eye(4), (len(rotation), 1, 1))
    poses[:, :3, :3] = rotation @ GRASPNET_TO_OURS
    poses[:, :3, 3] = tips - GRIPPERS[gripper]['tcp_depth'] * approach
    return poses


def graspnet_object_grasps(label_path, rng, max_grasps, fric_thresh=0.4, collision=None, max_width=None):
    """
    Decodes an object label file (<root>/grasp_label/XXX_labels.npz) into a subset of grasps in the object frame.

    Positive: 0 < friction coefficient <= fric_thresh and not in collision (scene labels), negative: the
    other defined grasps. quality = 1.1 - friction coefficient as in graspnetAPI.
    """
    label = np.load(label_path)
    points, offsets, fric = label['points'], label['offsets'], label['scores']
    defined = fric > 0
    if max_width is not None:
        defined &= offsets[..., 2] <= max_width
    success = defined & (fric <= fric_thresh)
    if collision is not None:
        success &= ~collision
    flat_defined = np.flatnonzero(defined)
    idx = flat_defined[select_grasps(success.reshape(-1)[flat_defined].astype(np.int8), max_grasps, rng)]
    p, v, a, d = np.unravel_index(idx, fric.shape)
    angle, depth, width = (offsets[p, v, a, d, i] for i in range(3))
    rotation = viewpoint_to_matrix(-graspnet_views()[v], angle)
    return dict(target=points[p], rotation=rotation, depth=depth, width=width,
                success=success.reshape(-1)[idx].astype(np.int8), quality=1.1 - fric.reshape(-1)[idx])


def graspnet_full_rows(root, object_ids, num_points, max_grasps, gripper='franka_panda', seed=0):
    """ full clouds from the 88 GraspNet object models with their object level grasp labels (no scene collisions) """
    import trimesh
    for obj in object_ids:
        rng = np.random.default_rng(seed + obj)
        mesh_file = os.path.join('models', f'{obj:03d}', 'nontextured.ply')
        model = trimesh.load(os.path.join(root, mesh_file))
        g = graspnet_object_grasps(os.path.join(root, 'grasp_label', f'{obj:03d}_labels.npz'), rng, max_grasps,
                                   max_width=GRIPPERS[gripper]['max_width'])
        if hasattr(model, 'faces') and len(model.faces):
            points, _ = trimesh.sample.sample_surface(model, num_points, seed=int(rng.integers(2 ** 31)))
        else:
            points = resample(np.asarray(model.vertices), num_points, rng)
        poses = graspnet_to_ours(g['rotation'], g['target'], g['depth'], gripper)
        yield make_row(source='graspnet1b', cloud_type='full', gripper=gripper, object_id=f'{obj:03d}', category='',
                       mesh_file=mesh_file, scale=1.0, view=-1, points=points, camera_pose=np.eye(4), grasps=poses,
                       success=g['success'], quality=g['quality'], width=g['width'],
                       **_graspnet_contacts(g, poses, model, points, gripper, full=True))


def _graspnet_contacts(g, poses, model, cloud, gripper, full, tolerance=0.005):
    """
    Contacts by closing the fingers on the object model (posed like the cloud). Without a mesh model both fingers
    are assumed at target -/+ width / 2 along the closing axis. Visibility: full clouds see every contact,
    real partial clouds count a contact as visible if an observed point lies within `tolerance`.
    """
    from scipy.spatial import cKDTree
    from dataset_tools.mesh_views import finger_contacts, raycasting_scene
    if hasattr(model, 'faces') and len(model.faces):
        contacts = finger_contacts(raycasting_scene(model), poses, gripper)
    else:
        closing = g['rotation'][:, :, 1]
        contacts = g['target'][:, None] + np.array([-1, 1])[None, :, None] * (g['width'][:, None, None] / 2) * closing[:, None]
    valid = np.isfinite(contacts).all(axis=(1, 2))
    dist = cKDTree(cloud).query(np.nan_to_num(contacts.reshape(-1, 3)))[0].reshape(-1, 2)
    visible = (np.ones_like(dist, bool) if full else dist < tolerance) & valid[:, None]
    return dict(contacts=contacts, contact_visible=visible, contact_cloud_distance=np.where(valid[:, None], dist, np.nan),
                visibility_class=np.where(valid, visible.sum(1), -1))


def graspnet_scene_split(scene_id):
    """ official: scenes 0-99 train (95-99 used as validation here), 100-189 test (seen / similar / novel) """
    return 'test' if scene_id >= 100 else ('validation' if scene_id >= 95 else 'train')


def _read_pose_annotations(xml_path):
    """ object id -> 4x4 object pose in the camera frame of this image (annotations/XXXX.xml) """
    from scipy.spatial.transform import Rotation
    poses = {}
    for obj in ET.parse(xml_path).getroot():
        fields = list(obj)
        obj_id = int(fields[0].text)
        t = np.array(fields[3].text.split(), dtype=np.float64)
        q = np.array(fields[4].text.split(), dtype=np.float64)              # w x y z
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        pose[:3, 3] = t
        poses[obj_id] = pose
    return poses


def graspnet_partial_rows(root, scene_ids, camera='kinect', ann_ids=range(0, 256, 16), num_points=1024, max_grasps=500,
                          gripper='franka_panda', min_points=64, seed=0):
    """
    One row per (scene, image, object): the object's points cut from the real depth image with the
    segmentation label (label value = object id + 1), grasps of that object in the camera frame with the
    scene collision labels applied. Everything in the camera frame, camera_pose = camera -> object frame.
    """
    import trimesh
    from PIL import Image
    models = {}
    for scene_id in scene_ids:
        scene_dir = os.path.join(root, 'scenes', f'scene_{scene_id:04d}', camera)
        intrinsics = np.load(os.path.join(scene_dir, 'camK.npy'))
        collision = np.load(os.path.join(root, 'collision_label', f'scene_{scene_id:04d}', 'collision_labels.npz'))
        collision = [collision[f'arr_{i}'] for i in range(len(collision.files))]
        for ann in ann_ids:
            if not os.path.exists(os.path.join(scene_dir, 'depth', f'{ann:04d}.png')):
                continue
            rng = np.random.default_rng(seed + 1000 * scene_id + ann)
            depth = np.array(Image.open(os.path.join(scene_dir, 'depth', f'{ann:04d}.png'))).astype(np.float64) / 1000.0
            label = np.array(Image.open(os.path.join(scene_dir, 'label', f'{ann:04d}.png')))
            v, u = np.nonzero(depth > 0)
            z = depth[v, u]
            cloud = np.stack([(u - intrinsics[0, 2]) / intrinsics[0, 0] * z, (v - intrinsics[1, 2]) / intrinsics[1, 1] * z, z], 1)
            seg = label[v, u]
            poses = _read_pose_annotations(os.path.join(scene_dir, 'annotations', f'{ann:04d}.xml'))
            for i, (obj_id, pose) in enumerate(poses.items()):        # collision labels follow the xml order
                pts = cloud[seg == obj_id + 1]
                if len(pts) < min_points:
                    continue
                g = graspnet_object_grasps(os.path.join(root, 'grasp_label', f'{obj_id:03d}_labels.npz'), rng, max_grasps,
                                           collision=collision[i], max_width=GRIPPERS[gripper]['max_width'])
                g['target'] = g['target'] @ pose[:3, :3].T + pose[:3, 3]
                g['rotation'] = pose[None, :3, :3] @ g['rotation']
                if obj_id not in models:
                    models[obj_id] = trimesh.load(os.path.join(root, 'models', f'{obj_id:03d}', 'nontextured.ply'))
                model = models[obj_id].copy().apply_transform(pose)
                poses = graspnet_to_ours(g['rotation'], g['target'], g['depth'], gripper)
                yield make_row(source='graspnet1b', cloud_type='partial', gripper=gripper,
                               object_id=f'{obj_id:03d}', category='', mesh_file=os.path.join('models', f'{obj_id:03d}', 'nontextured.ply'),
                               scale=1.0, view=scene_id * 1000 + ann, points=resample(pts, num_points, rng),
                               camera_pose=np.linalg.inv(pose), grasps=poses,
                               success=g['success'], quality=g['quality'], width=g['width'], split=graspnet_scene_split(scene_id),
                               **_graspnet_contacts(g, poses, model, pts, gripper, full=False))
