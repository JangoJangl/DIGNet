"""
Tiny synthetic copies of the original dataset layouts (ACRONYM, GraspGen, GraspNet-1Billion) with a box and a
cylinder and geometrically valid antipodal Panda grasps, used to test the converters end to end.
"""
import io
import json
import os
import tarfile
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import trimesh

TCP = 0.11217


def pose(x_axis, z_axis, finger_centre):
    """ base link pose with closing axis x, approach z and the finger centre at finger_centre """
    x, z = np.asarray(x_axis, float), np.asarray(z_axis, float)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    T[:3, 3] = np.asarray(finger_centre) - TCP * z
    return T


def box_grasps(extents, rng, n=60):
    """ grasps closing over the box x extent, approaching from +y, -y or +z """
    ex, ey, ez = np.asarray(extents) / 2
    grasps = []
    for _ in range(n):
        side = rng.integers(3)
        if side < 2:
            s = 1 if side == 0 else -1
            c = [0, s * (ey - 0.01), rng.uniform(-ez + 0.01, ez - 0.01)]
            grasps.append(pose([1, 0, 0], [0, -s, 0], c))
        else:
            grasps.append(pose([1, 0, 0], [0, 0, -1], [0, rng.uniform(-ey + 0.005, ey - 0.005), ez - 0.01]))
    return np.array(grasps)


def cylinder_grasps(radius, height, rng, n=60):
    """ side grasps closing over the diameter, approach orthogonal to the axis """
    grasps = []
    for _ in range(n):
        phi = rng.uniform(0, 2 * np.pi)
        approach = -np.array([np.cos(phi), np.sin(phi), 0])
        closing = np.array([-np.sin(phi), np.cos(phi), 0])
        grasps.append(pose(closing, approach, -approach * (radius - 0.01) + [0, 0, rng.uniform(-height / 3, height / 3)]))
    return np.array(grasps)


def random_failures(rng, n=40):
    from scipy.spatial.transform import Rotation
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rotation.random(n, random_state=int(rng.integers(1e6))).as_matrix()
    T[:, :3, 3] = rng.normal(scale=0.15, size=(n, 3))
    return T


def objects(rng):
    box = trimesh.creation.box(extents=(0.05, 0.03, 0.12))
    cyl = trimesh.creation.cylinder(radius=0.025, height=0.1, sections=32)
    for name, mesh, good in (('Box', box, box_grasps((0.05, 0.03, 0.12), rng)), ('Cylinder', cyl, cylinder_grasps(0.025, 0.1, rng))):
        grasps = np.concatenate([good, random_failures(rng)])
        success = np.r_[np.ones(len(good)), np.zeros(len(grasps) - len(good))].astype(np.int8)
        yield name, mesh, grasps, success


def make_acronym(root, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, 'grasps'), exist_ok=True)
    for name, mesh, grasps, success in objects(rng):
        mesh_file = f'meshes/{name}/{name.lower()}0.obj'
        os.makedirs(os.path.join(root, os.path.dirname(mesh_file)), exist_ok=True)
        mesh.export(os.path.join(root, mesh_file))
        with h5py.File(os.path.join(root, 'grasps', f'{name}_{name.lower()}0_1.0.h5'), 'w') as f:
            f['object/file'] = mesh_file
            f['object/scale'] = 1.0
            f['grasps/transforms'] = grasps
            f['grasps/qualities/flex/object_in_gripper'] = success
    return root


def make_graspgen(root, object_root, gripper='franka_panda', seed=1, scale=0.5):
    """ meshes are stored twice as large and scaled by 0.5 in the json, as GraspGen does with Objaverse assets """
    rng = np.random.default_rng(seed)
    shard_dir = os.path.join(root, 'grasp_data', gripper)
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(os.path.join(root, 'splits', gripper), exist_ok=True)
    os.makedirs(object_root, exist_ok=True)
    uuid_to_path, index, uuids = {}, {}, []
    with tarfile.open(os.path.join(shard_dir, 'shard_000.tar'), 'w') as tar:
        for i, (name, mesh, grasps, success) in enumerate(objects(rng)):
            uuid = f'{i:032x}'
            path = os.path.join(object_root, f'{uuid}.glb')
            mesh.copy().apply_scale(1 / scale).export(path)
            uuid_to_path[uuid], index[uuid] = path, 0
            uuids.append(uuid)
            data = json.dumps({'object': {'file': f'{uuid}.glb', 'scale': scale},
                               'grasps': {'transforms': grasps.tolist(), 'object_in_gripper': success.astype(bool).tolist()}}).encode()
            info = tarfile.TarInfo(f'{uuid}.grasps.json')
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    json.dump(index, open(os.path.join(shard_dir, 'uuid_index.json'), 'w'))
    json.dump(uuid_to_path, open(os.path.join(object_root, 'map_uuid_to_path.json'), 'w'))
    open(os.path.join(root, 'splits', gripper, 'train.txt'), 'w').write(uuids[0] + '\n')
    open(os.path.join(root, 'splits', gripper, 'valid.txt'), 'w').write(uuids[1] + '\n')
    return root


def make_graspnet(root, seed=2, extents=(0.05, 0.03, 0.12), width=0.07, image=(160, 120)):
    """
    One object (box, id 0) and one scene with one image. Labels: points on the +-y and +-z faces, a grasp is
    good (friction 0.2) when it approaches against the face normal and closes along the box x axis.
    """
    import open3d as o3d
    from PIL import Image
    from dataset_tools.sources import graspnet_views, viewpoint_to_matrix, NUM_ANGLES, NUM_DEPTHS, NUM_VIEWS
    rng = np.random.default_rng(seed)
    box = trimesh.creation.box(extents=extents)
    os.makedirs(os.path.join(root, 'models', '000'), exist_ok=True)
    box.export(os.path.join(root, 'models', '000', 'nontextured.ply'))

    half = np.asarray(extents) / 2
    normals = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] * 10, float)
    points = normals * half + (1 - np.abs(normals)) * rng.uniform(-1, 1, (len(normals), 3)) * (half - 0.01)
    views = graspnet_views()
    angles = np.arange(NUM_ANGLES) * np.pi / NUM_ANGLES
    R = viewpoint_to_matrix(-np.repeat(views, NUM_ANGLES, 0), np.tile(angles, NUM_VIEWS)).reshape(NUM_VIEWS, NUM_ANGLES, 3, 3)
    closes_x = np.abs(R[..., 0, 1]) > 0.95                                       # closing axis (y) along box x
    good = (views @ normals.T).T[:, :, None] > 0.95                              # [P, V, 1] approach against the normal
    fric = np.where(good & closes_x[None], 0.2, 0.9)[..., None].repeat(NUM_DEPTHS, -1).astype(np.float32)
    offsets = np.zeros(fric.shape + (3,), np.float32)
    offsets[..., 0] = angles[None, None, :, None]
    offsets[..., 1] = np.array([0.01, 0.02, 0.03, 0.04])
    offsets[..., 2] = width
    os.makedirs(os.path.join(root, 'grasp_label'), exist_ok=True)
    np.savez(os.path.join(root, 'grasp_label', '000_labels.npz'), points=points.astype(np.float32), offsets=offsets,
             scores=fric, collision=np.zeros(fric.shape, bool))

    # scene: box 0.5 m in front of the camera, tilted
    from scipy.spatial.transform import Rotation
    obj_pose = np.eye(4)
    obj_pose[:3, :3] = Rotation.from_euler('xyz', [0.4, 0.3, 0.2]).as_matrix()
    obj_pose[:3, 3] = [0.0, 0.0, 0.5]
    scene_dir = os.path.join(root, 'scenes', 'scene_0000', 'kinect')
    for sub in ('depth', 'label', 'annotations'):
        os.makedirs(os.path.join(scene_dir, sub), exist_ok=True)
    w, h = image
    K = np.array([[200.0, 0, w / 2], [0, 200.0, h / 2], [0, 0, 1]])
    np.save(os.path.join(scene_dir, 'camK.npy'), K)
    posed = box.copy().apply_transform(obj_pose)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(posed.vertices, np.float32)), o3d.core.Tensor(np.asarray(posed.faces, np.uint32)))
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    dirs = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    rays = np.concatenate([np.zeros_like(dirs), dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)], -1).astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    depth = np.where(np.isfinite(t), t * rays[..., 5], 0)
    Image.fromarray((depth * 1000).astype(np.uint16)).save(os.path.join(scene_dir, 'depth', '0000.png'))
    Image.fromarray(np.isfinite(t).astype(np.uint8)).save(os.path.join(scene_dir, 'label', '0000.png'))  # object 0 -> 1

    top = ET.Element('scene')
    obj = ET.SubElement(top, 'obj')
    q = Rotation.from_matrix(obj_pose[:3, :3]).as_quat()  # x y z w
    for tag, text in (('obj_id', '0'), ('obj_name', 'box'), ('obj_path', 'models/000'),
                      ('pos_in_world', ' '.join(map(str, obj_pose[:3, 3]))), ('ori_in_world', ' '.join(map(str, q[[3, 0, 1, 2]])))):
        ET.SubElement(obj, tag).text = text
    ET.ElementTree(top).write(os.path.join(scene_dir, 'annotations', '0000.xml'))
    os.makedirs(os.path.join(root, 'collision_label', 'scene_0000'), exist_ok=True)
    np.savez(os.path.join(root, 'collision_label', 'scene_0000', 'collision_labels.npz'), np.zeros(fric.shape, bool))
    return root, obj_pose
