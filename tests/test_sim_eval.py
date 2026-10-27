"""
Tests of the simulation grasp validation that run without Isaac Lab (geometry, grasp transfer, episode files,
object assets, result summaries). run_newton.py itself needs Isaac Lab and a GPU.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

trimesh = pytest.importorskip('trimesh')

from sim_eval import grippers as G  # noqa: E402
from sim_eval.episodes import EpisodeBuilder, cloud_to_mesh_frame, load_episodes  # noqa: E402
from sim_eval.report import summarize, wandb_metrics  # noqa: E402
from sim_eval.transforms import (invert, matrix_from_pose7, matrix_to_quat_xyzw, pose7_from_matrix,  # noqa: E402
                                 quat_xyzw_to_matrix, rpy, translation)

SO101_DIR = os.environ.get('SO101_DIR', '')


def random_pose(rng):
    q = rng.normal(size=4)
    t = np.eye(4)
    t[:3, :3] = quat_xyzw_to_matrix(q)
    t[:3, 3] = rng.normal(size=3) * 0.1
    return t


def test_transforms_roundtrip():
    rng = np.random.default_rng(0)
    poses = np.stack([random_pose(rng) for _ in range(20)])
    assert np.allclose(matrix_from_pose7(pose7_from_matrix(poses)), poses, atol=1e-9)
    assert np.allclose(invert(poses) @ poses, np.eye(4), atol=1e-9)
    # (x, y, z, w): 90 degrees about z
    q = matrix_to_quat_xyzw(rpy(0, 0, np.pi / 2)[:3, :3])
    assert np.allclose(q, [0, 0, np.sqrt(0.5), np.sqrt(0.5)])


def test_franka_urdf_matches_grasp_convention(tmp_path):
    path = G.franka_urdf(str(tmp_path))
    root = ET.parse(path).getroot()
    joints = {j.get('name'): j for j in root.findall('joint')}
    assert set(joints) == {'panda_finger_joint1', 'panda_finger_joint2'}
    # left finger mesh rotated by pi about z and moved by q along +x: inner pad surface at x = q, tips at tcp_depth
    finger = trimesh.load(os.path.join(G.REPO, 'gripper', 'models', 'panda_gripper', 'finger.stl'))
    left = finger.copy()
    left.apply_transform(translation(0.04, 0, 0.0584) @ rpy(0, 0, np.pi))
    assert abs(left.bounds[0, 0] - 0.04) < 1e-3
    assert abs(left.bounds[1, 2] - G.PANDA_TCP_DEPTH) < 1e-3
    right = finger.copy()
    right.apply_transform(translation(-0.04, 0, 0.0584))
    assert abs(right.bounds[1, 0] + 0.04) < 1e-3


def box_and_grasp(width, rng=None):
    """ box of the given width along the grasp's closing axis, centred between the Panda pads """
    box = trimesh.creation.box(extents=(width, 0.03, 0.06))
    grasp = translation(z=-G.PANDA_CONTACT_DEPTH)
    if rng is not None:  # same configuration in an arbitrary object frame
        pose = random_pose(rng)
        box.apply_transform(pose)
        grasp = pose @ grasp
    return box, grasp


def test_closing_line_extent():
    rng = np.random.default_rng(1)
    box, grasp = box_and_grasp(0.03, rng)
    a, b = G.closing_line_extent(box, grasp)
    assert np.isclose(a, -0.015, atol=1e-4) and np.isclose(b, 0.015, atol=1e-4)
    a, b = G.closing_line_extent(box, grasp @ translation(z=-0.2))  # fingers behind the object
    assert np.isnan(a) and np.isnan(b)


def test_so101_transfer_places_fixed_jaw_on_the_object():
    rng = np.random.default_rng(2)
    box, grasp = box_and_grasp(0.03, rng)
    poses, q_open, fits = G.root_poses('so101', grasp[None], box)
    frame = poses[0] @ G.SO101_LINK_TO_FRAME
    # same approach and closing axes as the Panda grasp
    assert np.allclose(frame[:3, :3], grasp[:3, :3], atol=1e-9)
    # fixed jaw pad centre: on the +x face of the box plus the clearance, at the Panda contact depth
    pad = frame @ np.array([G.SO101_FIXED_JAW_X, 0, G.SO101_CONTACT_Z, 1])
    local = invert(grasp) @ pad
    assert np.allclose(local[:3], [0.015 + G.FIXED_JAW_CLEARANCE, 0, G.PANDA_CONTACT_DEPTH], atol=1e-6)
    # the moving jaw opens just beyond the -x face: 3 cm + clearances
    jaw_x = np.interp(q_open[0, 0], G.SO101_JAW_TABLE_Q, G.SO101_JAW_TABLE_X)
    assert np.isclose(G.SO101_FIXED_JAW_X - jaw_x, 0.03 + G.FIXED_JAW_CLEARANCE + G.MOVING_JAW_CLEARANCE, atol=1e-3)
    assert fits[0]
    wide, wide_grasp = box_and_grasp(0.079)
    assert not G.root_poses('so101', wide_grasp[None], wide)[2][0]


def test_franka_transfer_is_identity():
    box, grasp = box_and_grasp(0.05)
    poses, q_open, fits = G.root_poses('franka', grasp[None], box)
    assert np.allclose(poses[0], grasp) and np.allclose(q_open, 0.04) and fits[0]


def test_cloud_to_mesh_frame_inverts_the_processor():
    """ processor: centre the cloud, move to the finger centre; the export has to undo both and the camera pose """
    rng = np.random.default_rng(3)
    camera = random_pose(rng)
    grasps_mesh = np.stack([random_pose(rng) for _ in range(5)])
    points_mesh = rng.normal(size=(100, 3)) * 0.05
    points_cam = (points_mesh - camera[:3, 3]) @ camera[:3, :3]
    grasps_cam = invert(camera) @ grasps_mesh
    centre = points_cam.mean(axis=0)
    model_frame = grasps_cam.copy()
    model_frame[:, :3, 3] -= centre
    model_frame = model_frame @ translation(z=0.11217)
    assert np.allclose(cloud_to_mesh_frame(model_frame, camera, centre=centre, tcp_depth=0.11217), grasps_mesh)
    assert np.allclose(cloud_to_mesh_frame(grasps_cam, camera), grasps_mesh)


def test_episode_builder(tmp_path):
    rng = np.random.default_rng(4)
    builder = EpisodeBuilder(grasps_per_object=8)
    rows = [{'mesh_file': 'a.glb', 'scale': 0.1, 'object_id': 'a', 'source': 'graspgen', 'sample_id': f'a/{v}',
             'cloud_type': 'partial'} for v in range(3)]
    rows.append({'mesh_file': 'b.glb', 'scale': 1.0, 'object_id': 'b', 'source': 'graspgen', 'sample_id': 'b/0',
                 'cloud_type': 'full'})
    for row in rows:
        n = 5
        builder.add(row, np.tile(np.eye(4), (n, 1, 1)), np.ones(n))
    path = builder.save(str(tmp_path / 'ep.npz'), {'kind': 'test'}, rng)
    ep = load_episodes(path)
    assert len(ep['objects']) == 2 and ep['meta']['grasps_per_object'] == 8
    assert np.bincount(ep['object_index']).tolist() == [8, 8]
    # object a: grasps from all three views, contiguous blocks per object
    assert set(ep['obs_index'][:8]) == {0, 1, 2}
    assert (np.diff(ep['object_index']) >= 0).all()


def test_convex_usd(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdPhysics
    from sim_eval.objects import object_mass, write_convex_usd
    parts = [(trimesh.creation.box(extents=(0.02, 0.02, 0.02)).vertices + [dx, 0, 0],
              trimesh.creation.box(extents=(0.02, 0.02, 0.02)).faces) for dx in (-0.02, 0.02)]
    mesh = trimesh.creation.box(extents=(0.06, 0.02, 0.02))
    path = write_convex_usd(parts, str(tmp_path / 'obj.usda'), object_mass(mesh), visual_mesh=mesh)
    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    assert root.HasAPI(UsdPhysics.RigidBodyAPI) and root.HasAPI(UsdPhysics.MassAPI)
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 2
    assert all(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == 'convexHull' for p in colliders)
    assert not stage.GetPrimAtPath('/Object/visual').HasAPI(UsdPhysics.CollisionAPI)


def test_pad_parts_keeps_padding_inside():
    from sim_eval.objects import pad_parts
    box = trimesh.creation.box(extents=(0.04, 0.02, 0.02))
    small = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    parts = [(box.vertices, box.faces), (small.vertices + [0.03, 0, 0], small.faces)]
    padded = pad_parts(parts, 5)
    assert len(padded) == 5 and padded[:2] == parts
    for v, f in padded[2:]:
        assert box.contains(v).all() and trimesh.Trimesh(v, f).volume > 0


def test_watertight_simplified_keeps_frame(tmp_path):
    pytest.importorskip('skimage')
    from sim_eval.objects import watertight_simplified
    shell = trimesh.creation.box(extents=(0.04, 0.02, 0.06))
    shell.apply_translation((0.3, -0.1, 0.05))
    shell.update_faces(np.arange(len(shell.faces)) != 0)  # open shell, like many Objaverse meshes
    out = watertight_simplified(shell, str(tmp_path), 'box', voxels=64, max_faces=2000)
    assert np.abs(out.bounds - shell.bounds).max() < 2 * 0.06 / 64
    assert len(out.faces) <= 2000


def test_summary():
    rows = []
    for i in range(10):
        for gripper in ('franka', 'so101'):
            rows.append({'gripper': gripper, 'success': str(int(i < 6 if gripper == 'franka' else i < 3)),
                         'fits': str(int(gripper == 'franka' or i < 5)), 'label': str(int(i < 5)),
                         'object_index': str(i // 5), 'initial_collision': str(int(i == 9)),
                         'first_failed_phase': '' if i < 6 else 'load+z'})
    summary = summarize(rows)
    assert summary['franka']['success'] == 0.6 and summary['so101']['success'] == 0.3
    assert summary['so101']['success_fits'] == 0.6
    assert summary['franka']['success_pos'] == 1.0 and summary['franka']['success_neg'] == 0.2
    metrics = wandb_metrics(summary)
    assert metrics['sim/success_franka'] == 0.6 and 'sim/failed_phase_franka' not in metrics


@pytest.mark.skipif(not os.path.exists(os.path.join(SO101_DIR, 'so101_new_calib.urdf')),
                    reason='set SO101_DIR to SO-ARM100/Simulation/SO101')
def test_so101_urdf(tmp_path):
    pytest.importorskip('coacd')
    path = G.so101_urdf(SO101_DIR, str(tmp_path))
    root = ET.parse(path).getroot()
    assert [j.get('name') for j in root.findall('joint')] == ['gripper']
    links = {link.get('name'): link for link in root.findall('link')}
    # the collision parts of the fixed jaw leave the space between the jaws free
    parts = [trimesh.load(c.find('geometry/mesh').get('filename'))
             for c in links['gripper_link'].findall('collision')]
    frame_inv = invert(G.SO101_LINK_TO_FRAME)
    gap_point = (G.SO101_LINK_TO_FRAME @ np.array([-0.02, 0, G.SO101_CONTACT_Z, 1]))[:3]
    assert not any(p.contains([gap_point])[0] for p in parts)
    fixed_x = min(trimesh.transform_points(p.vertices, frame_inv)[:, 0][
        np.abs(trimesh.transform_points(p.vertices, frame_inv)[:, 2] - G.SO101_CONTACT_Z) < 0.004].min(initial=1)
        for p in parts)
    assert abs(fixed_x - G.SO101_FIXED_JAW_X) < 0.004
