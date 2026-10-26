"""
CPU unit tests for the parts of the pipeline that do not need the ACRONYM dataset or a GPU.

Run from the repository root:
    PYTHONPATH=.:diffusion pytest tests
"""
import numpy as np
import pytest
import torch

from helper.augmentation import Augmentation
from helper.losses import trafo_chamfer_distance, transform_control_points, pairwise_gripper_distance


def random_grasps(n, generator=None):
    grasps = torch.eye(4, dtype=torch.float64).repeat(n, 1, 1)
    for i in range(n):
        grasps[i, :3, :3] = Augmentation.random_rotation(generator=generator)
    grasps[:, :3, 3] = torch.randn(n, 3, dtype=torch.float64, generator=generator) * 0.1
    return grasps


def sphere_points(n, radius=0.05):
    p = torch.randn(n, 3, dtype=torch.float64)
    return p / torch.linalg.vector_norm(p, dim=1, keepdim=True) * radius


def test_random_rotation_is_proper():
    R = Augmentation.random_rotation()
    assert torch.allclose(R @ R.T, torch.eye(3, dtype=R.dtype), atol=1e-10)
    assert torch.isclose(torch.linalg.det(R), torch.tensor(1.0, dtype=R.dtype))


def test_rotation_keeps_grasps_attached_to_object():
    points = sphere_points(256)
    grasps = random_grasps(8)
    R = Augmentation.random_rotation()
    r_points, r_grasps = Augmentation.rotate(points, grasps, R)
    # the object expressed in every gripper frame must not change
    before = torch.linalg.inv(grasps) @ torch.cat([points, torch.ones(256, 1, dtype=points.dtype)], 1).T
    after = torch.linalg.inv(r_grasps) @ torch.cat([r_points, torch.ones(256, 1, dtype=points.dtype)], 1).T
    assert torch.allclose(before, after, atol=1e-10)


def test_hidden_point_removal_keeps_front_side():
    points = sphere_points(2000).numpy()
    camera = np.array([0.0, 0.0, 0.5])
    mask = Augmentation.visible_point_mask(points, camera)
    assert 0 < mask.sum() < len(points)
    # visible points face the camera
    assert (points[mask, 2] > -0.01).mean() > 0.95


def test_partial_view_shape_and_recentering():
    points = sphere_points(4096)
    grasps = random_grasps(4)
    p_points, p_grasps = Augmentation.partial_view(points, grasps, num_points=1024)
    assert p_points.shape == (1024, 3)
    assert torch.allclose(p_points.mean(dim=0), torch.zeros(3, dtype=p_points.dtype), atol=1e-10)
    # grasps were shifted by the same offset as the points, rotations untouched
    assert torch.allclose(p_grasps[:, :3, :3], grasps[:, :3, :3])


def naive_chamfer(x, y, direction):
    px, py = transform_control_points(x), transform_control_points(y)
    d = np.array([[np.mean(np.linalg.norm((a - b).numpy(), axis=1)) for b in py] for a in px])
    return d.min(axis=0).mean() if direction == 'y_to_x' else d.min(axis=1).mean()


@pytest.mark.parametrize("direction", ['y_to_x', 'x_to_y'])
def test_chamfer_matches_naive_implementation(direction):
    x, y = random_grasps(17), random_grasps(23)
    assert np.isclose(trafo_chamfer_distance(x, y, direction), naive_chamfer(x, y, direction))


def test_symmetric_metric_ignores_finger_flip():
    grasps = random_grasps(5)
    flip = torch.eye(4, dtype=torch.float64)
    flip[:3, :3] = torch.tensor([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]], dtype=torch.float64)  # 180 deg about z
    flipped = grasps @ flip
    assert trafo_chamfer_distance(flipped, grasps, 'x_to_y') > 0.05
    assert trafo_chamfer_distance(flipped, grasps, 'x_to_y', symmetric=True) < 1e-3


def test_coverage_distances_have_one_entry_per_ground_truth_grasp():
    x, y = random_grasps(10), random_grasps(30)
    _, dist = trafo_chamfer_distance(x, y, 'y_to_x', coverage=True)
    assert dist.shape == (30,)


def test_pairwise_distance_chunking():
    px, py = transform_control_points(random_grasps(40)), transform_control_points(random_grasps(7))
    assert torch.allclose(pairwise_gripper_distance(px, py, chunk_size=3), pairwise_gripper_distance(px, py))


def test_repeat_condition():
    hydra = pytest.importorskip("hydra")
    from networks.diffusion.point_diffusion import Diffusion
    cond = torch.arange(3, dtype=torch.float64).view(3, 1)
    rep = Diffusion.repeat_condition(cond, 3, 6)
    assert rep.view(-1).tolist() == [0, 0, 1, 1, 2, 2]
    hierarchical = [(torch.zeros(1), torch.zeros(1))]
    assert Diffusion.repeat_condition(hierarchical, 1, 6) is hierarchical
    with pytest.raises(ValueError):
        Diffusion.repeat_condition(cond, 3, 7)


def test_points_in_gripper_frame():
    pytest.importorskip("hydra")
    from networks.diffusion.embedder_free_diffusion import Diffusion
    grasps = random_grasps(4)
    points = sphere_points(64)
    homogeneous = torch.cat([points, torch.ones(64, 1, dtype=points.dtype)], 1).T.repeat(4, 1, 1)
    model = Diffusion.__new__(Diffusion)  # only the geometric helper is tested, no network needed
    model.inv_fct = lambda x: x
    local = Diffusion.points_in_gripper_frame(model, grasps, homogeneous)
    expected = (torch.linalg.inv(grasps) @ homogeneous).transpose(1, 2)[..., :3]
    assert torch.allclose(local, expected, atol=1e-10)


def translation_only_inverse(x):
    grasps = torch.eye(4, dtype=x.dtype).repeat(len(x), 1, 1)
    grasps[:, :3, 3] = x[:, :3]
    return grasps


def test_point_diffusion_trains_and_samples_with_several_grasps_per_object():
    pytest.importorskip("hydra")
    from networks.diffusion.point_diffusion import Diffusion
    torch.set_default_dtype(torch.float64)
    try:
        model = Diffusion(
            embedder={'_target_': 'networks.vision_encoder.pointnet2.Pointnet2', 'output_dim': 64, 'radius': 0.05},
            input_dim=6,
            model={'_target_': 'networks.diffusion.diffusion_models.DiffusionMLPNetwork', 'input_dim': 6,
                   'cond_dim': 64, 't_dim': 16, 'hidden_dim': 32, 'num_hidden_layers': 2, 'device': 'cpu'},
            beta_schedule='cosine', n_timesteps=5, loss_type='l2', clip_denoised=True, device='cpu')
        model.inv_fct = translation_only_inverse
        model.max_input, model.min_input = torch.ones(6), -torch.ones(6)
        points = sphere_points(1024).repeat(2, 1, 1)
        grasps = torch.rand(2 * 4, 6) * 0.1  # 2 objects with 4 grasps each, object-major
        loss = model.loss(grasps, points)
        loss.backward()
        assert torch.isfinite(loss)
        model.eval()
        with torch.no_grad():
            samples = model(points, num_samples=3)
        assert samples.shape == (6, 6)
    finally:
        torch.set_default_dtype(torch.float32)
