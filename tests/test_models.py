"""
CPU tests for the encoders, the cross-attention denoiser, SE(3) flow matching and the depth renderer.

Run from the repository root:
    PYTHONPATH=.:diffusion pytest tests
"""
import math

import numpy as np
import pytest
import torch

from helper import so3

hydra = pytest.importorskip("hydra")


@pytest.fixture(autouse=True)
def float64_default():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


def sphere_cloud(batch, n=1024, radius=0.05):
    p = torch.randn(batch, n, 3)
    return p / torch.linalg.vector_norm(p, dim=-1, keepdim=True) * radius


def test_so3_exp_log_roundtrip_including_angles_near_pi():
    w = torch.randn(256, 3)
    w = w / torch.linalg.vector_norm(w, dim=-1, keepdim=True) * torch.rand(256, 1) * (math.pi - 1e-3)
    w[0] = torch.tensor([0.0, 0.0, 0.0])
    w[1] = torch.tensor([1e-7, 0.0, 0.0])
    w[2] = torch.tensor([0.0, math.pi - 1e-6, 0.0])
    R = so3.exp_map(w)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand_as(R), atol=1e-10)
    assert torch.allclose(so3.exp_map(so3.log_map(R)), R, atol=1e-8)
    assert torch.allclose(so3.log_map(R)[3:], w[3:], atol=1e-8)


def test_serialized_attention_restores_point_order():
    from networks.vision_encoder.ptv3 import SerializedAttention, serialization_orders
    attn = SerializedAttention(16, 2, patch_size=64)
    x, xyz = torch.randn(2, 64, 16), sphere_cloud(2, 64)
    order = serialization_orders(xyz, 0.002)[0]
    # with one patch covering all points, patch attention equals global attention whatever the order
    assert torch.allclose(attn(x, order), attn(x, torch.arange(64).expand(2, -1)), atol=1e-10)


@pytest.mark.parametrize("embedder", ["pointnet2_local", "ptv3"])
def test_token_embedders(embedder):
    from networks.vision_encoder.pointnet2_local import PointNet2Local
    from networks.vision_encoder.ptv3 import PTv3Encoder
    enc = PointNet2Local(output_dim=32) if embedder == "pointnet2_local" else PTv3Encoder(output_dim=32, patch_size=64)
    condition, B = enc(sphere_cloud(2))
    assert B == 2 and condition.shape == (2, 128, 3 + 32)
    rel = enc.compute_features(condition, None, torch.ones(2, 3))
    assert torch.allclose(rel[..., :3], condition[..., :3] - 1)


def build(model_cfg):
    return hydra.utils.instantiate(model_cfg)


def compose_model(model, embedder):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    import os
    OmegaConf.register_new_resolver("add", lambda *n: sum(n), replace=True)
    GlobalHydra.instance().clear()
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diffusion", "configs"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose("train_config", overrides=[
            f"agents/model={model}", f"agents/model/embedder={embedder}", "device=cpu", "token_dim=32",
            "attention_hidden_dim=32", "n_timesteps=5", "flow_steps=3"])
    return cfg.agents.model


def identity_inverse(x):
    grasps = torch.eye(4).repeat(len(x), 1, 1)
    grasps[:, :3, :3] = so3.exp_map(x[:, 3:6])
    grasps[:, :3, 3] = x[:, :3]
    return grasps


@pytest.mark.parametrize("model", ["flow_matching_model", "attention_diffusion_model"])
@pytest.mark.parametrize("embedder", ["pointnet2_local", "ptv3"])
def test_models_train_and_sample_from_configs(model, embedder):
    net = build(compose_model(model, embedder))
    net.inv_fct = identity_inverse
    net.max_input, net.min_input = torch.full((6,), math.pi), torch.full((6,), -math.pi)
    points = sphere_cloud(2)
    grasps = torch.cat([torch.randn(8, 3) * 0.05, torch.randn(8, 3)], dim=-1)  # 2 objects x 4 grasps
    loss = net.loss(grasps, points)
    loss.backward()
    assert torch.isfinite(loss)
    net.eval()
    with torch.no_grad():
        samples = net(points, num_samples=3)
    assert samples.shape == (6, 6) and torch.isfinite(samples).all()


def test_flow_matching_targets_follow_the_geodesic():
    # the body frame velocity Log(R0^T R1) integrated for t = 1 must reach R1 exactly
    R0, R1 = so3.random_rotation(16, dtype=torch.float64), so3.random_rotation(16, dtype=torch.float64)
    w = so3.log_map(R0.transpose(-1, -2) @ R1)
    assert torch.allclose(R0 @ so3.exp_map(w), R1, atol=1e-8)


def test_depth_renderer_sees_only_the_front_of_a_box():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("open3d")
    from helper.rendering import DepthRenderer
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    renderer = DepthRenderer(sigma_base=0.0, sigma_quad=0.0, dropout=0.0, edge_dropout=0.0)
    rng = np.random.default_rng(0)
    eye, _, _ = renderer.random_camera(np.random.default_rng(0))
    points = renderer.render(box, rng)
    assert len(points) > 100
    # noise free points lie on the box surface ...
    assert np.allclose(np.abs(points).max(axis=1), 0.05, atol=1e-4)
    # ... and only on faces that point towards the camera
    normals = np.eye(3)[np.abs(points).argmax(axis=1)] * np.sign(points[np.arange(len(points)), np.abs(points).argmax(axis=1)])[:, None]
    assert ((eye - points) * normals).sum(axis=1).min() > 0
    cloud = renderer.partial_cloud(box, 1024, rng=np.random.default_rng(1))
    assert cloud.shape == (1024, 3)
