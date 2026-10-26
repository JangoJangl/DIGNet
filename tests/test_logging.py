"""
CPU tests for the headless plots, the validation visualisations of GraspDiffusionAgent and the
Hugging Face upload hook (HfApi is replaced by a fake, nothing leaves the machine).

    PYTHONPATH=.:diffusion pytest tests
"""
import os
import sys
import types

import numpy as np
import pytest
import torch

hydra = pytest.importorskip("hydra")
wandb = pytest.importorskip("wandb")
pytest.importorskip("matplotlib")

from helper.plotting import gripper_segments, grasp_figure  # noqa: E402
from tests.test_models import identity_inverse, sphere_cloud  # noqa: E402


def test_gripper_marker_fingertips_at_the_grasp_origin():
    pose = torch.eye(4)[None]
    seg = gripper_segments(pose)
    fingertips = seg[0, 2:, 1]  # end points of both fingers
    assert np.allclose(fingertips[:, 2], 0) and np.allclose(np.abs(fingertips[:, 0]), 0.041)
    assert np.allclose(gripper_segments(pose, frame='wrist')[0, 0, 0], 0)  # wrist frame: stem starts at the origin


def test_grasp_figure_renders(tmp_path):
    poses = identity_inverse(torch.randn(5, 6) * 0.05)
    fig = grasp_figure([{'points': sphere_cloud(1)[0], 'gt': poses, 'pred': poses, 'title': 'a'}] * 2,
                       views=((25, 45), (80, -90)))
    fig.savefig(tmp_path / 'f.png')
    assert len(fig.axes) == 4 and (tmp_path / 'f.png').stat().st_size > 0


class FakeHfApi:
    calls = []

    def create_repo(self, repo_id, **kwargs):
        FakeHfApi.calls.append(('create_repo', repo_id))

    def upload_folder(self, **kwargs):
        FakeHfApi.calls.append(('upload_folder', kwargs['path_in_repo'], kwargs['commit_message'],
                                sorted(os.listdir(kwargs['folder_path']))))


class Loader(list):
    """ list of batches with the .dataset.processor attribute the agent reads """
    def __init__(self, batches):
        super().__init__(batches)
        self.dataset = types.SimpleNamespace(processor=types.SimpleNamespace(current_obj=None))


@pytest.fixture
def agent_module(monkeypatch):
    # helper.visualize needs the ALR simulation framework, which is not required for training
    for name in ('alr_sim', 'alr_sim.utils', 'alr_sim.utils.sim_path', 'alr_sim.utils.geometric_transformation'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules['alr_sim.utils.sim_path'].sim_framework_path = lambda *a: ''
    sys.modules['alr_sim.utils.geometric_transformation'].quat2mat = lambda *a: None
    pytest.importorskip("sklearn")
    pytest.importorskip("open3d")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'HfApi', FakeHfApi)
    from agents.diffusion_agents import grasp_diffusion_agent
    return grasp_diffusion_agent


def test_agent_visualises_at_fractions_and_uploads(agent_module, tmp_path, monkeypatch):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    torch.set_default_dtype(torch.float64)
    monkeypatch.chdir(tmp_path)
    OmegaConf.register_new_resolver("add", lambda *n: sum(n), replace=True)
    GlobalHydra.instance().clear()
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diffusion", "configs"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose("train_config", overrides=[
            "agents/model=flow_matching_model", "agents/model/embedder=pointnet2_local", "device=cpu",
            "token_dim=32", "attention_hidden_dim=32", "flow_steps=2", "max_train_steps=8",
            "eval_every_n_steps=100", "num_vis_objects=2", "hub.repo_id=someone/dignet-test"])
    wandb.init(mode='disabled')
    try:
        agent = hydra.utils.instantiate(cfg.agents)
        agent.set_processor(types.SimpleNamespace(postprocess=identity_inverse, representation_name='se3_pseudo_log_map_representation',
                                                  representation='se3_pseudo_log_map_representation'))
        agent.model.max_input, agent.model.min_input = torch.full((6,), np.pi), torch.full((6,), -np.pi)
        monkeypatch.setattr(agent, 'evaluate_agent', lambda loader: (0.1, 0.1, 0.5, 0.0))  # metrics are tested elsewhere
        grasps = lambda b: torch.cat([torch.randn(b, 4, 3) * 0.05, torch.randn(b, 4, 3)], dim=-1)
        train = Loader([(sphere_cloud(2, n=512), grasps(2))])
        test = Loader([(sphere_cloud(1, n=512), grasps(1)) for _ in range(3)])
        agent.train_agent(train, test)
    finally:
        wandb.finish()

    pngs = sorted(os.listdir(tmp_path / 'visualizations'))
    assert [p.split('_')[-1] for p in pngs] == ['025pct.png', '050pct.png', '075pct.png', '100pct.png']
    assert os.path.exists(tmp_path / 'final' / 'model_state_dict.pth')
    uploads = [c for c in FakeHfApi.calls if c[0] == 'upload_folder']
    assert uploads[0][1].startswith('runs/') and 'best' in uploads[0][2] and 'final' in uploads[-1][2]


def test_training_state_roundtrip_and_resume(agent_module, tmp_path, monkeypatch):
    """ optimizer, EMA and step survive save_training_state / resume_training, training continues from there """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    torch.set_default_dtype(torch.float64)
    monkeypatch.chdir(tmp_path)
    OmegaConf.register_new_resolver("add", lambda *n: sum(n), replace=True)
    GlobalHydra.instance().clear()
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diffusion", "configs"))

    def make_agent(max_steps):
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose("train_config", overrides=[
                "agents/model=flow_matching_model", "agents/model/embedder=pointnet2_local", "device=cpu",
                "token_dim=32", "attention_hidden_dim=32", "flow_steps=2", f"max_train_steps={max_steps}",
                "eval_every_n_steps=100", "checkpoint_every_n_steps=3", "visualize_at=[]", "use_ema=true"])
        agent = hydra.utils.instantiate(cfg.agents)
        agent.set_processor(types.SimpleNamespace(postprocess=identity_inverse, representation_name='se3_pseudo_log_map_representation',
                                                  representation='se3_pseudo_log_map_representation'))
        agent.model.max_input, agent.model.min_input = torch.full((6,), np.pi), torch.full((6,), -np.pi)
        monkeypatch.setattr(agent, 'evaluate_agent', lambda loader: (0.1, 0.1, 0.5, 0.0))
        return agent

    grasps = lambda b: torch.cat([torch.randn(b, 4, 3) * 0.05, torch.randn(b, 4, 3)], dim=-1)
    train = Loader([(sphere_cloud(2, n=512), grasps(2))])
    test = Loader([(sphere_cloud(1, n=512), grasps(1))])
    wandb.init(mode='disabled')
    try:
        first = make_agent(6)
        first.train_agent(train, test)
        assert os.path.exists(tmp_path / 'last' / 'training_state.pth')  # written at steps 3 and 6
        resumed = make_agent(9)
        resumed.resume_training(str(tmp_path / 'final'))
        assert resumed.steps == 6
        a, b = first.optimizer.state_dict()['state'], resumed.optimizer.state_dict()['state']
        assert a.keys() == b.keys() and all(torch.equal(a[k]['exp_avg'], b[k]['exp_avg']) for k in a)
        assert all(torch.equal(p, q) for p, q in zip(first.ema_helper.shadow_params, resumed.ema_helper.shadow_params))
        resumed.train_agent(train, test)
        assert resumed.steps == 9
    finally:
        wandb.finish()
