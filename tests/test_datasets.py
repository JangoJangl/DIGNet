"""
End-to-end tests of the dataset converters (dataset_tools/) on synthetic copies of ACRONYM, GraspGen and
GraspNet-1Billion, the unified training loader and a short flow matching training on full + partial clouds.
Figures go to test_outputs/ (see conftest.py).

    PYTHONPATH=.:diffusion pytest tests/test_datasets.py
"""
import os
import shutil

import numpy as np
import pytest
import torch

pytest.importorskip('datasets')
pytest.importorskip('open3d')
pytest.importorskip('h5py')

from conftest import register_figure  # noqa: E402
from synthetic_data import make_acronym, make_graspgen, make_graspnet  # noqa: E402

PATHS = {}  # (source, cloud) -> folder written by the converter
SOURCES = [('acronym', 'full'), ('acronym', 'partial'), ('graspgen', 'full'), ('graspgen', 'partial'),
           ('graspnet1b', 'full'), ('graspnet1b', 'partial')]


@pytest.fixture(scope='module')
def built(tmp_path_factory, figure_dir):
    """ converts all synthetic sources once, returns {(source, cloud): DatasetDict} """
    from dataset_tools.build_dataset import main
    from datasets import load_from_disk
    root = tmp_path_factory.mktemp('raw')
    roots = {'acronym': make_acronym(str(root / 'acronym')), 'graspgen': make_graspgen(str(root / 'graspgen'), str(root / 'objaverse')),
             'graspnet1b': make_graspnet(str(root / 'graspnet'))[0]}
    extra = {'graspgen': ['--object-root', str(root / 'objaverse')], 'graspnet1b': ['--max-objects', '1'], 'acronym': []}
    out = {}
    for source, cloud in SOURCES:
        path = str(root / 'unified' / f'{source}_{cloud}')
        main(['--source', source, '--cloud', cloud, '--root', roots[source], '--output', path, '--num-proc', '1',
              '--views', '2', '--preview', '4'] + extra[source])
        fig = os.path.join(figure_dir, f'dataset_{source}_{cloud}.png')
        shutil.copy(os.path.join(path, 'preview.png'), fig)
        register_figure(fig, f'{source}: {cloud} point clouds (converter output)')
        out[(source, cloud)] = load_from_disk(path)
        PATHS[(source, cloud)] = path
    return out


def rows(dataset):
    return [r for split in dataset.values() for r in split.with_format('numpy')]


def test_graspnet_frame_conversion():
    """ finger tips at target + depth * approach, closing along GraspNet y, approach along GraspNet x """
    from dataset_tools.sources import graspnet_to_ours, viewpoint_to_matrix
    rng = np.random.default_rng(0)
    R = viewpoint_to_matrix(rng.normal(size=(20, 3)), rng.uniform(0, np.pi, 20))
    target, depth = rng.normal(size=(20, 3)) * 0.1, rng.uniform(0.01, 0.04, 20)
    poses = graspnet_to_ours(R, target, depth)
    tips = (poses @ np.array([0, 0, 0.11217, 1.0]))[:, :3]
    assert np.allclose(tips, target + depth[:, None] * R[:, :, 0], atol=1e-9)
    assert np.allclose(poses[:, :3, 0], R[:, :, 1]) and np.allclose(poses[:, :3, 2], R[:, :, 0])
    assert np.allclose(np.linalg.det(poses[:, :3, :3]), 1)


@pytest.mark.parametrize('source,cloud', SOURCES)
def test_converter_output(built, source, cloud):
    data = rows(built[(source, cloud)])
    assert data, 'no rows written'
    for r in data:
        assert r['source'] == source and r['cloud_type'] == cloud and r['points'].shape[1] == 3
        good = r['success'] == 1
        assert good.any()
        # finger centres of successful grasps lie on or in the object: close to the observed cloud
        tips = (r['grasps'][good] @ np.array([0, 0, r['tcp_depth'], 1.0]))[:, :3]
        from scipy.spatial import cKDTree
        assert np.median(cKDTree(r['points']).query(tips)[0]) < 0.03
        # successful grasps found contacts on the object with both fingers
        assert (r['visibility_class'][good] >= 0).mean() > 0.9
        if cloud == 'full':
            assert np.allclose(r['camera_pose'], np.eye(4))
            assert set(np.unique(r['visibility_class'][good])) <= {2, -1}
        else:
            # a single view never sees every contact
            assert (r['visibility_class'][good] < 2).any()


def test_graspgen_scale_and_splits(built):
    data = rows(built[('graspgen', 'full')])
    assert {r['split'] for r in data} == {'train', 'test'} or {r['split'] for r in data} == {'validation', 'test'}
    assert all(abs(r['scale'] - 0.5) < 1e-6 for r in data)
    extent = max(np.ptp(r['points'], axis=0).max() for r in data)
    assert extent < 0.13  # meshes were stored twice as large, the converter has to apply the json scale


def test_flow_matching_trains_on_full_and_partial(built, tmp_path, monkeypatch, figure_dir):
    """ unified loader (full + partial ACRONYM), PointNet++ tokens + flow matching, float32 as in flow_pn2_20m """
    import hydra
    import wandb
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    paths = [PATHS[('acronym', 'full')], PATHS[('acronym', 'partial')]]

    torch.set_default_dtype(torch.float32)
    monkeypatch.chdir(tmp_path)
    OmegaConf.register_new_resolver('add', lambda *n: sum(n), replace=True)
    GlobalHydra.instance().clear()
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'diffusion', 'configs'))
    steps = 300
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose('flow_pn2_20m', overrides=[
            'device=cpu', f'unified.paths=[{paths[0]},{paths[1]}]', 'token_dim=64', 'pointnet2_width=1',
            'attention_hidden_dim=64', 'attention_layers=2', 'attention_heads=4', f'max_train_steps={steps}',
            f'eval_every_n_steps={steps}', 'num_objects=4', 'batch_size=16', 'num_vis_objects=3', 'max_eval_objects=3',
            'num_eval_samples=50', 'unified.test_split=train', 'random_rotation=false', 'agents.optimization.lr=1e-3'])
    if not os.environ.get('ALR_SIM_AVAILABLE'):
        # the gripper collision check loads the Panda meshes through the ALR simulation framework
        from agents.diffusion_agents import grasp_diffusion_agent as gda
        monkeypatch.setattr(gda, 'collision_detection_pc_mesh', lambda *a, **k: torch.IntTensor([]))
        monkeypatch.setattr(gda, 'not_in_gripper_detection_pc_mesh', lambda *a, **k: torch.IntTensor([]))
    wandb.init(mode='disabled')
    try:
        workspace = hydra.utils.instantiate(cfg.workspaces)
        loader = workspace.data_loaders['train']
        points, grasps = next(iter(loader))
        assert points.shape == (4, cfg.pc_dim, 3) and grasps.shape == (4, 16, 6) and points.dtype == torch.float32
        agent = hydra.utils.instantiate(cfg.agents)
        agent.set_processor(workspace.get_processor())
        agent.set_bounds(cfg.grasp_dim)
        agent.train_agent(loader, workspace.data_loaders['test'])
    finally:
        wandb.finish()
        torch.set_default_dtype(torch.float32)
    pngs = sorted(os.listdir(tmp_path / 'visualizations'))
    assert len(pngs) == 4
    fig = os.path.join(figure_dir, 'model_flow_matching_after_training.png')
    shutil.copy(tmp_path / 'visualizations' / pngs[-1], fig)
    register_figure(fig, f'flow matching (small model, {steps} CPU steps) on full + partial ACRONYM test objects')
