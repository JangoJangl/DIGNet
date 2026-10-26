"""
Writes an episode file (sim_eval/episodes.py) for the simulation.

  # grasps of a checkpoint on the validation split (data paths are taken from the run's config)
  python -m sim_eval.export_grasps model --run Thetatec/dignet-flow-pn2-20m:runs/<run> --weights last \
      --split validation --max-objects 100 --grasps-per-object 32 --out episodes.npz

  # dataset grasps with their success labels: checks the simulator against the labels (calibration)
  python -m sim_eval.export_grasps dataset --paths Thetatec/graspgen-unified-full --split validation \
      --max-objects 50 --grasps-per-object 32 --out gt_episodes.npz

--run is a local run directory (hydra output dir with .hydra/config.yaml) or '<hf repo>:<path in repo>'.
"""
import argparse
import logging
import os
import sys

import numpy as np

from sim_eval.episodes import EpisodeBuilder, cloud_to_mesh_frame

log = logging.getLogger(__name__)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows_of_split(paths, split, grippers=('franka_panda',), cloud_types=('full', 'partial'), max_objects=None, seed=0):
    """ dataset rows with at least one successful grasp, all views of at most max_objects objects """
    sys.path[:0] = [REPO, os.path.join(REPO, 'diffusion')]
    from datasets import concatenate_datasets
    from environments.diffusion_grasp.unified_dataset import load_unified
    parts = [d for d in (load_unified(p, split) for p in paths) if d is not None and len(d)]
    if not parts:
        raise ValueError(f"no '{split}' split in {paths}")
    data = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    data = data.filter(lambda g, c: g in grippers and c in cloud_types, input_columns=['gripper', 'cloud_type'])
    meta = data.select_columns(['object_id', 'mesh_file', 'scale', 'success'])
    objects = sorted({(m, round(float(s), 6)) for m, s in zip(meta['mesh_file'], meta['scale'])})
    rng = np.random.default_rng(seed)
    if max_objects is not None and len(objects) > max_objects:
        objects = [objects[i] for i in np.sort(rng.choice(len(objects), max_objects, replace=False))]
    chosen = set(objects)
    keep = [i for i, (m, s, succ) in enumerate(zip(meta['mesh_file'], meta['scale'], meta['success']))
            if (m, round(float(s), 6)) in chosen and np.any(np.asarray(succ) == 1)]
    return data.select(keep).with_format('numpy')


def download_run(run, weights):
    """ '<repo>:<path>' or a local directory -> local run directory with .hydra/config.yaml and the weights """
    if os.path.isdir(run):
        return run
    from huggingface_hub import snapshot_download
    repo_id, path = run.split(':', 1)
    path = path.strip('/')
    sub = {'best': '', 'last': 'last/', 'final': 'final/'}[weights]
    local = snapshot_download(repo_id, allow_patterns=[f'{path}/.hydra/*', f'{path}/{sub}*.pth'])
    return os.path.join(local, path)


def load_agent(run_dir, weights='last', device='cuda'):
    """ agent and processor of a training run with the chosen weights (EMA weights if the run used EMA) """
    import hydra
    import torch
    from omegaconf import OmegaConf
    sys.path[:0] = [REPO, os.path.join(REPO, 'diffusion')]
    OmegaConf.register_new_resolver('add', lambda *n: sum(n), replace=True)
    cfg = OmegaConf.load(os.path.join(run_dir, '.hydra', 'config.yaml'))
    cfg.device = device
    cfg.hub.repo_id = None
    torch.set_default_dtype(getattr(torch, cfg.get('dtype', 'float64')))
    agent = hydra.utils.instantiate(cfg.agents)
    processor = hydra.utils.instantiate(cfg.workspaces.dataset_fn.processor)
    agent.set_processor(processor)
    agent.set_bounds(cfg.grasp_dim)
    step = None
    if weights == 'best':
        agent.model.load_state_dict(torch.load(os.path.join(run_dir, 'model_state_dict.pth'), map_location=device))
        ema_path = os.path.join(run_dir, 'ema_state_dict.pth')
        ema = torch.load(ema_path, map_location=device) if os.path.exists(ema_path) else None
    else:
        state = torch.load(os.path.join(run_dir, weights, 'training_state.pth'), map_location=device, weights_only=False)
        agent.model.load_state_dict(state['model'])
        ema, step = state.get('ema'), int(state['steps'])
    if agent.use_ema and ema is not None:
        agent.ema_helper.load_state_dict(ema)
        agent.ema_helper.copy_to(agent.model.get_params())
    agent.model.eval()
    return agent, processor, cfg, step


def predict_row(agent, processor, row, num_samples, num_points, rng, device):
    """ num_samples grasps for one observation, base link poses in the mesh frame """
    import torch
    points = row['points']
    points = points[rng.choice(len(points), num_points, replace=len(points) < num_points)]
    centre = points.mean(axis=0)
    dtype = torch.get_default_dtype()
    tcp = float(row['tcp_depth']) if np.isfinite(row['tcp_depth']) else 0.11217
    pts, _ = processor.preprocess_cloud(torch.as_tensor(points, dtype=dtype, device=device),
                                        torch.as_tensor(row['grasps'][:1], dtype=dtype, device=device),
                                        tcp_depth=tcp, augment=False)
    gram_schmidt = processor.representation_name == 'rotation_6d_representation'
    with torch.no_grad():
        pred = processor.postprocess(agent.model(pts[None], gram_schmidt=gram_schmidt, num_samples=num_samples))
    pred = pred.detach().cpu().double().numpy()
    return cloud_to_mesh_frame(pred, row['camera_pose'], centre=centre, tcp_depth=tcp)


def export_model(args):
    import torch
    run_dir = download_run(args.run, args.weights)
    agent, processor, cfg, step = load_agent(run_dir, args.weights, args.device)
    paths = args.paths or list(cfg.unified.paths)
    rows = rows_of_split(paths, args.split, max_objects=args.max_objects, seed=args.seed,
                         cloud_types=tuple(args.cloud_types or cfg.unified.cloud_types))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    builder = EpisodeBuilder(args.grasps_per_object)
    for i in range(len(rows)):
        row = rows[i]
        grasps = predict_row(agent, processor, row, args.grasps_per_object, cfg.pc_dim, rng, args.device)
        builder.add(row, grasps, np.full(len(grasps), -1))
    meta = {'kind': 'model', 'run': args.run, 'weights': args.weights, 'step': step, 'split': args.split,
            'paths': paths}
    builder.save(args.out, meta, rng)
    log.info(f"{len(builder.objects)} objects, {len(builder.observations)} observations -> {args.out} (step {step})")
    return args.out, step


def export_dataset(args):
    rows = rows_of_split(args.paths, args.split, max_objects=args.max_objects, seed=args.seed,
                         cloud_types=tuple(args.cloud_types or ('full',)))
    rng = np.random.default_rng(args.seed)
    builder = EpisodeBuilder(args.grasps_per_object)
    for i in range(len(rows)):
        row = rows[i]
        builder.add(row, cloud_to_mesh_frame(row['grasps'], row['camera_pose']), row['success'])
    builder.save(args.out, {'kind': 'dataset', 'split': args.split, 'paths': list(args.paths)}, rng)
    log.info(f"{len(builder.objects)} objects -> {args.out}")
    return args.out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    for name in ('model', 'dataset'):
        p = sub.add_parser(name)
        p.add_argument('--paths', nargs='*', default=None, help='unified datasets (folders or HF repo ids)')
        p.add_argument('--split', default='validation')
        p.add_argument('--cloud-types', nargs='*', default=None)
        p.add_argument('--max-objects', type=int, default=100)
        p.add_argument('--grasps-per-object', type=int, default=32)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--out', required=True)
        if name == 'model':
            p.add_argument('--run', required=True)
            p.add_argument('--weights', choices=('best', 'last', 'final'), default='last')
            p.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.mode == 'model':
        return export_model(args)
    if not args.paths:
        parser.error('--paths is required for dataset exports')
    return export_dataset(args)


if __name__ == '__main__':
    main()
