"""
Plots selected samples of the training data exactly as the model sees them (after preprocessing and
augmentation) with their ground truth grasps.

    python visualize_dataset.py dataset_vis.indices=[0,7,42] partial_view_prob=1.0 partial_mode=render

The figure is written to the Hydra run directory (dataset_samples.png) and optionally logged to wandb.
"""
import logging
import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from helper.plotting import grasp_figure

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train_config")
def main(cfg: DictConfig) -> None:
    torch.set_default_dtype(getattr(torch, cfg.get('dtype', 'float64')))
    workspace_manager = hydra.utils.instantiate(cfg.workspaces)
    dataset = workspace_manager.data_loaders[cfg.dataset_vis.split].dataset
    samples = []
    for idx in cfg.dataset_vis.indices:
        points, grasps = dataset[idx]
        poses = dataset.processor.postprocess(torch.flatten(grasps, 0, -2) if grasps.dim() > 2 else grasps)
        if hasattr(dataset, 'file_names'):  # ACRONYM loader
            name = os.path.splitext(dataset.file_names[dataset.batches[idx][0] if dataset.train_flag else idx])[0]
        else:  # unified datasets (workspaces=unified_grasp_manager)
            name = str(dataset.data[dataset.index[idx]]['sample_id'])
        samples.append({'points': points[:, :3], 'gt': poses,
                        'title': f"{cfg.dataset_vis.split} {idx}: {name[:20]}\n{len(points)} points, {len(poses)} grasps"})
    fig = grasp_figure(samples, max_grasps=cfg.dataset_vis.max_grasps, views=((25, 45), (80, -90)))
    path = os.path.join(os.getcwd(), 'dataset_samples.png')
    fig.savefig(path, dpi=120)
    log.info(f"Saved {path}")
    if cfg.dataset_vis.wandb:
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, group=cfg.group, job_type='dataset_vis',
                   config=OmegaConf.to_container(cfg, resolve=True))
        wandb.log({'dataset_samples': wandb.Image(fig)})
        wandb.finish()


if __name__ == "__main__":
    main()
