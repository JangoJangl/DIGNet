"""
Training data from the unified grasp datasets written by dataset_tools/build_dataset.py (ACRONYM, GraspGen,
GraspNet-1Billion; full and partial clouds). Several datasets can be mixed, e.g. full + partial ACRONYM.
"""
import os
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def load_unified(path, split):
    """ path: folder written by build_dataset.py or a Hugging Face dataset repo id """
    from datasets import load_dataset, load_from_disk
    if os.path.isdir(path):
        dataset = load_from_disk(path)
        return dataset[split] if split in dataset else None
    try:
        return load_dataset(path, split=split)
    except ValueError:  # split does not exist in this repo
        return None


class UnifiedGraspDataset(Dataset):
    """
    One item = one observation (full or partial cloud) with grasps.

    train: `subbatch_size` successful grasps drawn at random each time (with replacement if there are fewer),
           the processor applies the training augmentation.
    test:  all successful grasps (at most max_eval_grasps), no augmentation.
    """

    def __init__(self, paths: Sequence[str], split: str, subbatch_size: int, processor, num_points: int = 1024,
                 grippers: Sequence[str] = ('franka_panda',), cloud_types: Sequence[str] = ('full', 'partial'),
                 sources: Optional[Sequence[str]] = None, max_eval_grasps: int = 1000, train: bool = True,
                 device: str = 'cpu', seed: int = 0):
        from datasets import concatenate_datasets
        parts = [d for d in (load_unified(p, split) for p in paths) if d is not None and len(d)]
        if not parts:
            raise ValueError(f"no '{split}' data in {list(paths)}")
        common = set.intersection(*(set(d.column_names) for d in parts))
        dataset = concatenate_datasets([d.select_columns(sorted(common)) for d in parts])
        keep = lambda g, c, s: g in grippers and c in cloud_types and (sources is None or s in sources)
        self.data = dataset.filter(keep, input_columns=['gripper', 'cloud_type', 'source']).with_format('numpy')
        self.subbatch_size = subbatch_size
        self.processor = processor
        self.num_points = num_points
        self.max_eval_grasps = max_eval_grasps
        self.train_flag = train
        self.device = device
        self.rng = np.random.default_rng(seed)
        # rows without a successful grasp can not be used to train or evaluate a generator
        self.index = [i for i, s in enumerate(self.data['success']) if np.any(s == 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.data[self.index[idx]]
        points = row['points']
        points = points[self.rng.choice(len(points), self.num_points, replace=len(points) < self.num_points)]
        grasps = row['grasps'][row['success'] == 1]
        if self.train_flag:
            grasps = grasps[self.rng.choice(len(grasps), self.subbatch_size, replace=len(grasps) < self.subbatch_size)]
        elif len(grasps) > self.max_eval_grasps:
            grasps = grasps[self.rng.choice(len(grasps), self.max_eval_grasps, replace=False)]
        dtype = torch.get_default_dtype()
        tcp = float(row['tcp_depth']) if np.isfinite(row['tcp_depth']) else 0.11217
        return self.processor.preprocess_cloud(torch.as_tensor(points, dtype=dtype, device=self.device),
                                               torch.as_tensor(grasps, dtype=dtype, device=self.device),
                                               tcp_depth=tcp, augment=self.train_flag)


def get_train_val(paths, subbatch_size, processor, num_points=1024, grippers=('franka_panda',),
                  cloud_types=('full', 'partial'), sources=None, test_split='test', max_eval_grasps=1000,
                  device='cpu', **kwargs):
    """ hydra entry point (workspaces/unified_grasp_manager.yaml); processor is a DictConfig or an instance """
    import hydra
    from omegaconf import DictConfig
    if isinstance(processor, DictConfig):
        processor = hydra.utils.instantiate(processor)
    paths = [paths] if isinstance(paths, str) else list(paths)
    common = dict(subbatch_size=subbatch_size, processor=processor, num_points=num_points, grippers=tuple(grippers),
                  cloud_types=tuple(cloud_types), sources=None if sources is None else tuple(sources),
                  max_eval_grasps=max_eval_grasps, device=device)
    return (UnifiedGraspDataset(paths, 'train', train=True, **common),
            UnifiedGraspDataset(paths, test_split, train=False, **common))
