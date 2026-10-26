"""
Figures of unified dataset samples: point cloud + successful grasps, coloured by how many contacts are
visible in the cloud (green 2, orange 1, red 0, grey unknown).

    python -m dataset_tools.preview /data/unified/acronym_partial --split train --num 6 --out preview.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

CLASS_COLORS = {2: (0.1, 0.7, 0.2), 1: (1.0, 0.6, 0.0), 0: (0.85, 0.1, 0.1), -1: (0.5, 0.5, 0.5)}


def sample_figure(rows, grasps_per_class=8, seed=0):
    import matplotlib.pyplot as plt
    from helper.plotting import plot_grasps
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(4 * len(rows), 8.4))
    for k, s in enumerate(rows):
        cls, success = np.asarray(s['visibility_class']), np.asarray(s['success'])
        sets = []
        for c, color in CLASS_COLORS.items():
            sel = np.flatnonzero((cls == c) & (success == 1))
            sel = rng.choice(sel, min(grasps_per_class, len(sel)), replace=False) if len(sel) else sel
            sets.append((np.asarray(s['grasps'])[sel], color))
        counts = '/'.join(str(int(((cls == c) & (success == 1)).sum())) for c in (2, 1, 0))
        # top row: from the camera for partial clouds (z forward, y down), bottom row: from the side
        views = ((-70, -90), (10, -60)) if s['cloud_type'] == 'partial' else ((30, 45), (5, -45))
        for r, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, len(rows), r * len(rows) + k + 1, projection='3d')
            title = (f"{s['source']} {s['cloud_type']} {s['gripper']}\n{s['object_id'][:16]} view {s['view']}\n"
                     f"successful, contacts visible 2/1/0: {counts}") if r == 0 else None
            plot_grasps(ax, np.asarray(s['points']), sets, frame='wrist', title=title)
            ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    return fig


def pick_rows(dataset, num, split=None, seed=0):
    """ dataset: Dataset or DatasetDict; num rows spread over the dataset """
    if hasattr(dataset, 'keys'):
        split = split or next(name for name in ('train', 'validation', 'test') if name in dataset and len(dataset[name]))
        dataset = dataset[split]
    idx = np.linspace(0, len(dataset) - 1, min(num, len(dataset))).astype(int)
    return [dataset.with_format('numpy')[int(i)] for i in idx]


def save_preview(dataset, path, num=4, split=None):
    import matplotlib.pyplot as plt
    fig = sample_figure(pick_rows(dataset, num, split))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset', help='folder written by build_dataset.py or a HF repo id')
    p.add_argument('--split', default=None)
    p.add_argument('--num', type=int, default=4)
    p.add_argument('--out', default='preview.png')
    a = p.parse_args()
    from datasets import load_dataset, load_from_disk
    ds = load_from_disk(a.dataset) if os.path.isdir(a.dataset) else load_dataset(a.dataset)
    print(save_preview(ds, a.out, a.num, a.split))
