"""
Headless visualisation of point clouds and grasps (matplotlib and wandb), usable during training on a
cluster or in a notebook. helper/visualize.py needs an Open3D window and is meant for local debugging.
"""
import numpy as np
import torch

# Panda gripper marker (same geometry as Visualizer.create_gripper_marker) in the ACRONYM wrist frame
_MARKER = np.array([
    [[0, 0, 0], [0, 0, 0.066]],                      # stem
    [[-0.041, 0, 0.066], [0.041, 0, 0.066]],         # cross bar
    [[-0.041, 0, 0.066], [-0.041, 0, 0.11217]],      # left finger
    [[0.041, 0, 0.066], [0.041, 0, 0.11217]],        # right finger
])
FINGER_CENTER_OFFSET = np.array([0, 0, 0.11217])  # grasps coming out of the processor have their origin here

COLORS = {'gt': (0.2, 0.4, 1.0), 'pred': (0.1, 0.75, 0.2), 'bad': (0.9, 0.15, 0.1), 'points': (0.35, 0.35, 0.35)}


def _numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def gripper_segments(poses, frame='finger_center'):
    """ [K, 4, 4] grasp poses -> [K, 4, 2, 3] line segments of the gripper marker, frame: 'finger_center' or 'wrist' """
    poses = _numpy(poses).reshape(-1, 4, 4)
    marker = _MARKER - FINGER_CENTER_OFFSET if frame == 'finger_center' else _MARKER
    return np.einsum('kij,sej->ksei', poses[:, :3, :3], marker) + poses[:, None, None, :3, 3]


def plot_grasps(ax, points, grasp_sets, point_colors=None, point_size=1.0, title=None, frame='finger_center'):
    """
    Draws a point cloud and any number of grasp sets into a 3D matplotlib axis.

    :param points: [N, >=3] point cloud
    :param grasp_sets: list of (poses [K, 4, 4], rgb color) tuples
    :param point_colors: optional [N, 3] colors, defaults to grey
    :param frame: grasp origin, 'finger_center' (processor output) or 'wrist' (raw ACRONYM poses)
    """
    points = _numpy(points)[:, :3]
    if point_colors is None:
        ax.scatter(*points.T, s=point_size, color=COLORS['points'], depthshade=False)
    else:
        ax.scatter(*points.T, s=point_size, c=point_colors, depthshade=False)
    extent = [points]
    for poses, color in grasp_sets:
        segments = gripper_segments(poses, frame)
        for seg in segments.reshape(-1, 2, 3):
            ax.plot(*seg.T, color=color, linewidth=0.8)
        extent.append(segments.reshape(-1, 3))
    extent = np.concatenate(extent)
    center, radius = extent.mean(axis=0), np.abs(extent - extent.mean(axis=0)).max()
    for axis, c in zip('xyz', center):
        getattr(ax, f'set_{axis}lim')(c - radius, c + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=8)


def grasp_figure(samples, ncols=4, max_grasps=50, views=((25, 45),)):
    """
    One subplot per sample and view.

    :param samples: list of dicts with 'points', optional 'gt' and 'pred' poses [K, 4, 4], optional 'title'
    :return: matplotlib figure
    """
    import matplotlib.pyplot as plt  # falls back to Agg without a display

    n = len(samples) * len(views)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
    rng = np.random.default_rng(0)
    for i, sample in enumerate(samples):
        sets = []
        for key in ('gt', 'pred'):
            if sample.get(key) is not None and len(sample[key]):
                poses = _numpy(sample[key])
                if len(poses) > max_grasps:
                    poses = poses[rng.choice(len(poses), max_grasps, replace=False)]
                sets.append((poses, COLORS[key]))
        for v, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(nrows, ncols, i * len(views) + v + 1, projection='3d')
            plot_grasps(ax, sample['points'], sets, point_colors=sample.get('point_colors'), title=sample.get('title'))
            ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    return fig


def wandb_object3d(points, grasp_sets, points_per_segment=20):
    """
    Interactive 3D view for wandb: points in grey, grippers as densely sampled coloured line points.

    :param grasp_sets: list of (poses [K, 4, 4], rgb color in [0, 1]) tuples
    """
    import wandb
    clouds = [np.concatenate([_numpy(points)[:, :3], np.tile(np.array(COLORS['points']) * 255, (len(points), 1))], 1)]
    s = np.linspace(0, 1, points_per_segment)[:, None]
    for poses, color in grasp_sets:
        segments = gripper_segments(poses).reshape(-1, 2, 3)
        line_points = (segments[:, None, 0] * (1 - s) + segments[:, None, 1] * s).reshape(-1, 3)
        clouds.append(np.concatenate([line_points, np.tile(np.array(color) * 255, (len(line_points), 1))], 1))
    return wandb.Object3D(np.concatenate(clouds).astype(np.float32))
