import torch
import pytorch3d as p3d
import numpy as np
from helper.transform import wrist_to_finger_center
import os
from sklearn.neighbors import NearestNeighbors




def graspnet_loss(pred, target, obj):
    """
    Computes the graspnet loss

    Arguments:
        pred {np.array} -- predicted grasp 4x4 matrix
        target {np.array} -- target grasp 4x4 matrix

    Returns:
        [np.array] -- graspnet loss
    """

    # create transformation matrix of prediction and target
    # t_pred = p3d.transforms.Transform3d(pred)
    # t_target = p3d.transforms.Transform3d(target)

    #  load graspnet control points
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    control_points = torch.from_numpy(np.load(os.path.join(ROOT_DIR, 'gripper_control_points/panda.npy'))).double().cuda()

    w2f = torch.from_numpy(wrist_to_finger_center()).double().cuda()  # translate COS to finger center
    T_pred = torch.matmul(pred, w2f)
    T_target = torch.matmul(target, w2f)

    points_pred = torch.empty((pred.shape[0], 0, control_points.shape[1])).double().cuda()
    points_target = torch.empty((pred.shape[0], 0, control_points.shape[1])).double().cuda()


    #  transform graspnet control points with prediction/target transformation
    for c in control_points:
        points_pred = torch.cat((points_pred, torch.matmul(T_pred, c).unsqueeze(1)), dim=1)
        points_target = torch.cat((points_target, torch.matmul(T_target, c).unsqueeze(1)), dim=1)

    #  visualize control points for one grasp
    # Visualizer.visualize_cp_tar_pred(obj, target, pred, points_target, points_pred)

    #  compute loss with cumulative distance between transformed control points
    #  tensors have to be of shape (batch, control_points_size, 1, 4) last value is always 1, therefore it can be ignored
    loss = torch.cdist(points_pred.unsqueeze(-2), points_target.unsqueeze(-2), p=2).squeeze()

    # return mean over all control points distances
    return loss.sum(dim=-1)/control_points.shape[0]


def chamfer_loss(pred, target):
    """
    Computes the chamfer loss

    Arguments:
        pred {np.array} -- predicted grasp 4x4 matrix
        target {np.array} -- target grasp 4x4 matrix

    Returns:
        [np.array] -- chamfer loss
    """

    # trnasform grasp representations into 4x4 matrices

    # create transformation matrix of prediction and target
    t_pred = p3d.transforms.Transform3d(pred)
    t_target = p3d.transforms.Transform3d(target)

    #  load graspnet control points
    control_points = np.load('control_points.pt')
    #  transform graspnet control points with prediction transformation
    points_pred = t_pred.transform_points(control_points)
    #  transform graspnet control points with target transformation
    points_target = t_target.transform_points(control_points)
    #  compute loss with cumulative distance between transformed control points
    #  tensors have to be of shape (control_points_size, 1, 3)
    loss = p3d.loss.chamfer_distance(points_pred, points_target, p=2)

    return loss


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds

    https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367

    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist

