import torch
import pytorch3d as p3d
import numpy as np
from helper.transform import wrist_to_finger_center
from helper.visualize import Visualizer
import os




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


