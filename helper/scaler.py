import torch
import numpy as np


class Scaler(object):

    @staticmethod
    def ball_scale(obj, grasps, scale_factor, use_colors=False):
        """ Scales the object and the grasps by the same factor (rotations and normals are already scaled)
        :param obj: Object pointcloud
        :param grasps: Grasps
        :param scale_factor: Scale factor
        :param use_colors: Whether the object has colors"""
        obj[:, :3] = obj[:, :3] / scale_factor
        if use_colors:
            obj[:, 3:6] = obj[:, 3:6] / 255
        grasps[:, :3] = grasps[:, :3] / scale_factor
        return obj, grasps

    @staticmethod
    def xyz_scale(obj, grasps, scale_vector, use_colors=False):
        """ Scales the object and the grasps by the same factor (rotations and normals are already scaled)
        :param obj: Torch.tensor
        :param grasps: Torch.tensor
        :param scale_vector: [x_scale, y_scale, z_scale]
        :param use_colors: Whether the object has colors"""
        obj[:, :3] = obj[:, :3] / scale_vector
        if use_colors:
            obj[:, 3:6] = obj[:, 3:6] / 255
        grasps[:, :3] = grasps[:, :3] / scale_vector
        return obj, grasps


def min_max_norm(tensor):
    t = torch.linalg.vector_norm(tensor[:, :3], dim=1)
    return torch.min(t), torch.max(t)
