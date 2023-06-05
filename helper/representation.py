import numpy as np
import torch
import pytorch3d.transforms as p3d_t


class Representation(object):

    @staticmethod
    def rotation_6d_representation(grasps, device='cuda'):
        """ 6D representation of rotation grasp transformation used by GraspNet"""
        translation = grasps[:, :3, 3].to(device)
        rotation = grasps[:, :3, :3].to(device)
        rot_6d = p3d_t.matrix_to_rotation_6d(rotation)
        grasps = torch.cat([translation, rot_6d], dim=1)
        inverse_transform = Representation.__rotation_6d_inverse_transform

        return grasps, inverse_transform


    # Todo: logmap representation von pytorch3d kopieren oder pytorch3d updaten
    @staticmethod
    def se3_log_map_representation(grasps, device='cuda'):
        """In the SE(3) logarithmic representation SE(3) matrices
        are represented as 6-dimensional vectors [log_translation | log_rotation],
        i.e. a concatenation of two 3D vectors log_translation and log_rotation."""

        grasps = p3d_t.se3_log_map(grasps)
        inverse_transform = p3d_t.se3_exp_map

        return grasps, inverse_transform



    @staticmethod
    def se3_representation(grasps, device='cuda'):
        """ SE3 representation of grasp transformation """
        grasps = torch.flatten(grasps, start_dim=1)
        inverse_transform = Representation.__se3_inverse_transform
        return grasps, inverse_transform

    @staticmethod
    def trans_quat_representation(grasps, device='cuda'):
        """ Translation and quaternion representation of grasp transformation """
        translation = grasps[:, :3, 3].to(device)
        rotation = grasps[:, :3, :3].to(device)
        quat = p3d_t.matrix_to_quaternion(rotation)
        grasps = torch.cat([translation, quat], dim=1)
        inverse_transform = Representation.__trans_quat_inverse_transform
        return grasps, inverse_transform

    @staticmethod
    def __rotation_6d_inverse_transform(tensor_6d_rotation):
        translation = tensor_6d_rotation[:, :3]
        rot_6d = tensor_6d_rotation[:, 3:]
        rotation = p3d_t.rotation_6d_to_matrix(rot_6d)
        transform = torch.from_numpy(np.repeat(np.expand_dims(np.eye(4), axis=0), len(tensor_6d_rotation), axis=0)).to('cuda')
        transform[:, :3, :3] = rotation
        transform[:, :3, 3] = translation
        return transform

    @staticmethod
    def __se3_inverse_transform(tensor_se3):
        return tensor_se3.view(-1, 4, 4)

    @staticmethod
    def __trans_quat_inverse_transform(tensor_trans_quat):
        translation = tensor_trans_quat[:, :3]
        quat = tensor_trans_quat[:, 3:]
        rotation = p3d_t.quaternion_to_matrix(quat)
        transform = torch.repeat_interleave(torch.from_numpy(np.eye(4)).unsqueeze(0), len(tensor_trans_quat), dim=0).to('cuda')
        transform[:, :3, :3] = rotation
        transform[:, :3, 3] = translation
        return transform


