import numpy as np
import torch
try:
    import pytorch3d.transforms as p3d_t
except ImportError:  # only the 6D and quaternion representations need PyTorch3D
    p3d_t = None
from theseus.geometry import se3, so3


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


    @staticmethod
    def se3_log_map_representation(grasps, device='cuda'):
        """In the SE(3) logarithmic representation SE(3) matrices
        are represented as 6-dimensional vectors [log_translation | log_rotation],
        i.e. a concatenation of two 3D vectors log_translation and log_rotation."""
        se3_grasps = se3.SE3(dtype=torch.get_default_dtype())
        se3_grasps.update(grasps[:, :3, :]) # no need for the last row of the transformation matrix
        grasps = se3_grasps.log_map()
        inverse_transform = Representation.__se3_log_map_inverse_representation

        return grasps, inverse_transform
    @staticmethod
    def se3_pseudo_log_map_representation(grasps, device='cuda'):
        """In the SE(3) logarithmic representation SE(3) matrices
        are represented as 6-dimensional vectors [translation | log_rotation],
        i.e. a concatenation of two 3D vectors log_translation and log_rotation."""

        translation = grasps[:, :3, 3].to(device)
        rotation = grasps[:, :3, :3].to(device)

        so3_grasps = so3.SO3(dtype=torch.get_default_dtype())
        so3_grasps.update(rotation)
        grasps = torch.cat((translation, so3_grasps.log_map()), dim=1)
        inverse_transform = Representation.__so3_log_map_inverse_representation

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
        transform = torch.from_numpy(np.repeat(np.expand_dims(np.eye(4), axis=0), len(tensor_6d_rotation), axis=0)).to(
            device=tensor_6d_rotation.device, dtype=tensor_6d_rotation.dtype)
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
        transform = torch.repeat_interleave(torch.from_numpy(np.eye(4)).unsqueeze(0), len(tensor_trans_quat), dim=0).to(
            device=tensor_trans_quat.device, dtype=tensor_trans_quat.dtype)
        transform[:, :3, :3] = rotation
        transform[:, :3, 3] = translation
        return transform

    @staticmethod
    def __se3_log_map_inverse_representation(tensor):

        se3_tensor = se3.SE3.exp_map(tensor)
        return se3_tensor.to_matrix()

    @staticmethod
    def __so3_log_map_inverse_representation(tensor):
        translation = tensor[:, :3]
        so3_vec = tensor[:, 3:]
        rotation = so3.SO3.exp_map(so3_vec)
        transform = torch.repeat_interleave(torch.from_numpy(np.eye(4)).unsqueeze(0), len(tensor), dim=0).to(
            device=tensor.device, dtype=tensor.dtype)
        transform[:, :3, :3] = rotation.to_matrix()
        transform[:, :3, 3] = translation
        return transform

