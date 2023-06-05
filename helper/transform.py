import numpy as np
import torch
import trimesh


# Todo: transform into Boundingbox coord (think about the inference pipeline boundingbox)

class Transform(object):

    @staticmethod
    def start_transform(obj, grasps):
        obj_mean = np.mean(obj.vertices, 0)
        t_mean = np.eye(4)
        t_mean[:3, 3] = -obj_mean
        obj = obj.apply_transform(t_mean)

        f2w = finger_center_to_wrist()  # translate COS to finger center
        grasps = torch.matmul(grasps, torch.from_numpy(f2w).float().cuda())

        return obj, grasps


    @staticmethod
    def center_transform(obj, grasps):
        if isinstance(obj, trimesh.Trimesh):
            transform, _ = trimesh.bounds.oriented_bounds(obj)
            obj.apply_transform(transform)
        else:
            bbox = obj.get_oriented_bounding_box()
            obj_mean = np.mean(obj.vertices, 0)
            transform = np.eye(4)
            transform[:3, :3] = bbox.R
            transform[:3, 3] = obj_mean
            obj.transform(transform)

        grasps = torch.matmul(grasps, torch.from_numpy(transform).float().cuda())

        return obj, grasps, inverse_transform(transform)

    @staticmethod
    def bbox_transform(obj, grasps):
        # Todo: add trimesh support
        bbox = obj.get_oriented_bbox()
        transform = np.eye(4)
        transform[:3, :3] = bbox.R
        transform[:3, 3] = obj.get_min_bound()
        obj.transform(transform)
        grasps = torch.matmul(grasps, torch.from_numpy(transform).float().cuda())
        return obj, grasps, inverse_transform(transform)


def inverse_transform(trans):

    """
    Computes the inverse of 4x4 transform.

    Arguments:
        trans {np.ndarray} -- 4x4 transform.

    Returns:
        [np.ndarray] -- inverse 4x4 transform
    """
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t

    return output

def wrist_to_finger_center():
    transform = np.eye(4)
    transform[:3, 3] = np.array([0, 0, -.11217])
    return transform

def finger_center_to_wrist():
    transform = np.eye(4)
    transform[:3, 3] = np.array([0, 0, .11217])
    return transform