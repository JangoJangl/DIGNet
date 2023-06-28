import copy

import trimesh
import open3d as o3d
import numpy as np
from helper.visualize import create_gripper_mesh, create_finger_mesh
import torch


def collision_detection(grasps, obj):
    gripper_meshes = []
    collision_manager = trimesh.collision.CollisionManager()
    for i, g in enumerate(grasps):
        mesh = create_gripper_mesh(width=0.1)
        mesh = mesh.transform(g.to('cpu'))
        mesh = o3d_to_trimesh(mesh)
        gripper_meshes.append(mesh)
        collision_manager.add_object(str(i), mesh)

    col, idx_string =collision_manager.in_collision_single(obj, return_names=True)
    idx_collision = [int(i) for i in list(idx_string)]
    idx = [i for i in range(len(grasps)) if i not in idx_collision]



    return torch.IntTensor(idx), [gripper_meshes[i] for i in idx]

def collision_detection_pc_mesh(grasps, points, grasp_idcs=None):
    """
    calculate collision between gripper and pointcloud via signed distance function.
    TODO: acceleration with more abstract mesh

    :param grasp:
    :param pc:
    :return:
    """
    if grasp_idcs is None:
        grasp_idcs = list(range(len(grasps)))
    idx = []
    hand = create_gripper_mesh(width=0.1, simplified=True)
    for i in grasp_idcs:
        mesh = copy.deepcopy(hand)
        mesh = mesh.transform(grasps[i].to('cpu'))
        mesh = o3d_to_trimesh(mesh)
        sdf = torch.Tensor(trimesh.proximity.signed_distance(mesh, points))
        if sdf.max() > 0:
            idx.append(i)

    return torch.IntTensor(idx)

def not_in_gripper_detection_pc_mesh(grasps, points, grasp_idcs=None):
    """
    calculate collision between gripper and pointcloud via signed distance function.
    TODO: acceleration with more abstract mesh

    :param grasp:
    :param pc:
    :return:
    """
    if grasp_idcs is None:
        grasp_idcs = list(range(len(grasps)))
    gripper_meshes = []
    idx = []
    finger = create_finger_mesh(width=0.1)
    for i in grasp_idcs:
        mesh = copy.deepcopy(finger)
        mesh = mesh.transform(grasps[i].to('cpu'))
        mesh = o3d_to_trimesh(mesh)
        mesh = trimesh.convex.convex_hull(mesh) # convex hull
        sdf = torch.Tensor(trimesh.proximity.signed_distance(mesh, points))
        if sdf.max() <= 0: # in collision with object, means in gripper
            gripper_meshes.append(mesh)
            idx.append(i)

    return torch.IntTensor(idx)


def mesh_reconstruction():
    pass



def o3d_to_trimesh(mesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    face_normals = np.asarray(mesh.triangle_normals)
    vertices_normals = np.asarray(mesh.vertex_normals)
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertices_normals=vertices_normals, face_normals=face_normals)

def gripper_abstaction(gripper):
    """

    :param gripper:
    :return: simplified gripper mesh of boxes
    """
    pass


