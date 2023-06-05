from typing import Optional, Callable, Any
import logging

import os
from alr_sim.utils.sim_path import sim_framework_path
from alr_sim.utils.geometric_transformation import quat2mat
import numpy as np
import open3d as o3d
import trimesh
from acronym_tools import create_gripper_marker, load_mesh, load_grasps
import matplotlib.colors as mcolors
import copy

def main():
    root_directory = '/home/temp_store/jandl/Dataset/acronym'
    data_directory = '/home/temp_store/jandl/Dataset/acronym/mesh_contacts_graspnet'
    h5_directory = '/home/temp_store/jandl/Dataset/acronym/grasps'
    num_grasps = 10
    file_names = os.listdir(data_directory)
    for idx, file in enumerate(file_names):

        # join paths
        full_file_path_npz = os.path.join(data_directory, file)
        full_file_path_h5 = os.path.join(h5_directory, file.replace('.npz', '.h5'))

        # load object
        obj = load_mesh(full_file_path_h5, root_directory).as_open3d.compute_vertex_normals()

        # load grasps
        grasps_acronym = load_grasps(full_file_path_h5)
        grasps_npz = np.load(full_file_path_npz)
        grasps = {}
        for k in grasps_npz.files:      # unpack npz format
            grasps[k] = grasps_npz[k]

        # filter for successful grasps
        grasps_acronym = grasps_acronym[0][grasps_acronym[1].astype(bool), :]
        grasps = {k: grasps[k][grasps['successful'].astype(bool)] for k in grasps if k in grasps.keys()}

        # visualize
        obj_2 = copy.deepcopy(obj)
        visualize_grasp_graspnet(grasps, obj, num_grasps=num_grasps)
        visualize_grasp_acronym(grasps_acronym, obj_2, num_grasps=num_grasps)


def visualize_grasp_acronym(grasps, obj, num_grasps=10):
    """
    Visualizes grasps in Bbox COS

    :param grasps: Tuple (Transformation, successful)
    :param obj: TriangleMesh of object
    :param num_grasps: number of grasps to visualize
    :return:
    """
    objs = []
    # load object
    cos = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.1, center=(0, 0, 0))
    objs.append(obj)
    objs.append(cos)


    colors = create_color_list()
    vis = create_gripper_visualization_acronym(grasps, num_grasps, colors, gripper=True)
    objs.extend(vis)
    o3d.visualization.draw_geometries(objs, window_name='Acronym')


def visualize_grasp_graspnet(grasps, obj, num_grasps=10):
    """
        Visualizes grasps in center of gravity COS

        :param grasps: dict of grasps and properties
        :param obj: TriangleMesh of object
        :param num_grasps: number of grasps to visualize
        :return:
        """
    objs = []

    # load object
    obj_mean = np.mean(obj.vertices, 0)
    obj = obj.translate(-obj_mean) #center object COS
    cos = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.1, center=(0, 0, 0))
    objs.append(obj)
    objs.append(cos)


    # load contact points
    colors = create_color_list()
    vis = create_contact_gripper_visualization_graspnet(grasps, num_grasps, colors, gripper=True)
    objs.extend(vis)
    o3d.visualization.draw_geometries(objs, window_name='Graspnet')


def create_contact_gripper_visualization_graspnet(grasps, num_grasps=10, colors=[[0, 0, 255]], gripper=False):
    vis = []
    idcs = np.random.choice(len(grasps['contact_points']), num_grasps, replace=False)
    for i in idcs:
        color = colors[i % len(colors)]
        points = create_contact_point_mesh(grasps['contact_points'][i], color)
        vis.extend(points)
        if gripper:
            gripper = create_gripper_mesh(color)
        else:
            gripper = create_gripper_marker(color)
        gripper = gripper.transform(grasps['grasp_transform'][i])
        vis.append(gripper)
    return vis


def create_gripper_visualization_acronym(grasps, num_grasps=10, colors=[[0, 0, 1]], gripper: bool = False):
    vis = []
    idcs = np.random.choice(len(grasps), num_grasps, replace=False)
    for i in idcs:
        color = colors[i % len(colors)]
        if gripper:
            gripper = create_gripper_mesh(color)
        else:
            gripper = create_gripper_marker(color)
        gripper = create_gripper_marker(color)
        gripper = gripper.transform(grasps[i])
        vis.append(gripper)
    return vis

def create_contact_point_mesh(contact_points, color=[0, 0, 255]):
    s = []
    for point in contact_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001).translate(point).paint_uniform_color(color).compute_vertex_normals()
        s.append(sphere)
    return s


def create_gripper_marker(color=[0, 1, 1], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        o3d.Trianglemesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp = tmp.as_open3d.compute_vertex_normals().paint_uniform_color(color)

    return tmp


def create_gripper_mesh(color=[0, 0, 1], width=0.1):
    """

    :param color: color of gripper
    :param width: desired gripper width
    :return: open3d mesh of panda gripper
    """
    finger_path = sim_framework_path("models", "mj", "robot", "assets", "fingerv.stl")
    hand_path = sim_framework_path("models", "mj", "robot", "assets", "handv.stl")

    finger_mesh_left = o3d.io.read_triangle_mesh(finger_path)
    finger_mesh_right = o3d.io.read_triangle_mesh(finger_path)
    hand_mesh = o3d.io.read_triangle_mesh(hand_path)

    finger_mesh_right.rotate(quat2mat([0, 0, 0, 1]), center=[0, 0, 0])
    finger_mesh_right.translate([0, -width / 2, -0.055])
    finger_mesh_left.translate([0, width / 2, -0.055])
    hand_mesh.translate([0, 0, -0.1134])

    gripper_mesh = copy.deepcopy(hand_mesh)
    gripper_mesh += finger_mesh_right
    gripper_mesh += finger_mesh_left

    gripper_mesh.paint_uniform_color(color)
    gripper_mesh.compute_vertex_normals()

    gripper_mesh = transform_in_acronym_cos(gripper_mesh)

    return gripper_mesh


def create_color_list():
    """

    :return: list of colors [[r,g,b],...]
    """
    return [[int(c.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)] for c in mcolors.TABLEAU_COLORS.values()]


def transform_in_acronym_cos(mesh):
    """
    Transforms the mesh from the urdf-File COS coordinate system to the acronym coordinate system.
    :param mesh: open3d mesh
    :return: transformed open3d mesh
    """
    mesh.rotate(quat2mat([0.707, 0, 0, 0.707]), center=[0, 0, 0])
    mesh.translate([0, 0, 0.11217])
    return mesh

if __name__ == "__main__":
    main()
