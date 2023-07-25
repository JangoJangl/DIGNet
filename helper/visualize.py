import open3d as o3d
import trimesh
import torch
import copy
import numpy as np
from helper.transform import wrist_to_finger_center as w2f
from alr_sim.utils.sim_path import sim_framework_path
from alr_sim.utils.geometric_transformation import quat2mat
from collections.abc import Iterable


class Visualizer(object):

    # Todo: Adding trimesh visualization after upgrading python from 3.7.16 to 3.8+
    # Todo: add Coloring
    # Todo: Boundigboxes

    # works just with printable object for o3d
    @staticmethod
    def visualize(objs, grasps, inverse=lambda x: x, scale=0.1, center=(0, 0, 0), colors=[], use_gripper_mesh=False, bad_idx=[]):

        if isinstance(objs, Iterable) is False or isinstance(objs, torch.Tensor) is True:
            objs = [objs]
        scene = [Visualizer.create_cos(scale, center)]
        for o in objs:
            if isinstance(o, trimesh.Trimesh):
                obj = o.as_open3d
                obj.compute_vertex_normals()
                obj.paint_uniform_color([0.5, 0.5, 0.5])
            elif isinstance(o, torch.Tensor):
                obj = o3d.geometry.PointCloud()
                obj.points = o3d.utility.Vector3dVector(torch.Tensor.cpu(o[:, :3]))
                obj.paint_uniform_color([0, 0, 0])
                obj.normals = o3d.utility.Vector3dVector(torch.Tensor.cpu(o[:, -3:]))
            elif isinstance(o, o3d.geometry.Geometry3D):
                obj = o
            else:
                raise NotImplementedError

            scene = scene + [obj]

        for i, g in enumerate(grasps):
            g = inverse(g.to('cpu'))  # transform grasp to homogenous matrix
            # color = colors[i % len(colors)]
            if len(colors) and len(grasps) == len(colors):
                color = colors[i]

            elif i in bad_idx:
                color = [1, 0, 0]
            else:
                color = [0, 1, 0]
            if use_gripper_mesh is True:
                gripper = create_gripper_mesh(color)
            else:
                gripper = Visualizer.create_gripper_marker(color)
            gripper = gripper.transform(g.detach()) if g.requires_grad else gripper.transform(g)
            scene.append(gripper)

        o3d.visualization.draw_geometries(scene)


    @staticmethod
    def render_images(objs, grasps, inverse=lambda x: x, scale=0.1, center=(0, 0, 0), colors=[], use_gripper_mesh=False,
                  bad_idx=[], image_name="image"):

        if isinstance(objs, Iterable) is False or isinstance(objs, torch.Tensor) is True:
            objs = [objs]
        scene = Visualizer.create_cos(scale, center)
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1000, height=1000, visible=True)
        vis.add_geometry(scene)
        vis.update_geometry(scene)
        for o in objs:
            if isinstance(o, trimesh.Trimesh):
                obj = o.as_open3d
                obj.compute_vertex_normals()
                obj.paint_uniform_color([0.5, 0.5, 0.5])
            elif isinstance(o, torch.Tensor):
                obj = o3d.geometry.PointCloud()
                obj.points = o3d.utility.Vector3dVector(torch.Tensor.cpu(o[:, :3]))
                obj.paint_uniform_color([0, 0, 0])
                obj.normals = o3d.utility.Vector3dVector(torch.Tensor.cpu(o[:, -3:]))
            elif isinstance(o, o3d.geometry.Geometry3D):
                obj = o
            else:
                raise NotImplementedError

            vis.add_geometry(obj)
            vis.update_geometry(obj)

        for i, g in enumerate(grasps):
            g = inverse(g.to('cpu'))  # transform grasp to homogenous matrix
            # color = colors[i % len(colors)]
            if len(colors) and len(grasps) == len(colors):
                color = colors[i]

            elif i in bad_idx:
                color = [1, 0, 0]
            else:
                color = [0, 1, 0]
            if use_gripper_mesh is True:
                gripper = create_gripper_mesh(color)
            else:
                gripper = Visualizer.create_gripper_marker(color)
            gripper = gripper.transform(g.detach()) if g.requires_grad else gripper.transform(g)
            vis.add_geometry(gripper)
            vis.update_geometry(gripper)


        opt = vis.get_render_option()
        opt.background_color = np.asarray([25.9 / 255, 57.6 / 255, 51.0 / 255])
        opt.point_size = 9.0

        zoom = 0.5
        ctr = vis.get_view_control()
        ctr.set_front(np.asarray([15, 0, 0.3]))
        ctr.set_lookat(np.asarray([15, 0, 0.3]))
        ctr.set_up(np.asarray([15, 0, 0.3]))
        ctr.set_zoom(zoom)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"/home/i53/student/jandl/repos/DIGNet/diffusion/images/{image_name}.png", False)
        vis.destroy_window()



    @staticmethod
    def render_pointcloud_with_panda_grippers(points, colors, grasp_pos, grasp_quat, width=0.01, vis=False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create the hand
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

        # Move the hand to the proper position
        gripper_mesh.rotate(quat2mat(grasp_quat), center=[0, 0, 0])
        gripper_mesh.translate(grasp_pos)
        gripper_mesh.paint_uniform_color([0.8, 0.149, 0.149])

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        mesh_frame_goal = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=grasp_pos)
        mesh_frame_goal.rotate(mesh_frame_goal.get_rotation_matrix_from_quaternion(grasp_quat))

        if vis:
            o3d.visualization.draw_geometries([pcd, mesh_frame, gripper_mesh])

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1000, height=1000, visible=False)
        vis.add_geometry(pcd)
        vis.add_geometry(mesh_frame)
        vis.add_geometry(mesh_frame_goal)
        vis.add_geometry(gripper_mesh)
        vis.update_geometry(pcd)
        vis.update_geometry(mesh_frame)
        vis.update_geometry(mesh_frame_goal)
        vis.update_geometry(gripper_mesh)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([25.9 / 255, 57.6 / 255, 51.0 / 255])
        opt.point_size = 9.0

        zoom = 0.2
        ctr = vis.get_view_control()
        ctr.set_front([1, 0, 0.3])
        ctr.set_lookat([0.4, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(zoom)
        vis.poll_events()
        vis.update_renderer()
        img_mid = vis.capture_screen_float_buffer(True)
        ctr.set_front([0, -1, 0.3])
        ctr.set_lookat([0.4, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(zoom)
        vis.poll_events()
        vis.update_renderer()
        img_left = vis.capture_screen_float_buffer(True)
        ctr.set_front([0, 1, 0.3])
        ctr.set_lookat([0.4, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(zoom)
        vis.poll_events()
        vis.update_renderer()
        img_right = vis.capture_screen_float_buffer(True)
        vis.destroy_window()

        return img_mid, img_left, img_right

    @staticmethod
    def create_cos(scale=0.1, center=[0, 0, 0]):
        return o3d.geometry.TriangleMesh.create_coordinate_frame().scale(scale, center=center)

    @staticmethod
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

        tmp.transform(w2f())

        return tmp

    @staticmethod
    def visualize_cp_tar_pred(obj, target, pred, points_target, points_pred):

        scene = [Visualizer.create_cos(0.1, (0, 0, 0))]

        gripper_t = Visualizer.create_gripper_marker()
        gripper_t.transform(target[1].cpu())

        gripper_p = Visualizer.create_gripper_marker()
        gripper_p.transform(pred[1].cpu())

        cp_t = o3d.geometry.PointCloud()
        cp_t.points = o3d.utility.Vector3dVector(torch.Tensor.cpu(points_target[1, :, :3].cpu()))
        cp_t.paint_uniform_color([0, 1, 0])

        cp_p = o3d.geometry.PointCloud()
        cp_p.points = o3d.utility.Vector3dVector(torch.Tensor.cpu(points_pred[1, :, :3].cpu()))
        cp_p.paint_uniform_color([0, 0, 1])

        o = o3d.geometry.PointCloud()
        o.points = o3d.utility.Vector3dVector(torch.Tensor.cpu(obj[0, :, :3].cpu()))
        o.paint_uniform_color([0, 0, 0])

        scene = scene + [gripper_t, gripper_p, cp_t, cp_p, o]

        o3d.visualization.draw_geometries(scene)




def transform_in_acronym_cos(mesh):
    """
    Transforms the mesh from the urdf-File COS coordinate system to the acronym coordinate system.
    :param mesh: open3d mesh
    :return: transformed open3d mesh
    """
    mesh.rotate(quat2mat([0.707, 0, 0, 0.707]), center=[0, 0, 0])
    mesh.translate([0, 0, 0.11217])
    return mesh

def create_gripper_mesh(color=[0, 0, 1], width=0.1, simplified=False):
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

    if simplified:
        finger_mesh_left = finger_mesh_left.simplify_vertex_clustering(0.005)
        finger_mesh_right = finger_mesh_right.simplify_vertex_clustering(0.005)
        hand_mesh, _ = hand_mesh.compute_convex_hull()

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

    gripper_mesh.transform(w2f())

    return gripper_mesh


def create_finger_mesh(color=[0, 0, 1], width=0.1):
    """

    :param color: color of gripper
    :param width: desired gripper width
    :return: open3d mesh of panda gripper
    """
    finger_path = sim_framework_path("models", "mj", "robot", "assets", "fingerv.stl")

    finger_mesh_left = o3d.io.read_triangle_mesh(finger_path)
    finger_mesh_right = o3d.io.read_triangle_mesh(finger_path)

    finger_mesh_right.rotate(quat2mat([0, 0, 0, 1]), center=[0, 0, 0])
    finger_mesh_right.translate([0, -width / 2, -0.055])
    finger_mesh_left.translate([0, width / 2, -0.055])

    gripper_mesh = copy.deepcopy(finger_mesh_right)
    gripper_mesh += finger_mesh_left

    gripper_mesh.paint_uniform_color(color)
    gripper_mesh.compute_vertex_normals()

    gripper_mesh = transform_in_acronym_cos(gripper_mesh)

    gripper_mesh.transform(w2f())

    return gripper_mesh