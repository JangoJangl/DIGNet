import open3d as o3d
import trimesh
import torch
import copy
import numpy as np
from alr_sim.utils.geometric_transformation import quat2mat

class Visualizer(object):

    # Todo: Adding trimesh visualization after upgrading python from 3.7.16 to 3.8+
    # Todo: add Coloring
    # Todo: add instance query, Pointcloud, o3d_Mesh and other printables
    # Todo: Boundigboxes
    # Todo: installing alr_sim

    # works just with printable object for o3d
    @classmethod
    def o3d(cls, objs, scale=0.1, center=(0, 0, 0)):

        scene = [cls.__create_cos(scale, center)]
        for o in objs:
            if isinstance(o, trimesh.Trimesh):
                o = objs.as_open3d
                o.compute_vertex_normals()
            scene = scene + o

        o3d.visualization.draw_geometries(scene)

    @classmethod
    def o3d_tensor(cls, tensor, scale=0.1, center=(0, 0, 0)):
        scene = [cls.__create_cos(scale, center)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(torch.Tensor.cpu(tensor['pos']))
        pcd.colors = o3d.utility.Vector3dVector(torch.Tensor.cpu(tensor['colors']))
        scene.append(pcd)
        o3d.visualization.draw_geometries(scene)

        print('test')

    @classmethod
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
    def __create_cos(scale, center):
        return o3d.geometry.TriangleMesh.create_coordinate_frame().scale(scale, center=center)



