import open3d as o3d
import trimesh
import torch


class Visualizer(object):

    # Todo: Adding trimesh visualization after upgrading python from 3.7.16 to 3.8+
    # Todo: add Coloring
    # Todo: add instance query, Pointcloud, o3d_Mesh and other printables
    # Todo: Boundigboxes

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


    @staticmethod
    def __create_cos(scale, center):
        return o3d.geometry.TriangleMesh.create_coordinate_frame().scale(scale, center=center)



