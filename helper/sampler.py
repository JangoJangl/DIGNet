import torch
import numpy as np
import trimesh


class Sampler(object):

    @staticmethod
    def random_vertex_sample(obj, num_points=1024, use_normals=True, use_colors=False, device='cuda'):

        """
        Sample points from the vertices of the mesh
        :param obj: Object open3d mesh
        :param device: Device to store the points
        :param use_normals: Whether to use normals as features
        :param num_points: Number of points to sample
        :return: Torch tensor of shape (batch_size, 3 + len(features))
        """
        idcs = np.random.choice(len(obj.vertices), num_points, replace=False)
        pos = torch.tensor(obj.vertices, dtype=torch.float, device=device)[idcs]
        features = torch.empty((num_points, 0), dtype=torch.float, device=device)
        if use_colors:
            colors = torch.tensor(obj.visual.vertex_colors, dtype=torch.float, device=device)[idcs][:, 0:3] / 255
            features = torch.cat((features, colors), dim=1)
        if use_normals:
            normals = torch.tensor(obj.vertex_normals, dtype=torch.float, device=device)[idcs]
            features = torch.cat((features, normals), dim=1)

        return torch.cat((pos, features), dim=1)

    @staticmethod
    def uniform_sample(obj, num_points=1024, use_normals=True, device='cpu'):
        """
        Uniformly sample points from the mesh
        :param obj: Object open3d mesh
        :param device: Device to store the points
        :param use_normals: Whether to use normals as features
        :param num_points: Number of points to sample
        :return: Pointcloud
        """
        if use_normals:
            obj.compute_vertex_normals()
            obj.compute_triangle_normals()
        o3d_pc = obj.mesh_o3d.sample_points_uniformly(num_points)
        if use_normals:
            assert o3d_pc.has_normals()
            normals = torch.tensor(o3d_pc.normals, dtype=torch.float, device=device)
        else:
            normals = torch.empty((num_points, 0), dtype=torch.float, device=device)
        color = torch.tensor(o3d_pc.colors, dtype=torch.float, device=device)
        pos = torch.tensor(o3d_pc.points, dtype=torch.float, device=device)

        return torch.cat((pos, color, normals), dim=1)

    @staticmethod
    def poisson_disk_sample(obj, num_points=1024, use_normals=True, use_colors=False, device='cpu'):
        """
        Poisson disk sample points from the mesh, quasi equidistant
        :param obj: Object open3d mesh
        :param device: Device to store the points
        :param num_points: Number of points to sample
        :param use_normals: Whether to use normals as features
        :return: Torch tensor of shape (batch_size, 3 + len(features))
        """
        try:

            # samples, face_idcs = trimesh.sample.sample_surface_even(obj, num_points)
            if isinstance(obj, trimesh.Trimesh):
                o3d_obj = obj.to_o3d()
            samples = o3d_obj.sample_points_poisson_disk(number_of_points=num_points, init_factor=4)
            pos = torch.tensor(samples, dtype=torch.float, device=device)
            features = torch.empty((num_points, 0), dtype=torch.float, device=device)
            ##### use coolors and use_normals does not work becuase method switched from trimesh to open3d ####
            # if use_colors:
            #     colors = torch.tensor(obj.visual.face_colors[face_idcs][:, 0:3] / 255, dtype=torch.float, device=device)
            #     features = torch.cat((features, colors), dim=1)
            # if use_normals:
            #     normals = torch.tensor(obj.face_normals[face_idcs], dtype=torch.float, device=device)
            #     features = torch.cat((features, normals), dim=1)

            tensor = torch.cat((pos, features), dim=1)

        except:
            tensor = Sampler.random_vertex_sample(obj, num_points, use_normals, use_colors, device)

        return tensor


