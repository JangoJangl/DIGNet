import torch


class Sampler(object):

    @staticmethod
    def vertex_sample(obj, device='cpu'):

        """
        Sample points from the vertices of the mesh
        :param obj: Object
        :param device: Device to store the points
        :return: Pointcloud
        Todo: add num_points parameter with different sampling methods
        """

        pc = {'pos': torch.tensor(obj.mesh.vertices, dtype=torch.float, device=device),
              'colors': torch.tensor(obj.mesh.visual.vertex_colors, dtype=torch.float,
                                     device=device)[:, 0:3] / 255}
        return pc

    @staticmethod
    def uniform_sample(obj, device='cpu', num_points=1024):
        """
        Uniformly sample points from the mesh
        :param obj: Object
        :param device: Device to store the points
        :param num_points: Number of points to sample
        :return: Pointcloud
        """

        o3d_pc = obj.mesh_o3d.sample_points_uniformly(num_points)

        pc = {'pos': torch.tensor(o3d_pc.points, dtype=torch.float, device=device),
              'colors': None}
        return pc

    @staticmethod
    def poisson_disk_sample(obj, device='cpu', num_points=1024):
        """
        Poisson disk sample points from the mesh, quasi equidistant
        :param obj: Object
        :param device: Device to store the points
        :param num_points: Number of points to sample
        :return: Pointcloud
        """

        pcd = obj.mesh_o3d.sample_points_uniformly(number_of_points=8000)
        o3d_pc = obj.mesh_o3d.sample_points_poisson_disk(number_of_points=num_points, pcl=pcd)

        pc = {'pos': torch.tensor(o3d_pc.points, dtype=torch.float, device=device),
              'colors': None}
        return pc

