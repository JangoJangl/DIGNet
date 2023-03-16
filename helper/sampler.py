import torch


class Sampler(object):

    @staticmethod
    def vertex_sample(obj, device='cpu'):

        """
        Sample points from the vertices of the mesh
        :param obj: Object
        :param device: Device to store the points
        :return: Pointcloud
        """

        pc = {'pos': torch.tensor(obj.mesh.vertices, dtype=torch.float, device=device),
              'colors': torch.tensor(obj.mesh.visual.vertex_colors, dtype=torch.float,
                                     device=device)[:, 0:3] / 255}
        return pc

