import torch
import numpy as np
import os


class Scaler(object):

    def __init__(self, device='cuda', root_directory='/home/temp_store/jandl/Dataset/acronym'):
        data_object = {}
        grasp_data = {}
        npz_object = np.load(os.path.join(root_directory, 'object_statistics.npz'))
        npz_grasp = np.load(os.path.join(root_directory, 'grasp_statistics.npz'))
        for k in npz_object.files:
            data_object[k] = npz_object[k]
        for k in npz_grasp.files:
            grasp_data[k] = npz_grasp[k]
        width = [0]
        self.representations = {
            'trans': [1, 1, 1],
            'euler_representation': [1, 1, 1, 2, 3, 4],
            'rotation_6d_representation': [1, 1, 1, -1, -1, -1, -1, -1, -1],
            'se3_log_map_representation': [9, 9, 9, 9, 9, 9],
            'se3_pseudo_log_map_representation': [1, 1, 1, 10, 10, 10], # is se3 logmap scaled right or do I need the vec norm?
            'trans_quat_representation':  [1, 1, 1, 8, 8, 8, 8]
        }



        self.grasp_mean = torch.from_numpy(grasp_data["mean"]).to(device)
        self.grasp_var = torch.from_numpy(grasp_data["var"]).to(device)
        self.grasp_min = torch.from_numpy(grasp_data["min"]).to(device)
        self.grasp_max = torch.from_numpy(grasp_data["max"]).to(device)

        self.grasp_var[-1] = torch.tensor(1.0)
        self.grasp_var[8] = torch.tensor(1.0)

    def min_max_scale(self, obj, grasps, representation='trans_quat_representation'):
        """ Scales the object and the grasps by min and max values
        :param obj: Torch.tensor
        :param grasps: Torch.tensor
        """
        pass

    def mean_var_scale(self, obj, grasps, representation='trans_quat_representation'):
        """ Scales the object and the grasps by mean and variance
        :param obj: Torch.tensor
        :param grasps: Torch.tensor
        """
        grasps = grasps - self.grasp_mean(self.representations[representation]) / torch.sqrt(self.grasp_var(self.representations[representation]))
        obj = obj - self.grasp_mean(self.representations["trans"]) / torch.sqrt(self.grasp_var(self.representations["trans"]))

        def inverse(grasps):
            return grasps * torch.sqrt(self.grasp_var(self.representations[representation])) + self.grasp_mean(self.representations[representation])

        return obj, grasps, inverse

    def var_scale(self, obj, grasps, representation='trans_quat_representation'):
        """ Scales the object and the grasps by mean and variance
        :param obj: Torch.tensor
        :param grasps: Torch.tensor
        """
        grasps = grasps / torch.sqrt(self.grasp_var[self.representations[representation]])
        obj = obj / torch.sqrt(self.grasp_var[self.representations["trans"]])

        def inverse(g):
            return g * torch.sqrt(self.grasp_var[self.representations[representation]])

        return obj, grasps, inverse


    def no_scale(self, obj, grasps, representation='trans_quat_representation'):
        """ Scales the object and the grasps by mean and variance
        :param obj: Torch.tensor
        :param grasps: Torch.tensor
        """
        def inverse(g):
            return g

        return obj, grasps, inverse


def min_max_norm(tensor):
    t = torch.linalg.vector_norm(tensor[:, :3], dim=1)
    return torch.min(t), torch.max(t)



if __name__ == "__main__":
    # root_directory = "/home/temp_store/jandl/Dataset/acronym"
    # dataset_directory = "/home/temp_store/jandl/Dataset/acronym/mesh_contacts_graspnet"
    # h5_directory = "/home/temp_store/jandl/Dataset/acronym/grasps"



    scaler = Scaler().mean_var_scale
    scaler(torch.rand(1, 3), torch.rand(1, 7))
    print("end")
    # data_object = np.load(os.path.join(root_directory, 'object_statistics.npz'))
    # data_grasp = np.load(os.path.join(root_directory, 'grasp_statistics.npz'))
    # print(data_object)
    # print(data_grasp)
