from typing import Optional, Callable, Any
import logging

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
from acronym_tools import load_mesh
import h5py
from omegaconf import DictConfig
import hydra
from pytorch3d.transforms import matrix_to_euler_angles, matrix_to_quaternion
from theseus.geometry import se3, so3
from helper.transform import Transform


# TODO: create Scene Dataset
# TODO: Transform einbauen????
# TODO: instantiate dataset with hydra
# TODO: make Acronym Dataset usable


def get_train_val(
    root_directory,
    data_directory,
    h5_directory,
    subbatch_size,
    train_fraction=0.9,
    device="cpu",
    processor: DictConfig = None,
):
    print("create datasets")
    files = os.listdir(data_directory)
    train_files, test_files = random_split(files, [train_fraction, 1 - train_fraction], generator=torch.Generator().manual_seed(42))

    trainset = GraspnetAcronymDataset(
        root_directory,
        data_directory,
        h5_directory,
        list(train_files),
        subbatch_size,
        device,
        train_fraction,
        processor)

    testset = GraspnetAcronymDataset(
        root_directory,
        data_directory,
        h5_directory,
        list(test_files),
        subbatch_size,
        device,
        train_fraction,
        processor,
        train=False)

    return trainset, testset

class GraspnetAcronymDataset_old(Dataset):
    ''' Preprocessed grasps by contact graspnet team. Every grasp is collision proved with panda gripper'''
    def __init__(
            self,
            root_directory: os.PathLike,
            dataset_directory: os.PathLike,
            h5_directory: os.PathLike,
            subbatch_size: int,
            device="cpu",
            train_split: float = 0.9,
            processor: Callable = lambda x: x, # function to process the data
    ):
        self.device = device
        self.root_directory = Path(root_directory)
        self.data_directory = Path(dataset_directory)
        self.h5_directory = Path(h5_directory)
        self.file_names = os.listdir(self.data_directory)
        self.train_split = train_split
        self.sample_size = subbatch_size
        self.processor = processor

        self.object_ids, self.grasp_transforms, self.contact_points = self.__load_data()
        # data is saved in a dict where each key is the path to a object embedding and each value is a tensor of grasps
        # with dimensions [num_samples, sample_size, Transformation Matrix  = 4x4]
        #  a sample consists of sample_size grasps

        _, self.num_objects = torch.unique(self.object_ids, return_counts=True)
        self.num_samples = self.grasp_transforms.size(dim=0)
        self.samples_per_object = torch.bincount(self.object_ids)
        self.buckets = torch.cumsum(self.samples_per_object, dim=0)

    # return sample given a index
    def __getitem__(self, idx):
        # TODO: add internal Grasp
        self.idx = idx
        path = os.path.join(self.h5_directory, self.file_names[self.object_ids[idx]].replace('.npz', '.h5'))
        trimesh = load_mesh(path, self.root_directory)
        obj = trimesh
        return self.processor(obj, self.grasp_transforms[idx])  #, self.contact_points[idx]

    def get_current(self):
        idx = self.idx
        path = os.path.join(self.h5_directory, self.file_names[self.object_ids[idx]].replace('.npz', '.h5'))
        trimesh = load_mesh(path, self.root_directory)
        obj = trimesh
        return self.processor(obj, self.grasp_transforms[idx])  #, self.contact_points[idx]

    def __len__(self):
        return self.num_samples

    def get_post_processing(self):
        return self.processor.post_processing

    def get_inverse_representation(self):
        return self.processor.inverse_representation_function

    def __load_data(self):
        grasp_transforms = torch.empty((0, self.sample_size, 4, 4), dtype=torch.get_default_dtype(), device=self.device)
        grasp_contact_points = torch.empty((0, self.sample_size, 2, 3), dtype=torch.get_default_dtype(), device=self.device)
        obj_idcs = torch.empty((0,), dtype=torch.int, device=self.device) # object indices for each sample
        for idx, file in enumerate(self.file_names):

            grasps_npz = np.load(os.path.join(self.data_directory, file))
            grasps = {}
            for k in grasps_npz.files:
                grasps[k] = grasps_npz[k]
            grasps = {k: grasps[k][grasps['successful'].astype(bool)] for k in grasps if k in grasps.keys()}
            if grasps['grasp_transform'].shape[0] > self.sample_size:
                num_kick = grasps['grasp_transform'].shape[0] % self.sample_size
                kick_indices = np.random.choice(grasps['grasp_transform'].shape[0], num_kick, replace=False)
                grasps = {k: np.delete(grasps[k], kick_indices, axis=0) for k in grasps if k in grasps.keys()}
                grasps = {k: grasps[k].reshape((-1, self.sample_size, *grasps[k].shape[1:])) for k in grasps.keys()}

                grasp_transforms = torch.vstack((grasp_transforms, torch.from_numpy(grasps['grasp_transform']).to(self.device)))
                grasp_contact_points = torch.vstack((grasp_contact_points, torch.from_numpy(grasps['contact_points']).to(self.device)))
                obj_idcs = torch.cat((obj_idcs, torch.ones(grasps['grasp_transform'].shape[0], dtype=torch.int).to(self.device) * int(idx)), dim=0)
                if idx == 100:
                    break

        return obj_idcs, grasp_transforms, grasp_contact_points

    def get_screwdriver(self):
        file = 'ScrewDriver_28d88f47398c462977d2282deeabd2a4_0.014402062869257133.npz'
        path = os.path.join(self.h5_directory, file.replace('.npz', '.h5'))
        trimesh = load_mesh(path, self.root_directory)
        obj = trimesh

        grasps_npz = np.load(os.path.join(self.data_directory, file))
        grasps = {}
        for k in grasps_npz.files:
            grasps[k] = grasps_npz[k]
        # grasps = {k: grasps[k][grasps['successful'].astype(bool)] for k in grasps if k in grasps.keys()}
        good_grasps = torch.tensor(grasps['grasp_transform'][grasps['successful'].astype(bool)], device=self.device, dtype=torch.get_default_dtype())
        bad_grasps = torch.tensor(grasps['grasp_transform'][~grasps['successful'].astype(bool)], device=self.device, dtype=torch.get_default_dtype())

        return self.processor(obj, good_grasps), bad_grasps

    # iterator to generate a shuffled train/val split
    def get_train_val_split_iterator(self, train_share: float = 0.9, shuffle: bool = True):
        train_size = int(train_share * len(self))
        val_size = len(self) - train_size
        indices = np.arrange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield self[i][:train_size], self[i][train_size:]

    @torch.no_grad()
    def get_statistics(self):
        """
        returns statistics of the dataset in a dictionary
        :return:
        """
        obj_data = torch.empty((0, 6), dtype=torch.get_default_dtype(), device=self.device)
        grasp_data = torch.empty((0, 18), dtype=torch.get_default_dtype(), device=self.device)
        print("start object analysis")
        for file in self.file_names:
            try:
                o_data, g_data = self.sample_properties(file)
                obj_data = torch.cat((obj_data, o_data), dim=0)
                grasp_data = torch.cat((grasp_data, g_data), dim=0)
                print('file done: ', file)
            except:
                print('error in file: ', file)

        object_statistics, grasp_statistics = self.evaluate_statistics(obj_data, grasp_data)
        self.save_statistics(object_statistics, grasp_statistics)
        print('statistics saved')

        return {'object_statistics': object_statistics, 'grasp_statistics': grasp_statistics}


    @torch.no_grad()
    def sample_properties(self, file):
        object_data = torch.empty((0, 6), dtype=torch.get_default_dtype(), device=self.device)
        grasp_data = torch.empty((0, 18), dtype=torch.get_default_dtype(), device=self.device)
        path = os.path.join(self.h5_directory, file.replace('.npz', '.h5'))
        trimesh = load_mesh(path, self.root_directory)
        grasps_npz = np.load(os.path.join(self.data_directory, file))
        grasps = {}
        for k in grasps_npz.files:
            grasps[k] = grasps_npz[k]


        good_grasps = {k: grasps[k][grasps['successful'].astype(bool)] for k in grasps if k in grasps.keys()}
        num_good_grasps = torch.tensor(len(good_grasps['grasp_transform'])).to(self.device)
        num_bad_grasps = torch.tensor(len(grasps['grasp_transform']) - num_good_grasps).to(self.device)

        #### All into the same Coordinate system ####
        trimesh, grasps["grasp_transform"] = Transform.start_transform(trimesh, torch.Tensor(grasps['grasp_transform']).to(self.device))

        ###### grasp data ######
        for i in range(len(grasps['grasp_transform'])):
            w = torch.linalg.vector_norm(torch.tensor(grasps['contact_points'][i][0] - grasps['contact_points'][i][1]), ord=1).to(self.device)
            t = torch.linalg.vector_norm(torch.tensor(grasps['grasp_transform'][i][:3, 3]), ord=1).to(self.device)
            t_vec = torch.tensor(grasps['grasp_transform'][i][:3, 3]).to(self.device)
            euler = matrix_to_euler_angles(grasps['grasp_transform'][i][:3, :3], "ZXZ")
            quaternion = matrix_to_quaternion(grasps['grasp_transform'][i][:3, :3])
            s = se3.SE3()
            s.update(torch.unsqueeze(grasps['grasp_transform'][i][:3, :], dim=0))
            log_map = s.log_map()

            g_data = torch.hstack((w, t, t_vec, euler, quaternion, torch.squeeze(log_map)))
            grasp_data = torch.vstack((grasp_data, torch.unsqueeze(g_data, dim=0)))

        ###### object data ######
        points = torch.tensor(trimesh.vertices).to(device=self.device)
        distance = torch.linalg.vector_norm(points, ord=1, dim=1)
        max = torch.max(distance)
        min = torch.min(distance)
        mean = torch.mean(distance)
        var = torch.var(distance)

        object_data = torch.stack((max, min, mean, var, num_good_grasps, num_bad_grasps), dim=0)

        return torch.unsqueeze(object_data, dim=0), grasp_data

    @torch.no_grad()
    def evaluate_statistics(self, object_data, grasp_data):

        max_o, _ = torch.max(object_data, dim=0)
        min_o, _ = torch.min(object_data, dim=0)
        mean_o = torch.mean(object_data, dim=0)
        var_o = torch.var(object_data, dim=0)

        max_g, _ = torch.max(grasp_data, dim=0)
        min_g, _ = torch.min(grasp_data, dim=0)
        mean_g = torch.mean(grasp_data, dim=0)
        var_g = torch.var(grasp_data, dim=0)

        object_statistics = {'max': max_o.cpu(),
                            'min': min_o.cpu(),
                            'mean': mean_o.cpu(),
                            'var': var_o.cpu(),
                            }

        grasp_statistics = {'max': max_g.cpu(),
                            'min': min_g.cpu(),
                            'mean': mean_g.cpu(),
                            'var': var_g.cpu()
                            }

        return object_statistics, grasp_statistics


    def save_statistics(self, object_statistics, grasp_statistics):

        np.savez(os.path.join(self.root_directory, 'object_statistics.npz'), **object_statistics)
        np.savez(os.path.join(self.root_directory, 'grasp_statistics.npz'), **grasp_statistics)


class GraspnetAcronymDataset(Dataset):
    ''' Preprocessed grasps by contact graspnet team. Every grasp is collision proved with panda gripper'''
    def __init__(
            self,
            root_directory: os.PathLike,
            dataset_directory: os.PathLike,
            h5_directory: os.PathLike,
            files: list,
            subbatch_size: int,
            device="cpu",
            train_split: float = 0.9,
            processor: Callable = lambda x: x, # function to process the data
            train: bool = True
    ):
        self.device = device
        self.root_directory = Path(root_directory)
        self.data_directory = Path(dataset_directory)
        self.h5_directory = Path(h5_directory)
        self.file_names = files
        self.train_split = train_split
        self.sample_size = subbatch_size
        self.processor = processor
        self.train_flag = train
        self.batches = []
        self.__load_data()
        # data is saved in a dict where each key is the path to a object embedding and each value is a tensor of grasps
        # with dimensions [num_samples, sample_size, Transformation Matrix  = 4x4]
        #  a sample consists of sample_size grasp

    # return sample given a index
    def __getitem__(self, idx):
        self.idx = idx
        if self.train_flag:
            (file_idx, batch_idx) = self.batches[idx]
            grasps_npz = np.load(os.path.join(self.data_directory, self.file_names[file_idx]))
            grasps = grasps_npz['grasp_transform'][grasps_npz['successful'].astype(bool)]
            grasps = grasps[batch_idx*self.sample_size:(batch_idx+1)*self.sample_size]  #, self.contact_points[idx]
            path = os.path.join(self.h5_directory, self.file_names[file_idx].replace('.npz', '.h5'))
        else:

            grasps_npz = np.load(os.path.join(self.data_directory, self.file_names[self.idx]))
            grasps = grasps_npz['grasp_transform'][grasps_npz['successful'].astype(bool)]
            path = os.path.join(self.h5_directory, self.file_names[self.idx].replace('.npz', '.h5'))

        trimesh = load_mesh(path, self.root_directory)
        return self.processor(trimesh, torch.tensor(grasps, dtype=torch.get_default_dtype(), device=self.device))


    def get_current(self):
        idx = self.idx
        if self.train_flag:
            (file_idx, batch_idx) = self.batches[idx]
            grasps_npz = np.load(os.path.join(self.data_directory, self.file_list[file_idx]))
            grasps = grasps_npz['grasp_transform'][grasps_npz['successful'].astype(bool)]
            grasps = grasps[batch_idx * self.sample_size:(batch_idx + 1) * self.sample_size]  # , self.contact_points[idx]
        else:
            grasps_npz = np.load(os.path.join(self.data_directory, self.file_list[self.idx]))
            grasps = grasps_npz['grasp_transform'][grasps_npz['successful'].astype(bool)]

        path = os.path.join(self.h5_directory, self.file_list[self.idx].replace('.npz', '.h5'))
        trimesh = load_mesh(path, self.root_directory)
        return self.processor(trimesh, torch.tensor(grasps, dtype=torch.get_default_dtype(), device=self.device)), trimesh

    def __len__(self):
        if self.train_flag:
            return len(self.batches)
        else:
            return len(self.file_names)

    def get_post_processing(self):
        return self.processor.post_processing

    def get_inverse_representation(self):
        return self.processor.inverse_representation_function

    def __load_data(self):
        valid_files = []
        for idx, file in enumerate(self.file_names):
            try:
                if idx == 100:
                    break
                load_mesh(os.path.join(self.h5_directory, file.replace('.npz', '.h5')), self.root_directory)
                grasps_npz = np.load(os.path.join(self.data_directory, file))
                grasps = {}
                for k in grasps_npz.files:
                    grasps[k] = grasps_npz[k]
                grasps = {k: grasps[k][grasps['successful'].astype(bool)] for k in grasps if k in grasps.keys()}
                if grasps['grasp_transform'].shape[0] > self.sample_size:
                    num_batches = grasps['grasp_transform'].shape[0] // self.sample_size
                    idx_list = list(range(num_batches))
                    objects = [len(valid_files)] * num_batches
                    batches = list(zip(objects, idx_list))
                    self.batches = self.batches + batches
                    valid_files = valid_files + [file]

            except Exception as e:
                logging.warning(str(e))
        self.file_names = valid_files

    def get_screwdriver(self):
        file = 'ScrewDriver_28d88f47398c462977d2282deeabd2a4_0.014402062869257133.npz'
        path = os.path.join(self.h5_directory, file.replace('.npz', '.h5'))
        trimesh = load_mesh(path, self.root_directory)
        obj = trimesh

        grasps_npz = np.load(os.path.join(self.data_directory, file))
        grasps = {}
        for k in grasps_npz.files:
            grasps[k] = grasps_npz[k]
        # grasps = {k: grasps[k][grasps['successful'].astype(bool)] for k in grasps if k in grasps.keys()}
        good_grasps = torch.tensor(grasps['grasp_transform'][grasps['successful'].astype(bool)], device=self.device, dtype=torch.get_default_dtype())
        bad_grasps = torch.tensor(grasps['grasp_transform'][~grasps['successful'].astype(bool)], device=self.device, dtype=torch.get_default_dtype())

        return self.processor(obj, good_grasps), bad_grasps

    # iterator to generate a shuffled train/val split
    def get_train_val_split_iterator(self, train_share: float = 0.9, shuffle: bool = True):
        train_size = int(train_share * len(self))
        val_size = len(self) - train_size
        indices = np.arrange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield self[i][:train_size], self[i][train_size:]

    @torch.no_grad()
    def get_statistics(self):
        """
        returns statistics of the dataset in a dictionary
        :return:
        """
        obj_data = torch.empty((0, 6), dtype=torch.get_default_dtype(), device=self.device)
        grasp_data = torch.empty((0, 12), dtype=torch.get_default_dtype(), device=self.device)
        print("start object analysis")
        for file in self.file_names:
            try:
                o_data, g_data = self.sample_properties(file)
                obj_data = torch.cat((obj_data, o_data), dim=0)
                grasp_data = torch.cat((grasp_data, g_data), dim=0)
                print('file done: ', file)
            except:
                print('error in file: ', file)

        object_statistics, grasp_statistics = self.evaluate_statistics(obj_data, grasp_data)
        self.save_statistics(object_statistics, grasp_statistics)
        print('statistics saved')

        return {'object_statistics': object_statistics, 'grasp_statistics': grasp_statistics}


    @torch.no_grad()
    def sample_properties(self, file):
        object_data = torch.empty((0, 6), dtype=torch.get_default_dtype(), device=self.device)
        grasp_data = torch.empty((0, 12), dtype=torch.get_default_dtype(), device=self.device)
        path = os.path.join(self.h5_directory, file.replace('.npz', '.h5'))
        trimesh = load_mesh(path, self.root_directory)
        grasps_npz = np.load(os.path.join(self.data_directory, file))
        grasps = {}
        for k in grasps_npz.files:
            grasps[k] = grasps_npz[k]


        good_grasps = {k: grasps[k][grasps['successful'].astype(bool)] for k in grasps if k in grasps.keys()}
        num_good_grasps = torch.tensor(len(good_grasps['grasp_transform'])).to(self.device)
        num_bad_grasps = torch.tensor(len(grasps['grasp_transform']) - num_good_grasps).to(self.device)

        #### All into the same Coordinate system ####
        trimesh, grasps["grasp_transform"] = Transform.start_transform(trimesh, torch.Tensor(grasps['grasp_transform']).to(self.device))

        ###### grasp data ######
        for i in range(len(grasps['grasp_transform'])):
            w = torch.linalg.vector_norm(torch.tensor(grasps['contact_points'][i][0] - grasps['contact_points'][i][1]), ord=1).to(self.device)
            t = torch.linalg.vector_norm(torch.tensor(grasps['grasp_transform'][i][:3, 3]), ord=1).to(self.device)
            t_vec = torch.tensor(grasps['grasp_transform'][i][:3, 3]).to(self.device)
            euler = matrix_to_euler_angles(grasps['grasp_transform'][i][:3, :3], "ZXZ")
            quaternion = torch.linalg.vector_norm(matrix_to_quaternion(grasps['grasp_transform'][i][:3, :3]), ord=1).to(self.device)
            se = se3.SE3()
            se.update(torch.unsqueeze(grasps['grasp_transform'][i][:3, :], dim=0))
            se_3_log_map_norm = torch.linalg.vector_norm(torch.squeeze(se.log_map()), ord=1).to(self.device)
            so = so3.SO3()
            so.update(torch.unsqueeze(grasps['grasp_transform'][i][:3, :3], dim=0))
            so_3_log_map_norm = torch.linalg.vector_norm(torch.squeeze(so.log_map()), ord=1).to(self.device)
            one = torch.tensor([1]).to(self.device)
            g_data = torch.hstack((w, t, t_vec, euler, quaternion, se_3_log_map_norm, so_3_log_map_norm, one))
            grasp_data = torch.vstack((grasp_data, torch.unsqueeze(g_data, dim=0)))

        ###### object data ######
        points = torch.tensor(trimesh.vertices).to(device=self.device)
        distance = torch.linalg.vector_norm(points, ord=1, dim=1)
        max = torch.max(distance)
        min = torch.min(distance)
        mean = torch.mean(distance)
        var = torch.var(distance)

        object_data = torch.stack((max, min, mean, var, num_good_grasps, num_bad_grasps), dim=0)

        return torch.unsqueeze(object_data, dim=0), grasp_data

    @torch.no_grad()
    def evaluate_statistics(self, object_data, grasp_data):

        max_o, _ = torch.max(object_data, dim=0)
        min_o, _ = torch.min(object_data, dim=0)
        mean_o = torch.mean(object_data, dim=0)
        var_o = torch.var(object_data, dim=0)

        max_g, _ = torch.max(grasp_data, dim=0)
        min_g, _ = torch.min(grasp_data, dim=0)
        mean_g = torch.mean(grasp_data, dim=0)
        var_g = torch.var(grasp_data, dim=0)

        object_statistics = {'max': max_o.cpu(),
                            'min': min_o.cpu(),
                            'mean': mean_o.cpu(),
                            'var': var_o.cpu(),
                            }

        grasp_statistics = {'max': max_g.cpu(),
                            'min': min_g.cpu(),
                            'mean': mean_g.cpu(),
                            'var': var_g.cpu()
                            }

        return object_statistics, grasp_statistics


    def save_statistics(self, object_statistics, grasp_statistics):

        np.savez(os.path.join(self.root_directory, 'object_statistics.npz'), **object_statistics)
        np.savez(os.path.join(self.root_directory, 'grasp_statistics.npz'), **grasp_statistics)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    root_directory = "/home/temp_store/jandl/Dataset/acronym"
    data_directory = "/home/temp_store/jandl/Dataset/acronym/mesh_contacts_graspnet"
    h5_directory = "/home/temp_store/jandl/Dataset/acronym/grasps"
    train_fraction = 0.8
    files = os.listdir(data_directory)
    train_files, test_files = random_split(files, [train_fraction, 1 - train_fraction],
                                           generator=torch.Generator().manual_seed(42))

    # trainset = GraspnetAcronymDataset(
    #     root_directory,
    #     data_directory,
    #     h5_directory,
    #     list(train_files),
    #     subbatch_size=100,
    #     device="cuda:0",
    #     train_split=0.9,
    #     processor = lambda x: x)
    #
    # testset = GraspnetAcronymDataset(
    #     root_directory,
    #     data_directory,
    #     h5_directory,
    #     list(test_files),
    #     subbatch_size=100,
    #     device="cuda:0",
    #     train_split=0.9,
    #     processor = lambda x: x,
    #     train=False)
    #
    # testset[5]
    # trainset[5]
    #

    dataset = GraspnetAcronymDataset(
        root_directory,
        data_directory,
        h5_directory,
        list(train_files),
        subbatch_size=100,
        device="cuda:0",
        train_split=0.9,
        processor = lambda x: x)

    dataset.get_statistics()

    data_object = np.load(os.path.join(root_directory, 'object_statistics.npz'))
    data_grasp = np.load(os.path.join(root_directory, 'grasp_statistics.npz'))
    print(data_object)
    print(data_grasp)
