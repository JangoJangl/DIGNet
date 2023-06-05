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

    # processor = hydra.utils.instantiate(processor)

    dataset = GraspnetAcronymDataset(
        root_directory,
        data_directory,
        h5_directory,
        subbatch_size,
        device,
        train_fraction,
        processor)

    return random_split(dataset, [train_fraction, 1 - train_fraction], generator=torch.Generator().manual_seed(42))


class AcronymPreEmbeddingDataset(Dataset):
    def __init__(
            self,
            dataset_directory: os.PathLike,
            embedding_directory: os.PathLike,
            external_grasp_directory: os.PathLike,
            internal_grasp_directory: os.PathLike,
            subbatch_size: int,
            device="cpu",
            train_split: float = 0.9,

    ):
        self.device = device
        self.data_directory = Path(dataset_directory)
        self.embedding_directory = Path(embedding_directory)
        self.external_grasp_directory = Path(external_grasp_directory)
        self.internal_grasp_directory = Path(internal_grasp_directory)
        self.file_names = os.listdir(self.external_grasp_directory)
        self.subbatch_size = subbatch_size
        self.train_split = train_split
        self.obj_embeddings, self.external_grasps, self.internal_grasps = self.__load_data()
        # data is saved in a dict where each key is the path to a object embedding and each value is a tensor of grasps
        # with dimensions [num_samples, sample_size, Transformation Matrix  = 4x4]
        #  a sample consists of sample_size grasps
        self.sample_size = self.external_grasps[0].shape[1]
        self.num_objects = len(self.external_grasps)
        self.num_samples = sum([self.external_grasps[i].shape[0] for i in range(len(self.external_grasps))])
        self.samples_per_object = [self.external_grasps[i].shape[0] * self.graps[i].shape[1] for i in range(len(self.external_grasps))]
        self.buckets = np.cumsum(self.grasps_per_object)

    # return sample given a index
    def __getitem__(self, idx):

        object_idx = np.argmax(self.buckets > idx)
        sample_idx = idx - self.buckets[object_idx - 1] if object_idx > 0 else idx
        sample_idx = self.external_grasps[object_idx][sample_idx]
        obj = np.load(os.path.join(self.data_directory, self.obj_embeddings[object_idx]))
        return self.obj_embeddings[object_idx], self.data[object_idx][sample_idx]

    def __len__(self):
        return self.num_samples
    
    def __load_data(self):
        embeddings = []
        external_grasps = {}
        internal_grasps = {}
        for idx, file in enumerate(self.file_names):
            with h5py.File(os.path.join(self.external_grasp_directory, file), "r") as f:
                mesh_embedding = f["object/file"][()].decode('utf-8').replace('.obj', '.npz').replace('meshes',
                                                                                                      'embeddings')
                grasps = np.array(f["grasps/transforms"][()])
                grasp_success = np.array(f["grasps/qualities/flex/object_in_gripper"][()], dtype=bool)
                valid_grasps = grasps[grasp_success]
                if valid_grasps.shape[0] > self.subbatch_size:
                    num_kick = valid_grasps.shape[0] % self.subbatch_size
                    kick_indices = np.random.choice(valid_grasps.shape[0], num_kick, replace=False)
                    internal_grasps = np.load(os.path.join(self.internal_grasp_directory, file.replace('.h5', '.npz')))
                    valid_grasps = np.delete(valid_grasps, kick_indices, axis=0)
                    valid_internal_grasps = np.delete(internal_grasps, kick_indices, axis=0)
                    # reshape to (samples, sub_samples, 4, 4)
                    valid_grasps = valid_grasps.reshape((-1, self.subbatch_size, 4, 4))
                    # Todo: reshape internal grasps
                    embedding_path = os.path.join(self.data_directory, mesh_embedding)
                    embeddings.append(np.load(embedding_path))
                    external_grasps[embedding_path] = valid_grasps
                    internal_grasps[embedding_path] = valid_internal_grasps
        return embeddings, (external_grasps, internal_grasps)

    # iterator to generate a shuffled train/val split
    def get_train_val_split_iterator(self, train_share: float = 0.9, shuffle: bool = True):
        train_size = int(train_share * len(self))
        val_size = len(self) - train_size
        indices = np.arrange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield self[i][:train_size], self[i][train_size:]


class GraspnetAcronymPreEmbeddingDataset(Dataset):
    ''' Preprocessed grasps by contact graspnet team. Every grasp is collision proved with panda gripper'''
    def __init__(
            self,
            root_directory: os.PathLike,
            dataset_directory: os.PathLike,
            h5_directory: os.PathLike,
            subbatch_size: int,
            device="cpu",
            train_split: float = 0.9,

    ):
        self.device = device
        self.root_directory = Path(root_directory)
        self.data_directory = Path(dataset_directory)
        self.h5_directory = Path(h5_directory)
        self.file_names = os.listdir(self.data_directory)
        self.train_split = train_split
        self.sample_size = subbatch_size
        self.embedding_ids, self.grasp_transforms, self.contact_points, self.embeddings = self.__load_data()
        # data is saved in a dict where each key is the path to a object embedding and each value is a tensor of grasps
        # with dimensions [num_samples, sample_size, Transformation Matrix  = 4x4]
        #  a sample consists of sample_size grasps

        self.num_objects = self.embeddings.size(dim=0)
        self.num_samples = self.grasp_transforms.size(dim=0)
        self.samples_per_object = torch.bincount(self.embedding_ids)
        self.buckets = torch.cumsum(self.samples_per_object, dim=0)

    # return sample given a index
    def __getitem__(self, idx):
        # TODO: add internal Grasp
        return self.embeddings[self.embedding_ids[idx]], self.grasp_transforms[idx]

    def __len__(self):
        return self.num_samples

    def __load_data(self):
        grasp_transforms = torch.empty((0, self.sample_size, 4, 4), dtype=torch.float, device=self.device)
        grasp_contact_points = torch.empty((0, self.sample_size, 2, 3), dtype=torch.float, device=self.device)
        obj_idcs = torch.empty((0,), dtype=torch.int, device=self.device) # object indices for each sample
        objs = torch.empty((0, 64), dtype=torch.float, device=self.device) #embedding dimension. Todo: create torch dimension generic
        for idx, file in enumerate(self.file_names):
            try:
                with h5py.File(os.path.join(self.h5_directory, file.replace('.npz', '.h5')), "r") as f:
                    mesh_embedding = f["object/file"][()].decode('utf-8').replace('.obj', '.npz').replace('meshes', 'embeddings')
            except FileNotFoundError as e:
                print(f'File {file.replace(".npz", ".h5")} not found')
                continue
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

                embedding_path = os.path.join(self.root_directory, mesh_embedding)
                try:
                    obj_npz = np.load(embedding_path)
                except FileNotFoundError as e:
                    print(f"Could not find {embedding_path}")
                    continue
                obj = {k: obj_npz[k] for k in obj_npz.files}
                grasp_transforms = torch.vstack((grasp_transforms, torch.from_numpy(grasps['grasp_transform']).float().to(self.device)))
                grasp_contact_points = torch.vstack((grasp_contact_points, torch.from_numpy(grasps['contact_points']).float().to(self.device)))
                obj_idcs = torch.cat((obj_idcs, torch.ones(grasps['grasp_transform'].shape[0], dtype=torch.int).to(self.device) * int(objs.size(dim=0))), dim=0)
                objs = torch.vstack((objs, torch.sum(torch.from_numpy(obj['Pc_Mae']).float().to(self.device), dim=-1)))
                if idx == 100:
                    break

        return obj_idcs, grasp_transforms, grasp_contact_points, objs

    # iterator to generate a shuffled train/val split
    def get_train_val_split_iterator(self, train_share: float = 0.9, shuffle: bool = True):
        train_size = int(train_share * len(self))
        val_size = len(self) - train_size
        indices = np.arrange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield self[i][:train_size], self[i][train_size:]


class GraspnetAcronymDataset(Dataset):
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
        grasp_transforms = torch.empty((0, self.sample_size, 4, 4), dtype=torch.float, device=self.device)
        grasp_contact_points = torch.empty((0, self.sample_size, 2, 3), dtype=torch.float, device=self.device)
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

                grasp_transforms = torch.vstack((grasp_transforms, torch.from_numpy(grasps['grasp_transform']).float().to(self.device)))
                grasp_contact_points = torch.vstack((grasp_contact_points, torch.from_numpy(grasps['contact_points']).float().to(self.device)))
                obj_idcs = torch.cat((obj_idcs, torch.ones(grasps['grasp_transform'].shape[0], dtype=torch.int).to(self.device) * int(idx)), dim=0)
                if idx == 100:
                    break

        return obj_idcs, grasp_transforms, grasp_contact_points

    # iterator to generate a shuffled train/val split
    def get_train_val_split_iterator(self, train_share: float = 0.9, shuffle: bool = True):
        train_size = int(train_share * len(self))
        val_size = len(self) - train_size
        indices = np.arrange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield self[i][:train_size], self[i][train_size:]


