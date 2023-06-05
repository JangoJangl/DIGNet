import copy
import glob

import yaml
import os
import sys
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path

from object_representation.openpoints.utils import EasyConfig
# from models import build_models
from utils.graspnet_mesh_utils import Object

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

def get_paths(paths):
    return os.listdir(paths['grasps'])

def load_data(files, root):
    sizes = [10, 50, 100, 500, 1000]
    for min_valid in sizes:
        print(min_valid)
        dataset = {}
        for idx, file in enumerate(files):
            with h5py.File(os.path.join(root, 'grasps', file), "r") as f:
                mesh_embedding = f["object/file"][()].decode('utf-8').replace('.obj', '.npz').replace('meshes', 'embeddings')
                grasps = np.array(f["grasps/transforms"][()])
                grasp_success = np.array(f["grasps/qualities/flex/object_in_gripper"][()], dtype=bool)
                valid_grasps = grasps[grasp_success]
                if valid_grasps.shape[0] > min_valid:
                    num_kick = valid_grasps.shape[0] % min_valid
                    kick_indices = np.random.choice(valid_grasps.shape[0], num_kick, replace=False)
                    valid_grasps = np.delete(valid_grasps, kick_indices, axis=0)
                    # reshape to (samples, sub_samples, 4, 4)
                    valid_grasps = valid_grasps.reshape((-1, min_valid, 4, 4))
                    dataset[mesh_embedding] = valid_grasps

        np.savez(os.path.join(root, 'dataset_lists', f'acroym{min_valid}.npz'), **dataset)
        print("done")




if __name__ == "__main__":
    paths = {    'acronym': '/home/i53/student/jandl/repos/Dataset/acronym',
                'grasps': '/home/i53/student/jandl/repos/Dataset/acronym/grasps',
                'meshes': '/home/i53/student/jandl/repos/Dataset/acronym/meshes'}
    grasp_paths = get_paths(paths)
    load_data(grasp_paths, paths['acronym'])
