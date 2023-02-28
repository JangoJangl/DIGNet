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

from openpoints.utils import EasyConfig
from models import build_models
from utils.graspnet_mesh_utils import Object

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

"""
creates custom dataset for training of a diffusion based grasp proposal method
data: 
- {meta}
- [object_embeddings]
- [grasps]
                
"""


def get_paths(paths):
    return os.listdir(paths.grasps)


def load_data(file, root):
    with h5py.File(os.path.join(root, 'grasps', file), "r") as f:
        print('data loaded')
        mesh_file = f["object/file"][()].decode('utf-8')
        mesh_scale = f["object/scale"][()]
        if os.path.exists(os.path.join(root, mesh_file)):
            o = Object(os.path.join(root, mesh_file))
            o.rescale(mesh_scale)
        else:
            o = None

    return os.path.join(root, 'grasps', file), o


def load_model(path):
    pass


def extract_features(o, mods):
    feat = {}
    for m in mods:
        m.add_object(o)
        feat[type(m).__name__] = m.extract()
        m.clear()
    return feat



def save_dataset(d, emb):
    pass


def h5py_dataset_iterator(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset):  # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group):  # test for group (go down)
            yield from h5py_dataset_iterator(item, path)


def visualize_mesh_o3d(mesh):
    cos = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.1, center=(0, 0, 0))
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([cos, mesh])


def visualize_mesh_trimesh(mesh):
    s = trimesh.Scene(mesh)
    s.show(background=[30, 30, 30, 255], smooth=True, flags={'axis': True})



if __name__ == "__main__":
    cfg_path = 'config.yaml'
    cfg = EasyConfig()
    cfg.load(cfg_path)
    grasp_paths = get_paths(cfg.paths)
    models = build_models(cfg.models)
    for p in grasp_paths:
        grasp_hdf5_path, obj = load_data(p, cfg.paths.acronym)
        # visualize_mesh_o3d(obj.mesh.as_open3d)
        if obj is not None:
            embeddings = extract_features(obj, models)
            save_dataset(grasp_hdf5_path, embeddings)