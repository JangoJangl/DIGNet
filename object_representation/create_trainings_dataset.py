import yaml
import os
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

from openpoints.utils import EasyConfig
from models import build_models

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

"""
creates custom dataset for training of a diffusion based grasp proposal method
data: 
- {meta}
- [object_embeddings]
- [grasps]
                
"""


def get_paths(dir_name):
    # data = {grasps, meshes}
    pass


def load_data(path):
    pass


def load_model(path):
    pass


def extract_features(d, mod):
    pass


def save_dataset(d, emb):
    pass


if __name__ == "__main__":
    cfg_path = 'config.yaml'
    cfg = EasyConfig()
    cfg.load(cfg_path)
    data_paths = get_paths(cfg.paths)
    models = build_models(cfg.models)
    for p in data_paths:
        data = load_data(p)
        embeddings = extract_features(data, models)
        save_dataset(data, embeddings)