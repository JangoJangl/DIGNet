import logging
import abc
import os
import copy

import torch 
import numpy as np
from torch.utils.data import random_split


log = logging.getLogger(__name__)


class BaseWorkspaceManger(abc.ABC):

    def __int__(
            self,
            seed: int,
            device: str
    ):
        self.seed = seed
        self.device = device
        self.working_dir = os.getcwd()
        self.env_name = 'BaseEnvironment'

    @abc.abstractmethod
    def test_agent(self, agent):
        pass

    @staticmethod
    def split_datasets(dataset, train_fraction=0.9, random_seed=42):
        dataset_length = len(dataset)
        lengths = [
            int(train_fraction * dataset_length),
            dataset_length - int(train_fraction * dataset_length),
        ]
        train_set, val_set = random_split(
            dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
        )
        return train_set, val_set
