import logging
from pathlib import Path
import sys

from omegaconf import DictConfig
import numpy as np
import torch
import hydra
from tqdm import tqdm
import wandb
from diffusion.workspaces.base_workspace_manager import BaseWorkspaceManger
# from diffusion.environments.utils import get_split_idx
# from diffusion.agents.diffusion_agents.grasp_diffusion_agent import GraspDiffusionAgent
# from diffusion.environments.diffusion_grasp.dataloader import AcronymPreEmbeddingDataset

log = logging.getLogger(__name__)


class DiffusionGraspManager(BaseWorkspaceManger):
    def __init__(
            self,
            seed: int,
            device: str,
            dataset_fn: DictConfig,
            eval_n_times: int,
            eval_n_steps,
            render: bool,
            train_batch_size: int = 80,
            test_batch_size: int = 20,
            num_workers: int = 4,
            pin_memory: bool = True,
            train_fraction: float = 0.95
    ):
        super().__int__(seed, device)

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.eval_n_times = eval_n_times
        self.eval_n_steps = eval_n_steps
        self.train_batch_size = train_batch_size
        self.data_path = dataset_fn.data_directory
        self.test_batch_size = test_batch_size
        # train and test datasets
        self.train_set, self.test_set = hydra.utils.call(dataset_fn)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_fraction = train_fraction
        # Do I need Scaler?
        self.scaler = None

        self.data_loaders = self.make_dataloaders()
        self.render = render

    def make_dataloaders(self):
        """
        Create a training and test dataloader using the dataset instances of the task
        """
        train_dataloader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        test_dataloader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return {
            "train": train_dataloader,
            "test": test_dataloader
        }


    def get_scaler_data(self, scaler):
        """
        Computes all necessary statistics for the scaler from dataset
        """


    def test_agent(self):
        # TODO: test agent with Benchmark Repo from Nic
        pass
