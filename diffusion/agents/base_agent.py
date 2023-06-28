import abc
import os
import logging

import torch
from omegaconf import DictConfig
import hydra

from agents.scaler.scaler_class import Scaler

# A logger for this file
log = logging.getLogger(__name__)


class BaseAgent(abc.ABC):

    def __init__(
        self,
            model: DictConfig,
            embedder: DictConfig,
            optimization: DictConfig,
            device: str,
            max_train_steps: int,
            eval_every_n_steps: int,
            epoch= None,
    ):
        self.scaler = None
        self.model = hydra.utils.instantiate(model).to(device)
        self.embedder = hydra.utils.instantiate(embedder).to(device)
        self.optimizer = hydra.utils.instantiate(
            optimization, params=list(self.embedder.parameters()) + list(self.model.get_params())
        )

        self.epoch = epoch

        self.device = device
        self.steps = 0
        self.max_train_steps = int(max_train_steps)
        self.eval_every_n_steps = eval_every_n_steps
        self.working_dir = os.getcwd()
        total_params = sum(p.numel() for p in list(self.embedder.parameters()) + list(self.model.get_params()))
        log.info("The model has a total amount of {} parameters".format(total_params))

    @abc.abstractmethod
    def train_agent(self, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader):
        """
        Main method to train the agent on the given train and test data
        """
        pass

    @abc.abstractmethod
    def train_step(self, state: torch.Tensor, action: torch.Tensor):
        """
        Executes a single training step on a mini-batch of data
        """
        pass

    @abc.abstractmethod
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """
        Method for evaluating the model on one batch of data consisting of two tensors
        """
        pass

    @abc.abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        pass

    @abc.abstractmethod
    def set_bounds(self, scaler: Scaler):
        """
        Method to define the required bounds for the sampling classes or agents
        """
        pass

    def get_scaler(self, scaler: Scaler):
        self.scaler = scaler

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """
        if sv_name is None:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        else:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, sv_name)))
        log.info('Loaded pre-trained model parameters')

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """
        if sv_name is None:
            torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(store_path, sv_name))
