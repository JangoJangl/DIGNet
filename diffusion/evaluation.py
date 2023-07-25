import os
import logging

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import numpy as np


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


@hydra.main(config_path="/home/i53/student/jandl/repos/DIGNet/logs/cluster/grasping_end_to_end_learning/runs/2023-07-22/00-10-25_architecture_optimization/.hydra", config_name="config")
def main(cfg: DictConfig) -> None:

    with open_dict(cfg):
        cfg.pretrained_model_path = "/home/i53/student/jandl/repos/DIGNet/logs/cluster/grasping_end_to_end_learning/runs/2023-07-22/00-10-25_architecture_optimization"

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_default_dtype(torch.float64)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # load the required classes to test the agent
    workspace_manager = hydra.utils.instantiate(cfg.workspaces)
    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(cfg.pretrained_model_path)
    # get the scaler instance and set teh bounds for the sampler if required
    agent.set_processor(workspace_manager.get_processor())
    agent.set_bounds(cfg.grasp_dim)

    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # run = wandb.init(
    #     project=cfg.wandb.project,
    #     entity=cfg.wandb.entity,
    #     group=cfg.group,
    #     # mode="disabled",
    #     config=wandb.config
    # )
    # agent.visualize_trajectory(
    #     workspace_manager.data_loaders['test'],
    #     num_trajectories=3
    # )
    # eval the agent
    diversity_loss, accuracy_loss, coverage = agent.evaluate_agent(workspace_manager.data_loaders['test'])
    print(f"precision-CD: {accuracy_loss}")
    print(f"coverage-CD: {diversity_loss}")
    print(f"coverage: {coverage}")
    print(f"gripper check accuracy: ")
    log.info("done")


if __name__ == "__main__":
    main()
