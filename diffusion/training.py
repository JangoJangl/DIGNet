import os
import logging

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


@hydra.main(config_path="configs", config_name="diffusion_grasp_graspnet_main_config")
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # init wandb logger and config from hydra path 
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project, 
        entity=cfg.wandb.entity,
        group=cfg.group,
        # mode="disabled",
        config=wandb.config
    )

    # load the required classes to train and test the agent
    workspace_manager = hydra.utils.instantiate(cfg.workspaces)
    agent = hydra.utils.instantiate(cfg.agents)
    # get the scaler instance and set teh bounds for the sampler if required
    agent.get_scaler(workspace_manager.scaler)
    agent.set_bounds()

    # train the agent
    agent.train_agent(
        workspace_manager.data_loaders['train'],
        workspace_manager.data_loaders['test']
    )

    log.info("done")
    wandb.finish()


if __name__ == "__main__":
    main()