import os
import logging

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from tkinter import filedialog
import torch

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


@hydra.main(config_path="configs", config_name="franka_kitchen_main_config.yaml")
def main(cfg: DictConfig) -> None:
    # get path to the bet repo -> required for the block push environment if needed
    # os.environ['ASSET_PATH'] = "/home/moritz/code/bet"
    os.environ['HYDRA_FULL_ERROR'] = '1'
    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=wandb.config, mode="disabled")
    # load the required classes to train and test the agent
    workspace_manager = hydra.utils.instantiate(cfg.workspaces)
    agent = hydra.utils.instantiate(cfg.agents)
    # get the scaler instance and set teh bounds for the sampler if required
    agent.get_scaler(workspace_manager.scaler)
    agent.set_bounds(workspace_manager.scaler)

    os_path = os.getcwd().rsplit("/", 3)[0]
    file_path = filedialog.askdirectory(
        title="Select run folder",
        # filetypes=[("PyTorch Model", ".*")],
        initialdir=os_path,
    )

    agent.load_pretrained_model(file_path, sv_name='train_Bet_lr1e-4.pth')

    # after training evaluate the performance of the agent by running on the environment
    result_dict = workspace_manager.test_agent(
        agent,
        get_mean=None,
        n_inference_steps=50,
    )
    log.info("done")


if __name__ == "__main__":
    main()