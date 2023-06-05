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


@hydra.main(config_path="configs", config_name="sim_block_push_config.yaml")
def main(cfg: DictConfig) -> None:
    # get path to the bet repo -> required for the block push environment if needed
    # os.environ['ASSET_PATH'] = "/home/moritz/code/bet"
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=wandb.config, mode="disabled")
    # load the required classes to train and test the agent
    workspace_manager = hydra.utils.instantiate(cfg.workspaces)
    agent = hydra.utils.instantiate(cfg.agents)
    # get the scaler instance and set teh bounds for the sampler if required
    agent.get_scaler(workspace_manager.scaler)
    agent.set_bounds(workspace_manager.scaler)

    agent.load_pretrained_model("/home/david/Project/diffusion/logs/sim_block_push/runs/2022-12-13/11-03-49", sv_name="train_BC.pth")

    from environments.sim_block_push.sim_block_push_datasets import SimBP_Dataset

    dataset = workspace_manager.train_set

    obs = dataset.dataset.get_all_observations()[:1000]
    act = dataset.dataset.get_all_actions()[:1000]

    pred_act = []
    for i in range(len(obs)):
        step_obs = obs[i:i + 1].to('cuda')
        pred_action = agent.predict(step_obs)
        # pred_action = dataset.scaler.inverse_scale_output(pred_action)
        # pred_action = pred_action.reshape(1, -1).detach().cpu().numpy()

        pred_act.append(pred_action)

    pred_act = np.concatenate((pred_act), axis=0)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2)

    traj = "/home/david/Dataset_robo/Block_Push/mode_1/env_000_07.pkl"

    a = ['x', 'y', 'z']

    for i in range(2):
        axs[i].plot(act[:, i], label='gt_' + a[i])
        axs[i].plot(pred_act[:, i], label='pred_' + a[i])
    fig.legend()
    plt.show()


if __name__ == "__main__":
    main()