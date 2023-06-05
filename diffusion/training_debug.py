import os
import logging

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch

from networks.LfP_models.utils import State
from networks.k_diffusion.classifier_free_sampler import ClassifierFreeSampleModel
from agents.diffusion_agents.k_diffusion_agent import KDiffusionAgent

log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)



@hydra.main(config_path="configs", config_name="multipath_main_config.yaml")
def main(cfg: DictConfig) -> None:
    
    # get path to the bet repo -> required for the block push environment if needed
    os.environ['ASSET_PATH'] = "/home/moritz/code/bet"
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
    agent.set_bounds(workspace_manager.scaler)

    # train the agent 
    agent.train_agent(
        workspace_manager.data_loader['train'],
        workspace_manager.data_loader['test']
    )
    # after training evaluate the performance of the agent by running on the environment
    # agent.model = ClassifierFreeSampleModel(agent.model, cond_lambda=1.5)
    if isinstance(agent, KDiffusionAgent):
        if cfg.cond_mask_prob > 0:
            agent.model = ClassifierFreeSampleModel(agent.model, cond_lambda=1.5)
            result_dict = workspace_manager.test_agent(
                agent,
                get_mean=None,
                log_wandb=True,
                n_inference_steps=5,
                new_sampler_type='ddim',
                noise_scheduler='iddpm'
                )
        else:
            result_dict = workspace_manager.test_agent(
                agent,
                get_mean=None,
                log_wandb=True,
                new_sampler_type='heun',
                n_inference_steps=5,
                noise_scheduler='iddpm'
                )
    else:
        result_dict = workspace_manager.test_agent(
            agent,
            get_mean=None,
            log_wandb=True,
            #extra_args={
            #    'keep_last_action': False
            # }
            )
    log.info("done")
    wandb.finish()


if __name__ == "__main__":
    main()