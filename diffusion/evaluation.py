import os
import logging

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import numpy as np
from csv import writer


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


@hydra.main(config_path="/home/i53/student/jandl/repos/DIGNet/logs/cluster_1/grasping_end_to_end_learning_final/runs/2023-07-31/20-14-09_architecture_optimization/.hydra", config_name="config")
def main(cfg: DictConfig) -> None:

    with open_dict(cfg):
        cfg.pretrained_model_path = "/home/i53/student/jandl/repos/DIGNet/logs/cluster_1/grasping_end_to_end_learning_final/runs/2023-07-31/20-14-09_architecture_optimization"

    # if cfg.pretrained_model_path is not None:
    #     root_path = cfg.root_path
    #     data_path = cfg.data_path
    #     h5_path = cfg.h5_path
    #     pre_trained_model_path = cfg.pretrained_model_path
    #     cfg = OmegaConf.load(cfg.pretrained_model_config)
    #     cfg.root_path = root_path
    #     cfg.data_path = data_path
    #     cfg.h5_path = h5_path
    #     cfg.pretrained_model_path = pre_trained_model_path
    #     cfg.group = cfg.group + "_eval"


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

    # wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    #
    # run = wandb.init(
    #     project="evaluation",
    #     entity=cfg.wandb.entity,
    #     group="size_evaluation_variance",
    #     # mode="disabled",
    #     config=wandb.config
    # )
    # agent.evaluate_agent_sizes(workspace_manager.data_loaders['test'])
    # wandb.finish()

    # agent.evaluate_agent(workspace_manager.data_loaders['test'])
    agent.visualize_trajectory_backward(
        workspace_manager.data_loaders['test'],
        num_obj=1,
        num_samples=1,
    )
    # eval the agent
    # agent.visualize_diffused_distribution(workspace_manager.data_loaders['test'], num_steps=5, num_obj=1, num_samples=100)
    # agent.visualize_forward_trajectory(workspace_manager.data_loaders['test'], num_obj=1, num_samples=10)
    # agent.visualize_backward_trajectory(workspace_manager.data_loaders['test'], num_obj=1, num_samples=10)
    # diversity_loss, accuracy_loss, coverage, gripper_check_loss = agent.evaluate_agent(workspace_manager.data_loaders['train'], path=cfg.pretrained_model_path)
    # print(f"precision-CD: {accuracy_loss}")
    # print(f"coverage-CD: {diversity_loss}")
    # print(f"coverage: {coverage}")
    # print(f"gripper check accuracy:  {gripper_check_loss}")
    # log.info("done")



    # row = [cfg.representation, cfg.pointnet_radius, cfg.scaler, cfg.predict_epsilon, accuracy_loss, diversity_loss, coverage, gripper_check_loss]
    #
    # with open('/home/i53/student/jandl/repos/DIGNet/diffusion/eval_table.csv', 'a') as f_object:
    #
    #     np.savetxt(f_object, [row], fmt="%s", delimiter=' & ', newline='\\\\\n')
    #
    #     # Close the file object
    #     f_object.close()

if __name__ == "__main__":
    main()
