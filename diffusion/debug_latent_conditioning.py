from math import fabs
import os
import logging

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from networks.k_diffusion.classifier_free_sampler import ClassifierFreeSampleModel
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import einops 
from matplotlib.ticker import NullFormatter, MaxNLocator
from numpy import linspace
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)

# Define a function to make the ellipses
def ellipse(ra,rb,ang,x0,y0,Nb=100):
    xpos,ypos=x0,y0
    radm,radn=ra,rb
    an=ang
    co,si=np.cos(an),np.sin(an)
    the=linspace(0,2*np.pi,Nb)
    X=radm*np.cos(the)*co-si*radn*np.sin(the)+xpos
    Y=radm*np.cos(the)*si+co*radn*np.sin(the)+ypos
    return X,Y


def main(model_store_path: str) -> None:
    
    use_max_values = False
    data_range = 2 
    make_gif = True
    n_inference_steps = 3
    gif_duration = 1
    random_idx = 172
    noise_scheduler = 'exponential'
    safe_individual_plot = True
    window_zoom = 1
    zoom_loc = 'lower left'
    filenames_2 = []
    filenames = []
    
    # make new subdir
    file_path = os.path.join(model_store_path, f'{n_inference_steps}_steps_idx_{random_idx}')
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    
    # get the path to the config yaml file
    cfg_store_path = os.path.join(model_store_path,'.hydra/config.yaml')
    # get path to the bet repo -> required for the block push environment if needed
    os.environ['ASSET_PATH'] = "/home/moritz/code/bet"
    # init wandb logger and config from hydra path 
    cfg = OmegaConf.load(cfg_store_path) 
    # load the required classes to train and test the agent 
    workspace_manager = hydra.utils.instantiate(cfg.workspaces)
    agent = hydra.utils.instantiate(cfg.agents)
    # get the scaler instance and set teh bounds for the sampler if required
    agent.get_scaler(workspace_manager.scaler)
    agent.set_bounds(workspace_manager.scaler)
    agent.load_pretrained_model(model_store_path)
    agent.model.eval()
    agent.moe.eval()
    # get a random state and goal 
    state, action, mask, goal = workspace_manager.datasets[0][random_idx]
    state = np.random.uniform([9.5, 9.5], [10.5, 10.5], (1,2))
    goal = np.array([10, 15]).reshape(1,2)
    state = torch.from_numpy(state).to('cuda').to(torch.float32)
    goal = torch.from_numpy(goal).to('cuda').to(torch.float32)
    action = agent.scaler.scale_output(torch.from_numpy(action).reshape(1,2))
    sigmas = agent.get_noise_schedule(5, 'exponential')
    sigmas = torch.flip(sigmas, [0])
    sigmas[0] = 0.01
    # predict first conditioning
    action = torch.from_numpy(np.array([0.1, 1]).reshape(1,2)).to('cuda').to(torch.float32)
    latent_action_1 = agent.forward_ode(state, action, goal, sigmas)
    latent_pred_1 = agent.moe(torch.cat([state, goal], dim=-1))
    results_1 = agent.predict(state,goal)
    # next goal 
    goal = np.array([15, 10]).reshape(1,2)
    goal = torch.from_numpy(goal).to('cuda').to(torch.float32)
    action = torch.from_numpy(np.array([1, 0.1]).reshape(1,2)).to('cuda').to(torch.float32)
    latent_action_2 = agent.forward_ode(state, action, goal, sigmas)
    latent_pred_2 = agent.moe(torch.cat([state, goal], dim=-1))
    results_2 = agent.predict(state,goal)

    goal = np.array([15, 15]).reshape(1,2)
    goal = torch.from_numpy(goal).to('cuda').to(torch.float32)
    action = torch.from_numpy(np.array([0.7, 0.7]).reshape(1,2)).to('cuda').to(torch.float32)
    latent_pred_3 = agent.moe(torch.cat([state, goal], dim=-1))
    latent_action_3 = agent.forward_ode(state, action, goal, sigmas)
    results_3 = agent.predict(state,goal)

    print(f'latent action 1: {latent_action_1}')
    print(f'latent action 2: {latent_action_2}')
    print(f'latent action 3: {latent_action_3}')
    print(f'latent pred 1: {latent_pred_1}')
    print(f'latent pred 2: {latent_pred_2}')
    print(f'latent pred 3: {latent_pred_3}')

    print(f'results 1: {results_1}')
    print(f'results 2: {results_2}')
    print(f'results 3: {results_3}')

    print('done')


if __name__ == "__main__":
    # model_store_path = '/home/moritz/code/generative_learning_from_play/logs/multipath/runs/2023-01-15/19-57-28'
    model_store_path = '/home/moritz/code/generative_learning_from_play/logs/multipath/runs/2023-02-16/09-27-10'
    main(model_store_path)