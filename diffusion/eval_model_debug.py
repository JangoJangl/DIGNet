from math import fabs
import os
import logging

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from networks.k_diffusion.classifier_free_sampler import ClassifierFreeSampleModel


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


def main(model_store_path: str, sampler_type: str, steps=5) -> None:
    
    # Choose, which evaluation method should be used
    test_classifier_free_guidance = False
    test_single_variant = True
    test_all_samplers = False
    compare_noisy_sampler = False
    compare_samplers_over_diffent_steps = False
    compare_classifier_free_guidance = False
    render_workspace = False
    compare_kde = False
    
    param_config = {
        'log_wandb': True,
        'num_runs': 50,
        'n_inference_steps': steps,
        'sampler_type': sampler_type,
        'sigma_min': 0.39,
        'sigma_max': 28,
        'rho': 5,
        's_churn': 0,
        's_min': 0,
        'keep_last_action': False,
        'use_scaler': False,
        'get_mean': None,
        'test_classifier_free_guidance': False,
        'cond_lambda': 5,
        'use_kde': False,
        'noise_scheduler': 'exponential',
        'device': 'cpu',
        'model_store_path': model_store_path,
    }
    
    param_config['group'] = f'noise_comp_' + param_config['sampler_type'] + f'steps'
    wandb.config = param_config
    if param_config['log_wandb']:
        run = wandb.init(
            project='block_push_sampler_comparison', 
            entity='bennoq', 
            group=f'k_run_sampler_step_{step}' + param_config['sampler_type'] + f'cfg_{5}', #+f'cond{0}',
            # mode="disabled",  
            config=wandb.config
        )
        param_config = wandb.config
    
    # get the path to the config yaml file
    cfg_store_path = os.path.join(model_store_path,'.hydra/config.yaml')
    # get path to the bet repo -> required for the block push environment if needed
    os.environ['ASSET_PATH'] = "/home/moritz/code/bet"
    # init wandb logger and config from hydra path 
    cfg = OmegaConf.load(cfg_store_path) 
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg['device'] = param_config['device']
    cfg.workspaces['device'] = param_config['device']
    cfg.agents['device'] = param_config['device']
    # cfg.agents.model['device'] = param_config['device']
    # load the required classes to train and test the agent 
    workspace_manager = hydra.utils.instantiate(cfg.workspaces)
    agent = hydra.utils.instantiate(cfg.agents)
    # get the scaler instance and set teh bounds for the sampler if required
    agent.get_scaler(workspace_manager.scaler)
    agent.set_bounds(workspace_manager.scaler)
    agent.load_pretrained_model(model_store_path)

    # agent.sigma_max = param_config['sigma_max']
    # agent.sigma_min = param_config['sigma_min']
    # agent.rho = param_config['rho']
    if render_workspace:
        workspace_manager.render = True
    else:
        workspace_manager.render = False

    if param_config['test_classifier_free_guidance']:
        # load the classifier free wrapper
        agent.model = ClassifierFreeSampleModel(agent.model, cond_lambda=param_config['cond_lambda'])
    elif param_config['cond_lambda'] != 1:
        agent.model = ClassifierFreeSampleModel(agent.model, cond_lambda=param_config['cond_lambda'])
    if param_config['use_kde']:
        agent.use_kde = True
    
    if test_single_variant:
        # workspace_manager.eval_n_steps = 500
        workspace_manager.eval_n_times = param_config['num_runs']
        
        workspace_manager.test_agent(
            agent, 
            log_wandb=param_config['log_wandb'],
            new_sampler_type=param_config['sampler_type'],
            n_inference_steps=param_config['n_inference_steps'],
            get_mean=param_config['get_mean'],
            extra_args={
                'keep_last_actions': param_config['keep_last_action'],
                's_churn': param_config['s_churn'],
                'use_scaler': param_config['use_scaler'],
                's_min': param_config['s_min']
                },
            noise_scheduler=param_config['noise_scheduler']
            )
    if test_all_samplers:
        workspace_manager.compare_sampler_types(
            agent, 
            num_runs=param_config['num_runs'], 
            num_steps_per_run=280, 
            log_wandb=False, 
            n_inference_steps=param_config['n_inference_steps'],
            get_mean=None,
            store_path=model_store_path
        )
    if compare_samplers_over_diffent_steps:
        workspace_manager.compare_sampler_types_over_n_steps(
            agent, 
            num_runs=param_config['num_runs'],  
            steps_list=  [1, 2, 3, 4, 5, 10, 20, 40, 50], # [5, 10, 50],
            log_wandb=False, 
            get_mean=None,
            store_path=model_store_path,
            extra_args={
                'keep_last_actions': param_config['keep_last_action'],
                's_churn': param_config['s_churn'],
                'use_scaler': param_config['use_scaler'],
                's_min': param_config['s_min']
                }
        )

    if compare_classifier_free_guidance:
        workspace_manager.compare_classifier_free_guidance(
            agent, 
            num_runs=param_config['num_runs'], 
            sampler_type=param_config['sampler_type'],
            num_steps_per_run=280,
            cond_lambda_list=[0, 1, 1.5, 2, 2.5], 
            n_inference_steps=param_config['n_inference_steps'],
            log_wandb=False, 
            get_mean=None,
            store_path=model_store_path,
            extra_args={
                'keep_last_actions': param_config['keep_last_action'],
                's_churn': param_config['s_churn'],
                'use_scaler': param_config['use_scaler'],
                's_min': param_config['s_min']
                }
        )
    if compare_noisy_sampler:
        workspace_manager.compare_noisy_sampler(
            agent,  
            num_runs=param_config['num_runs'],
            num_steps_per_run=280,
            n_inference_steps=param_config['n_inference_steps'],
            log_wandb=False,
            get_mean=None,
            store_path=model_store_path
            )

    if compare_kde:
        workspace_manager.compare_kde_vs_mean_vs_single(
            agent,
            num_runs=param_config['num_runs'], 
            sampler_type=param_config['sampler_type'],
            num_steps_per_run=280,
            n_inference_steps=param_config['n_inference_steps'],
            log_wandb=False, 
            get_mean=None,
            store_path=model_store_path,
            extra_args={
                'keep_last_actions': param_config['keep_last_action'],
                's_churn': param_config['s_churn'],
                'use_scaler': param_config['use_scaler'],
                's_min': param_config['s_min']
                }
            )
    log.info("done")
    wandb.finish()    


if __name__ == "__main__":
    # with cfg trained
    # store_dir = '/home/moritz/code/generative_learning_from_play/logs/block_pushing/sweeps/2023-01-20/22-50-34'
    store_dir = '/home/moritz/code/generative_learning_from_play/logs/block_pushing/sweeps/2023-02-01/15-41-18'
    # store_dir = '/home/moritz/code/generative_learning_from_play/logs/franka_kitchen/runs/2023-01-25/22-33-02'
    # simple goal conditioned
    # store_dir = '/home/moritz/code/generative_learning_from_play/logs/block_pushing/sweeps/2023-02-01/11-26-30'
    # store_dir = '/home/moritz/code/generative_learning_from_play/logs/franka_kitchen/runs/2023-01-25/11-33-19'
    # store_dir = '/home/moritz/code/generative_learning_from_play/logs/block_pushing/sweeps/2023-01-22/17-35-47'
    # samplers = ['euler', 'euler_ancestral', 'dpm', 'ancestral', 'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_2s', 'dpmpp_2s_ancestral', 'ddim']
    # samplers = ['heun', 'lms']
    # samplers = ['dpmpp_2s', 'dpmpp_2s_ancestral']
    # samplers = ['dpmpp_2s']
    # samplers = ['heun']
    # samplers = ['euler', 'heun', 'dpm','dpmpp_2m', 'dpmpp_2s', 'ddim']
    # sampler_type = 'ancestral'
    samplers = ['ddim']
    
    steps = [3]
    for model in os.listdir(store_dir):
        # if 'cond_lambda=1' in model:
        if 'yaml' in model:
            continue
        model_store_path = os.path.join(store_dir, model)
        for sampler in samplers:
            for step in steps:
                    main(model_store_path, sampler_type=sampler, steps=step)
        # model_store_path = '/home/moritz/code/generative_learning_from_play/logs/block_pushing/sweeps/2023-01-20/22-50-34'

    # main(model_store_path)