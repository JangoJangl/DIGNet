from faulthandler import disable
from functools import partial
import os
import logging
import math
from collections import deque


import einops
from omegaconf import DictConfig
import hydra
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
import vip
from sklearn.neighbors import KernelDensity
import torchvision.transforms as T

from agents.base_agent import BaseAgent
from agents.ema_helper.ema import ExponentialMovingAverage
from agents.scaler.scaler_class import Scaler
# from networks.k_diffusion.sampling import get_sigmas_karras, sample_dpm_2, sample_dpm_2_ancestral, sample_dpm_adaptive, sample_dpm_fast, sample_euler_ancestral, sample_lms, sample_heun, sample_euler
from networks.k_diffusion.gc_sampling import *
import networks.k_diffusion.utils as utils
from networks.k_diffusion.score_gpts import DiffusionGPT

# A logger for this file
log = logging.getLogger(__name__)


class KDiffusionAgent(BaseAgent):

    def __init__(
            self,
            model: DictConfig,
            vision_encoder: DictConfig,
            state_encoder: DictConfig,
            optimization: DictConfig,
            device: str,
            max_train_steps: int,
            eval_every_n_steps: int,
            use_ema: bool,
            goal_conditioned: bool,
            pred_last_action_only: bool,
            rho: float,
            num_sampling_steps: int,
            image_pre_encoded: bool,
            lr_scheduler: DictConfig,
            sampler_type: str,
            sigma_data: float,
            sigma_min: float,
            sigma_max: float,
            sigma_sample_density_type: str,
            sigma_sample_density_mean: float,
            sigma_sample_density_std: float,
            decay: float,
            update_ema_every_n_steps: int,
            window_size: int,
            goal_window_size: int,
            use_kde: bool=False,
    ):
        super().__init__(model, optimization, device, max_train_steps, eval_every_n_steps)
        self.ema_helper = ExponentialMovingAverage(self.model.get_params(), decay, self.device)
        self.use_ema = use_ema
        # self.vision_encoder = hydra.utils.call(vision_encoder)
        self.state_encoder = hydra.utils.call(state_encoder).to(self.device)
        self.vision_encoder = vip.load_vip().eval().to(self.device)
        self.lr_scheduler = hydra.utils.instantiate(
            lr_scheduler,
            optimizer=self.optimizer
        )
        # rho controls how much the step size close to minimum noise levels are shorted for the expanse of steps near
        # sigma max: high rho means we care more about steps near sigma min and vice-versa
        self.rho = rho
        self.gc = goal_conditioned
        self.image_pre_encoded = image_pre_encoded
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        self.sigma_sample_density_mean = sigma_sample_density_mean
        self.sigma_sample_density_std = sigma_sample_density_std
        self.decay = decay
        self.update_ema_every_n_steps = update_ema_every_n_steps
        # get the window size for prediction
        self.window_size = window_size
        self.goal_window_size = goal_window_size
        # bool if the model should only output the last action or all actions in a sequence
        self.pred_last_action_only = pred_last_action_only
        # set up the rolling window contexts
        self.obs_context = deque(maxlen=self.window_size)
        self.goal_context = deque(maxlen=self.goal_window_size)
        # if we use DiffusionGPT we need an action context
        self.action_context = deque(maxlen=self.window_size-1)
        self.que_actions = True
        self.image_transform = transforms = T.Compose([T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()]) # ToTensor() divides by 255
        # use kernel density estimatior
        self.use_kde = use_kde

    def get_scaler(self, scaler: Scaler):
        """
        Define the scaler from the Workspace class used to scale state and output data if necessary
        """
        self.scaler = scaler

    def set_bounds(self, scaler: Scaler):
        """
        Define the bounds for the sampler class
        """
        self.model.min_action = torch.from_numpy(scaler.y_bounds[0, :]).to(self.device)
        self.model.max_action = torch.from_numpy(scaler.y_bounds[1, :]).to(self.device)

    def train_agent(
            self,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader
    ):

        best_test_mse = 1e10
        mean_mse = 1e10
        generator = iter(train_loader)

        for step in tqdm(range(self.max_train_steps)):
            # run a test batch every n steps
            if not self.steps % self.eval_every_n_steps:
                test_mse = []
                for sequences in test_loader:
                    mean_mse = self.evaluate(sequences)
                    test_mse.append(mean_mse)
                avrg_test_mse = sum(test_mse) / len(test_mse)
                log.info("Step {}: Mean test mse is {}".format(step, avrg_test_mse))
                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    self.store_model_weights(self.working_dir)
                    log.info('New best test loss. Stored weights have been updated!')
            try:
                batch_loss = self.train_step(next(generator))
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(train_loader)
                batch_loss = self.train_step(next(generator))

            # log.info loss every 100 steps
            if not self.steps % 1000:
                log.info("Step {}: Mean batch loss mse is {}".format(step, batch_loss))
            wandb.log(
                {
                    "loss": batch_loss,
                    "test_loss": avrg_test_mse
                }
            )
        self.store_model_weights(self.working_dir)
        log.info("Training done!")
    
    def process_input_batch(self, state: dict):
        if self.image_pre_encoded:
            img_enc_1 = state["cam_1_enc"].to(self.device)
            img_enc_2 = state["cam_2_enc"].to(self.device)
            if self.gc:
                goal_enc_1 = state['goal_1_enc'].to(self.device)
                goal_enc_2 = state['goal_2_enc'].to(self.device)
            else:
                goal_enc_1 = None
                goal_enc_2 = None
            state_enc = state['sin_cos_state'].to(self.device)
            action = state['delta-joint-state'].to(self.device) + state['joint-state'].to(self.device)
            gripper = state['des_gripper_width'].to(self.device)
        else:
            preprocessed_image_1 = self.image_transform(state["cam_1"]).reshape(-1, 3, 224, 224).to(self.device)
            preprocessed_image_2 = self.image_transform(state["cam_1"]).reshape(-1, 3, 224, 224).to(self.device)
            img_enc_1 = self.vision_encoder(preprocessed_image_1)
            img_enc_2 = self.vision_encoder(preprocessed_image_2)
            preprocessed_image_1 = self.image_transform(state["goal_1"]).reshape(-1, 3, 224, 224).to(self.device)
            preprocessed_image_2 = self.image_transform(state["goal_2"]).reshape(-1, 3, 224, 224).to(self.device)
            if self.gc:
                goal_enc_1 = self.vision_encoder(preprocessed_image_1)
                goal_enc_2 = self.vision_encoder(preprocessed_image_2)
            else:
                goal_enc_1 = None
                goal_enc_2 = None
            state_enc = state['sin_cos_state'].to(self.device)
            action = state['delta-joint-state'].to(self.device) + state['joint-state'].to(self.device)
            gripper = state['des_gripper_width'].to(self.device)
        
        action = torch.cat((action, gripper), dim=-1).to(torch.float32)
        enc_states = torch.cat((img_enc_1, img_enc_2), dim=-1).to(self.device)
        image_embedding = self.state_encoder(enc_states)
        state_embedding = torch.concat((image_embedding, state_enc), dim=-1).to(torch.float32)
        if self.gc:
            goal_states = torch.cat((goal_enc_1, goal_enc_2), dim=-1).to(self.device)
            goal_embedding = self.state_encoder(goal_states).to(torch.float32)
        else:
            goal_embedding = None
        return state_embedding, action, goal_embedding
            

    def train_step(self, state: dict) -> float:
        self.model.train()
        self.model.training = True
        # scale data if necessary, otherwise the scaler will return unchanged values
        state_embedding, action, goal_embedding = self.process_input_batch(state)
        # set up the noise
        noise = torch.randn_like(action)
        # define the sigma values
        sigma = self.make_sample_density()(shape=(len(action),), device=self.device)

        loss = self.model.loss(state_embedding, action, goal_embedding, noise, sigma)
        # Before the backward pass, zero all the network gradients
        self.optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # clip the gradient
        # model_grad_norms = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
        # Calling the step function to update the parameters
        self.optimizer.step()
        self.lr_scheduler.step()
        self.steps += 1
        # update the ema model
        if self.steps % self.update_ema_every_n_steps == 0:
            self.ema_helper.update(self.model.parameters())
        return loss

    @torch.no_grad()
    def evaluate(self, state: dict) -> float:
        # scale data if necessary, otherwise the scaler will return unchanged values
        # scale data if necessarry, otherwise the scaler will return unchanged values
        state_embedding, action, goal_embedding = self.process_input_batch(state)
        
        total_mse = 0.0
        # use the EMA model variant
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()
        self.model.training = False
        # get the sigma distribution for sampling based on Karras et al. 2022
        sigmas = get_sigmas_karras(self.num_sampling_steps, self.sigma_min, self.sigma_max, self.rho, self.device)

        x = torch.randn_like(action) * self.sigma_max
        # generate the action based on the chosen sampler type 
        x_0 = self.sample_loop(sigmas, x, state_embedding, goal_embedding, self.sampler_type)
        # x_0 = self.scaler.clip_action(x_0)
            
        mse = nn.functional.mse_loss(x_0, action, reduction="none")
        total_mse += mse.mean().item()
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        return total_mse
    
    def reset(self):
        
        self.obs_context.clear()
        self.action_context.clear()

    @torch.no_grad()
    def predict(
        self, 
        state: torch.Tensor, 
        extra_args=None, 
    ) -> torch.Tensor:
        img_enc_1 = state["cam_1_enc"].to(self.device)
        img_enc_2 = state["cam_2_enc"].to(self.device)
        if self.gc:
            goal_enc_1 = state['goal_1_enc'].to(self.device)
            goal_enc_2 = state['goal_2_enc'].to(self.device)
        else:
            goal_enc_1 = None
            goal_enc_2 = None
        state_enc = state['sin_cos_state'].to(self.device)

        enc_states = torch.cat((img_enc_1, img_enc_2), dim=-1)
        image_embedding = self.state_encoder(enc_states)
        state_embedding = torch.concat((image_embedding, state_enc), dim=-1).to(torch.float32)
        if self.gc:
            goal_states = torch.cat((goal_enc_1, goal_enc_2), dim=-1)
            goal_embedding = self.state_encoder(goal_states).to(torch.float32)
        else:
            goal_embedding = None
        # manage the case for sequence models with multiple obs
        if self.window_size > 1:
            # rearrange from 2d -> sequence
            self.obs_context.append(state) # this automatically manages the number of allowed observations
            input_state = torch.stack(tuple(self.obs_context), dim=1)
        else:
            input_state = state

        # get the sigma distribution 
        sigmas = get_sigmas_exponential(self.n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        
        # depending if we use a single sample or the mean over n samples
        x = torch.randn((1, 1, 8), device=self.device) * self.sigma_max
            # check if we need to get thew hole action context for the DiffusionGPT model variant
            
        if len(self.action_context) > 0:  
            previous_a = torch.cat(tuple(self.action_context), dim=1)
            x = torch.cat([previous_a, x], dim=1)

        # repeat the state goal n times to get mean prediction
        x_0 = self.sample_loop(sigmas, x, state_embedding, goal_embedding, self.sampler_type, extra_args)
        
        # only get the last action if we use a sequence model 
        if x_0.size()[1] > 1 and len( x_0.size()) == 3:
            x_0 = x_0[:, -1, :]
        # if we predict a sequence we only want the last predicted action of our transformer model
        
        # if self.que_actions or self.pred_last_action_only:
        if len(x_0.shape) == 2:
            x_0 = einops.rearrange(x_0, 'b d -> b 1 d')
        self.action_context.append(x_0)
        return x_0
    
    def sample_loop(
        self, 
        sigmas, 
        x_t: torch.Tensor,
        state: torch.Tensor, 
        goal: torch.Tensor, 
        sampler_type: str,
        extra_args={}, 
        ):
        """
        Main method to generate samples depending on the chosen sampler type
        """
        # get the s_churn 
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        # extra_args.pop('s_churn', None)
        # extra_args.pop('use_scaler', None)
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x:extra_args[x] for x in keys}
        else:
            reduced_args = {}
        
        if use_scaler:
            scaler = self.scaler
        else:
            scaler=None
        # ODE deterministic
        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min, disable=True)
        # ODE deterministic 
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True) 
        # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas), disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            # x_0 = sample_euler_visualization(self.model, state, x_t, goal, sigmas, self.scaler, self.working_dir, disable=True, extra_args={'keep_last_actions': True})
        else:
            raise ValueError('desired sampler type not found!')
        return x_0    

    def load_pretrained_model(self, weights_path: str, **kwargs) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """
        self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        self.ema_helper = ExponentialMovingAverage(self.model.get_params(), self.decay, self.device)
        log.info('Loaded pre-trained model parameters')

    def store_model_weights(self, store_path: str) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        torch.save(self.model.state_dict(), os.path.join(store_path, "non_ema_model_state_dict.pth"))
    
    @staticmethod
    def get_diffusion_inverse_chain(callback_dict: dict):
        current_action = callback_dict['x']
        current_sigma = callback_dict['sigma']
    
    
    @torch.no_grad()
    def visualize_ode(
        self, 
        state: torch.Tensor, 
        goal, 
        get_mean=1000, 
        new_sampling_steps=None,
        noise_scheduler=None,
    ) -> torch.Tensor:
       
        if self.use_kde:
            get_mean = 100 if get_mean is None else get_mean

        # same with the number of sampling steps 
        if new_sampling_steps is not None:
            n_sampling_steps = new_sampling_steps
        else:
            n_sampling_steps = self.num_sampling_steps
        
        # scale data if necessary, otherwise the scaler will return unchanged values
        state = self.scaler.scale_input(state)
        goal = self.scaler.scale_input(goal)
        # manage the case for sequence models with multiple obs
        if self.window_size > 1:
            # rearrange from 2d -> sequence
            self.obs_context.append(state) # this automatically manages the number of allowed observations
            input_state = torch.stack(tuple(self.obs_context), dim=1)
        else:
            input_state = state
            
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()
        if noise_scheduler is not None:
            if noise_scheduler == 'karras':
                sigmas = get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, self.rho, self.device)
            elif noise_scheduler == 'exponential':
                sigmas = get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
            elif noise_scheduler == 'vp':
                sigmas = get_sigmas_vp(n_sampling_steps, device=self.device)
            elif noise_scheduler == 'linear':
                sigmas = get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
            elif noise_scheduler == 'cosine_beta':
                sigmas = cosine_beta_schedule(n_sampling_steps, device=self.device)
            elif noise_scheduler == 've':
                sigmas = get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
            elif noise_scheduler == 'iddpm':
                sigmas = get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        else:
            sigmas = get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        
        x = torch.randn((len(input_state)*get_mean,self.scaler.y_bounds.shape[1]), device=self.device) * self.sigma_max

        # repeat the state goal n times to get mean prediction
        state_rpt = torch.repeat_interleave(input_state, repeats=get_mean, dim=0)
        goal_rpt = torch.repeat_interleave(goal, repeats=get_mean, dim=0)
        
        # x = torch.repeat_interleave(x, repeats=get_mean, dim=0)
        # generate the action based on the chosen sampler type
        sampled_actions = [x] 
        x_0 = x
        for i in range(n_sampling_steps):
            simgas_2 = sigmas[i:(i+2)]
            x_0 = sample_euler(self.model, state_rpt, x_0, goal_rpt, simgas_2, disable=True)
            if self.use_kde:
                if len(x_0.shape) == 3:
                    x_kde = einops.rearrange(x_0, 'b t d -> (b t) d').detach().cpu()
                else:
                    x_kde = x_0.detach().cpu()
                kde = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(x_kde)
                # kde_prob = np.exp(kde.score_samples(x_kde))
                kde_prob = kde.score(x_kde)
                max_index = kde_prob.argmax(axis=0)
                x_all = x_0
                x_0 = x_0[max_index]
                if len(x_0.shape) == 1:
                    x_0 = x_0.unsqueeze(0)
            else:
                sampled_actions.append(x_0)

        # x_loop = sample_euler(self.model, state_rpt, x, goal_rpt, sigmas, disable=True)
        # .clamp_(self.scaler.tensor_y_bounds[0, :] - 0.1*self.scaler.tensor_y_bounds[0, :],
        #                    self.scaler.tensor_y_bounds[1, :] * 1.1)
        # restore the previous model parameters
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        model_pred = self.scaler.inverse_scale_output(x_0)

        return sampled_actions
        
    def make_sample_density(self):

        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)
        raise ValueError('Unknown sample density type')