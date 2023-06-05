from faulthandler import disable
from functools import partial
import os
import logging
from collections import deque

from omegaconf import DictConfig
import hydra
import torch
import torch.nn as nn
import numpy as np
import tqdm
import wandb
import einops 

from agents.base_agent import BaseAgent
from agents.ema_helper.ema import ExponentialMovingAverage
from agents.scaler.scaler_class import Scaler
from networks.edm_diffusion.sampling import *

# A logger for this file
log = logging.getLogger(__name__)


class DiffusionAgent(BaseAgent):

    def __init__(
            self,
            model: DictConfig,
            optimization: DictConfig,
            diffusion_loss: DictConfig,
            lr_scheduler: DictConfig,
            device: str,
            goal_conditioned: bool,
            pred_last_action_only: bool,
            max_train_steps: int,
            eval_every_n_steps: int,
            use_ema: bool,
            rho: float,
            C_1: float,
            C_2: float,
            alpha: float,
            S_churn: float,
            S_min: float,
            S_max: float,
            S_noise: float,
            epsilon_s: float,
            epsilon_t: float,
            use_edm_sampler: bool,
            solver_type: str,
            discretization: str,
            schedule: str,
            scaling: str,
            num_sampling_steps: int,
            sigma_data: float,
            sigma_min: float,
            sigma_max: float,
            decay: float,
            update_ema_every_n_steps: int,
            window_size: int,
            goal_window_size: int
    ):
        super().__init__(model, optimization, device, max_train_steps, eval_every_n_steps)
        self.ema_helper = ExponentialMovingAverage(self.model.get_params(), decay, self.device)
        self.use_ema = use_ema
        self.gc = goal_conditioned
        # get the window size for prediction
        self.window_size = window_size
        self.goal_window_size = goal_window_size
        # bool if the model should only output the last action or all actions in a sequence
        self.pred_last_action_only = pred_last_action_only
        # set up the rolling window contexts
        self.obs_context = deque(maxlen=self.window_size)
        self.goal_context = deque(maxlen=self.goal_window_size)
        # if we use DiffusionGPT we need an action context
        if not self.pred_last_action_only:
            self.action_context = deque(maxlen=self.window_size-1)
            self.que_actions = True
        else:
            self.que_actions = False
        self.lr_scheduler = hydra.utils.instantiate(
            lr_scheduler,
            optimizer=self.optimizer
        )
        # init the loss class
        self.loss = hydra.utils.instantiate(diffusion_loss)
        # rho controls how much the step size close to minimum noise levels are shorted for the expanse of steps near
        # sigma max: high rho means we care more about steps near sigma min and vice-versa
        self.rho = rho
        self.num_sampling_steps = num_sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.C_1 = C_1
        self.C_2 = C_2
        self.alpha = alpha
        self.S_churn = S_churn
        self.S_min = S_min
        if S_max == 0:
            self.S_max = float('inf')
        else:
            self.S_max = S_max
        self.S_noise = S_noise
        self.epsilon_s = epsilon_s
        self.epsilon_t = epsilon_t
        self.decay = decay
        self.update_ema_every_n_steps = update_ema_every_n_steps
        # define the evaluation sampler
        self.solver_type = solver_type
        self.use_edm = use_edm_sampler
        self.discretization = discretization
        self.schedule = schedule
        self.scaling = scaling

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

        for step in tqdm.tqdm(range(self.max_train_steps)):
            # run a test batch every n steps
            if not self.steps % self.eval_every_n_steps:
                test_mse = []
                for tensors in test_loader:
                    state, action, mask, goal = tensors
                    mean_mse = self.evaluate(state, action, goal)
                    test_mse.append(mean_mse)
                avrg_test_mse = sum(test_mse) / len(test_mse)
                log.info("Step {}: Mean test mse is {}".format(step, avrg_test_mse))
                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    self.store_model_weights(self.working_dir)
                    log.info('New best test loss. Stored weights have been updated!')
            try:
                batch_loss = self.train_step(*next(generator))
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(train_loader)
                batch_loss = self.train_step(*next(generator))

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

    def train_step(self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor, goal: torch.Tensor) -> float:

        # scale data if necessary, otherwise the scaler will return unchanged values
        state = self.scaler.scale_input(state)
        action = self.scaler.scale_output(action)
        goal = self.scaler.scale_input(goal)
        
        if self.pred_last_action_only:
            action = action[:, -1, :]
            action = einops.rearrange(action, 'b d -> b 1 d')# only train the last timestep 
            
        self.model.train()
        
        # Compute the loss.
        self.optimizer.zero_grad()
        loss = self.loss(self.model, state, action, goal, augment_pipe=False)

        loss = loss.mean()
        # Before the backward pass, zero all the network gradients
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Calling the step function to update the parameters
        self.optimizer.step()
        self.steps += 1
        self.lr_scheduler.step()
        # update the ema model
        if self.steps % self.update_ema_every_n_steps == 0:
            self.ema_helper.update(self.model.parameters())
        return loss

    @torch.no_grad()
    def evaluate(
            self, state: torch.tensor, action: torch.tensor,  goal: torch.Tensor
    ) -> float:
        # scale data if necessary, otherwise the scaler will return unchanged values
        state = self.scaler.scale_input(state)
        action = self.scaler.scale_output(action)
        if self.pred_last_action_only:
            action = action[:, -1, :]
            action = einops.rearrange(action, 'b d -> b 1 d')# only train the last timestep 
        goal = self.scaler.scale_input(goal)
        
        total_mse = 0.0
        # use the EMA model variant
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()

        x_t = torch.randn_like(action).to(self.device)
        # generate the action based on the chosen sampler type 
        x_0 = self.sample_loop(
            x_t,
            state, 
            goal,
            self.use_edm,
            self.solver_type,
            self.discretization,
            self.schedule,
            self.scaling,
            self.num_sampling_steps
        )

        x_0 = x_0.clamp_(self.scaler.tensor_y_bounds[0, :] - 0.15*self.scaler.tensor_y_bounds[0, :],
                         self.scaler.tensor_y_bounds[1, :] * 1.15)
        
        mse = nn.functional.mse_loss(x_0, action, reduction="none")
        total_mse += mse.mean().item()

        # restore the previous model parameters
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        return total_mse

    @torch.no_grad()
    def predict(
            self, 
            state: torch.Tensor, 
            goal: torch.Tensor, 
            sampler_dict: dict = {},
            get_mean=None
    ) -> torch.Tensor:
       
        # get the parameters for the sampler from the sampler_dict if they do not exist use the default ones 
        solver = sampler_dict['solver_type'] if 'solver_type' in sampler_dict else self.solver_type
        discretization = sampler_dict['discretization'] if 'discretization' in sampler_dict else self.discretization
        schedule = sampler_dict['schedule'] if 'schedule' in sampler_dict else self.schedule
        scaling = sampler_dict['scaling'] if 'scaling' in sampler_dict else self.scaling
        n_sampling_steps = sampler_dict['num_sampling_steps'] if 'num_sampling_steps' in sampler_dict else self.num_sampling_steps
        use_edm_sampler = sampler_dict['use_edm'] if 'use_edm' in sampler_dict else self.use_edm

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
        
        # adept for time sequence if chosen
        if self.window_size > 1:
            goal = einops.rearrange(goal, 'b d -> 1 b d')
            # depending if we use a single sample or the mean over n samples
            if get_mean is not None:
                x = torch.randn((len(input_state)*get_mean, 1, self.scaler.y_bounds.shape[1]), device=self.device) 
            else:
                x = torch.randn((len(input_state), 1, self.scaler.y_bounds.shape[1]), device=self.device) 
                # check if we need to get thew hole action context for the DiffusionGPT model variant
                if self.pred_last_action_only:
                    pass
                elif self.que_actions:
                    if len(self.action_context) > 0:
                        previous_a = torch.cat(tuple(self.action_context), dim=1)
                        x = torch.cat([previous_a, x], dim=1)
                        
        else:
            if get_mean is not None:
                x = torch.randn((len(input_state)*get_mean, self.scaler.y_bounds.shape[1]), device=self.device) 
        
            else:
                x = torch.randn((len(input_state), self.scaler.y_bounds.shape[1]), device=self.device) 
        
        
        # generate the action based on the chosen sampler type
        x_0 = self.sample_loop(
            x,
            input_state, 
            goal,
            use_edm_sampler,
            solver,
            discretization,
            schedule,
            scaling,
            n_sampling_steps
            )

        # scale the final output
        x_0 = x_0.clamp_(self.scaler.tensor_y_bounds[0, :] - 0.15*self.scaler.tensor_y_bounds[0, :],
                         self.scaler.tensor_y_bounds[1, :] * 1.15)
        # restore the previous model parameters
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        # only get the last action if we use a sequence model 
        if x_0.size()[1] > 1 and len( x_0.size()) ==3:
            x_0 = x_0[:, -1, :]
            
        model_pred = self.scaler.inverse_scale_output(x_0)
        
        if self.que_actions:
            if len(model_pred.shape) == 2:
                model_pred = einops.rearrange(model_pred, 'b d -> b 1 d')
            self.action_context.append(model_pred)

        return model_pred
    
    def sample_loop(
        self, 
        x_t: torch.Tensor,    
        state: torch.Tensor,            # current state vector
        goal: torch.Tensor,             # goal conditioning
        use_edm_sampler: bool,          # use the EDM sampler or the standard one              
        solver: str,                    # 'euler', 'heun'
        discretization: str,            # 'vp', 've', 'iddpm', 'edm'
        schedule: str,                  # 'vp', 've', 'linear'
        scaling: str,                   # 'vp', 'none'
        num_sampling_steps: int = 50,  # number of sampling steps
        ) -> torch.Tensor:
        """
        Main method to generate samples depending on the chosen sampler type
        """
        if use_edm_sampler:
            x_0 = edm_sampler(
                self.model,
                state,
                x_t,
                goal,
                num_steps=num_sampling_steps,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
                S_churn=self.S_churn,
                S_min=self.S_min,
                S_max=self.S_max,
                S_noise=self.S_noise 
            )
        else:
            x_0 = ablation_sampler(
                self.model,
                state,
                x_t,
                goal,
                num_steps=num_sampling_steps,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
                solver=solver,
                discretization=discretization,
                schedule=schedule,
                scaling=scaling,
                epsilon_s=self.epsilon_s,
                C_1=self.C_1,
                C_2=self.C_2,
                M=self.num_sampling_steps,
                alpha=self.alpha,
                S_churn=self.S_churn,
                S_min=self.S_min,
                S_max=self.S_max,
                S_noise=self.S_noise 
            )
        return x_0    

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
