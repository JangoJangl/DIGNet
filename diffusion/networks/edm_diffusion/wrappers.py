

import numpy as np
import torch
from torch.nn.functional import silu
from omegaconf import DictConfig
import hydra

#----------------------------------------------------------------------------
# based on https://github.com/NVlabs/edm/blob/main/training/networks.py


#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


class VPPrecond(torch.nn.Module):
    def __init__(self,
        state_dim: int,                 # state dimensions
        action_dim: int,                # action dimensions
        model: DictConfig,              # dict to initialize the model using hydra
        use_fp16: bool  = False,        # Execute the underlying model at FP16 precision?
        beta_d: float   = 19.9,         # Extent of the noise level schedule.
        beta_min: float = 0.1,          # Initial slope of the noise level schedule.
        M               = 50,           # number of timesteps in the DDPM formulation.
        epsilon_t       = 1e-5,         # Minimum t-value used during training.
        ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = hydra.utils.instantiate(model)

    def forward(self, state, action, goal, sigma, force_fp32=False, **model_kwargs):
        state = state.to(torch.float32)
        action = action.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and action.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model(state, (c_in * action).to(dtype), goal, c_noise.flatten(), **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * action + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
    def get_params(self):
        return self.model.parameters()

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".
class VEPrecond(torch.nn.Module):
    def __init__(self,
        state_dim: int,                 # state dimensions
        action_dim: int,                # action dimensions
        model: DictConfig,              # dict to initialize the model using hydra
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        sigma_min       = 0.02,         # Minimum supported noise level.
        sigma_max       = 50,           # Maximum supported noise level.
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.model = hydra.utils.instantiate(model)

    def forward(self, state, action, goal, sigma, force_fp32=False, **model_kwargs):
        state = state.to(torch.float32)
        action = action.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model(state, (c_in * action), goal, c_noise.flatten(), **model_kwargs)
        D_x = c_skip * action + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
    def get_params(self):
        return self.model.parameters()
    
#----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".
class iDDPMPrecond(torch.nn.Module):
    def __init__(self,
        state_dim: int,                     # state dimensions
        action_dim: int,                    # action dimensions
        model: DictConfig,                  # dict to initialize the model using hydra
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 50,               # number of timesteps for training
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = hydra.utils.instantiate(model)

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, state, action, goal, sigma, force_fp32=False, **model_kwargs):
        state = state.to(torch.float32)
        action = action.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and action.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        F_x = self.model(state, (c_in * action).to(dtype), goal, c_noise.flatten(),  **model_kwargs)
        D_x = c_skip * action + c_out * F_x.to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)
    
    def get_params(self):
        return self.model.parameters()


#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        state_dim: int,                 # state dimensions
        action_dim: int,                # action dimensions
        model: DictConfig,              # dict to initialize the model using hydra
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = hydra.utils.instantiate(model)
        
    def forward(self, state, action, goal, sigma, force_fp32=False, **model_kwargs):
        state = state.to(torch.float32)
        action = action.to(torch.float32)
        sigma = sigma.to(torch.float32)
        if len(sigma.shape) <= 1:
            sigma = sigma.reshape(-1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(state, (c_in * action), goal, c_noise.flatten(), **model_kwargs)

        D_x = c_skip * action + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
    def get_params(self):
        return self.model.parameters()