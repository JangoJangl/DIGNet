import math

import torch
import hydra
from torch import nn
from torch.nn import functional as F
import einops

from networks.mlps import *
from networks.diffusion.utils import SinusoidalPosEmb



class TimeScoreNetwork(nn.Module):

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int,
            goal_conditioning: bool,
            time_embed_dim: int,
            embed_fn: DictConfig,
            num_hidden_layers: int = 1,
            output_dim=1,
            dropout: int = 0,
            activation: str = "ReLU",
            model_style: str = 'residual',
            use_norm: bool = False,
            norm_style: str = 'BatchNorm',
            use_spectral_norm: bool = False,
            device: str = 'cuda'
    ):
        super(TimeScoreNetwork, self).__init__()
        self.network_type = "mlp"
        #  Gaussian random feature embedding layer for time
        self.embed = hydra.utils.call(embed_fn)
        self.goal_conditioning = goal_conditioning
        if self.goal_conditioning:
            input_dim = time_embed_dim + 2 * obs_dim + action_dim
        else:
            input_dim = time_embed_dim + obs_dim + action_dim
        # set up the network
        if model_style == 'residual':
            self.layers = ResidualMLPNetwork(
                input_dim,
                hidden_dim,
                num_hidden_layers,
                output_dim,
                dropout,
                activation,
                use_spectral_norm,
                use_norm,
                norm_style,
                device
            ).to(device)
        else:
            self.layers = MLPNetwork(
                input_dim,
                hidden_dim,
                num_hidden_layers,
                output_dim,
                dropout,
                activation,
                use_spectral_norm,
                device
            ).to(device)

        # build the activation layer
        self.act = return_activiation_fcn(activation)
        self.device = device
        self.sigma = None

    def forward(self, state, action, goal, sigma, return_variance=False):
        # Obtain the Gaussian random feature embedding for t

        embed = self.embed(sigma)
        if embed.shape[0] != state.shape[0]:
            embed = einops.repeat(embed, 'b d -> (b n) d', n=state.shape[0])
        if len(state.shape) == 3:
            embed = einops.rearrange(embed, 'b d -> b 1 d')
        # Concatenate the state, action, and time embedding
        if not self.goal_conditioning:
            x = torch.cat([state, action, embed], dim=-1)
        else:
            x = torch.cat([state, action, goal, embed], dim=-1)
        x = self.layers(x)
        if return_variance:
            return x, x.flatten(1).mean(1)
        else:
            return x  # / marginal_prob_std(t, self.sigma, self.device)[:, None]

    def get_device(self, device: torch.device):
        self.device = device
        self.layers.to(device)

    def get_params(self):
        return self.parameters()