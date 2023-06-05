import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np

from networks.k_diffusion.utils import SinusoidalPosEmb
from networks.utils import return_activiation_fcn
from networks.mlps import ResidualMLPNetwork, MLPNetwork
from networks.res_layers import TwoLayerPreActivationResNetLinear


class ScoreNetwork(nn.Module):
    """
    Simple multi layer perceptron network for benchmarking the performance of different networks. The model is used in
    several papers and can be used to compare all_plots model performances.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 100,
            num_hidden_layers: int = 1,
            output_dim=1,
            dropout: int = 0,
            activation: str = "ReLU",
            residual_style: bool = True,
            use_norm: bool = False,
            norm_style: str = 'BatchNorm',
            use_spectral_norm: bool = False,
            device: str = 'cuda'
    ):
        super(ScoreNetwork, self).__init__()
        self.network_type = "mlp"
        # set up the network
        if residual_style:
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

    def forward(self, x, action, sigma):

        x = torch.cat([x, action], dim=-1)
        x = self.layers(x)
        return x / sigma

    def get_device(self, device: torch.device):
        self.device = device
        self.layers.to(device)

    def get_params(self):
        return self.layers.parameters()


class TimeScoreNetwork(nn.Module):

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int,
            time_embed_dim: int,
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
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.Mish(),
            nn.Linear(time_embed_dim, time_embed_dim)
        ).to(device)
        
        
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

    def forward(self, x, action, t):
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.embed(t))
        x = torch.cat([x, action, embed], dim=-1)
        x = self.layers(x)
        return x # / marginal_prob_std(t, self.sigma, self.device)[:, None]

    def get_device(self, device: torch.device):
        self.device = device
        self.layers.to(device)

    def get_params(self):
        return self.parameters()


def marginal_prob_std(t, sigma, device):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


class TimeConditionedResidualMLPNetwork(nn.Module):
    """
    Variant of the MLP network, which is time conditioned and adds the time 
    information during every layer using sum. Similar to FiLM conditioning
    """

    def __init__(
            self,
            input_dim: int,
            embed_dim: int,
            hidden_dim: int,
            num_hidden_layers: int = 2,
            output_dim=1,
            dropout: int = 0,
            activation: str = "ReLU",
            use_spectral_norm: bool = False,
            use_norm: bool = False,
            norm_style: str = 'BatchNorm',
            device: str = 'cuda'
    ):
        super(TimeConditionedResidualMLPNetwork, self).__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        ).to(device)
        self.device = device
        # set up the network
        assert num_hidden_layers % 2 == 0

        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation=activation,
                    dropout_rate=dropout,
                    spectral_norm=use_spectral_norm,
                    use_norm=use_norm,
                    norm_style=norm_style
                )
                for i in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.to(self.device)
        # get linear projections for the time embedding
        self.dense_embed_layers = []
        self.dense_embed_layers.append(nn.Linear(embed_dim, hidden_dim).to(device))
        for i in range(1, num_hidden_layers, 2):
            self.dense_embed_layers.append(
                nn.Linear(embed_dim, hidden_dim).to(device)
            )
        self.marginal_prob_std = marginal_prob_std

        self.act = return_activiation_fcn(activation)

    def forward(self, state, action, t):
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.embed(t))
        x = torch.concat([state, action], dim=-1)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx <len(self.layers) -1:
                x += self.dense_embed_layers[idx](embed)
        # normalize output 
        # x = x / self.marginal_prob_std(t)  # TODO where does the sigma come from?
        # TODO Why h = h / self.marginal_prob_std(t)[:, None, None, None]
        return x

    def get_device(self, device: torch.device):
        self.device = device
        self.layers.to(device)

    def get_params(self):
        return self.parameters()
