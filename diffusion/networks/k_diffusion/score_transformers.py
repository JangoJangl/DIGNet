
from multiprocessing.sharedctypes import Value
import einops

import torch
import hydra
from torch import DictType, nn
from .utils import append_dims
from omegaconf import DictConfig

from networks.mlps import *
from networks.k_diffusion.utils import PositionalEncoding, InputEncoder, TEncoder
from networks.utils import CrossAttention, LayerNorm
from networks.diffusion.diffusion_models import LinearCrossAttentionLayer



class GCTransformerTimeScoreNetwork(nn.Module):
    
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            latent_dim: int,
            embed_fn: DictConfig,
            ff_dim: int,
            mlp: DictConfig,
            cond_mask_prob: float, 
            goal_seq_len: int,
            obs_seq_len: int,
            num_trans_layers: int = 1,
            num_heads: int = 4,
            dropout: int = 0.1,
            activation: str = "ReLU",
            device: str = 'cuda'
    ):
        super(GCTransformerTimeScoreNetwork, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.activation = activation
        self.ff_dim = ff_dim
        self.cond_mask_prob = cond_mask_prob
        # method returns an sigma embedding depemnding on the chosen type 'GaussianFourier' 'FourierFeatures' or 'Sinusoidal'
        self.sigma_emb = nn.Linear(1, self.latent_dim)
        self.embed_t = hydra.utils.call(embed_fn)
        self.time_embed_dim = embed_fn.time_embed_dim
        #  Gaussian random feature embedding layer for time
        # get positional encoding for transformer
        # self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.obs_seq_len = obs_seq_len
        self.goal_seq_len = goal_seq_len
        seq_size = goal_seq_len + obs_seq_len
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, self.latent_dim))
        # next we need linear models to transform the inputs to the same dimension
        self.obs_encoder = nn.Linear(self.obs_dim, self.latent_dim)
        # self.sigma_encoder = nn.Linear(self.sig)
        # we also need one for the noisy action to put it same latent dimensions as the observation sequence
        self.action_encoder = nn.Linear(self.action_dim, self.latent_dim)
        # define the transformer layer and stack it n times
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.latent_dim*2,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer,
            num_layers=num_trans_layers
        )
        
        self.mlp = hydra.utils.instantiate(mlp)
        self.act = return_activiation_fcn(activation)

    def forward(self, state, action, goal, sigma, return_variance=False, uncondiditonal=False):
        
        # given the probability in self.cond_mask_prob we randomly zero out the goal during training
        if self.training:
            goal = self.mask_cond(goal)
        if uncondiditonal:
            goals = torch.zeros_like(goals).to(self.device)  
        
        # combine state vectors to a sequence
        states = torch.cat([goal, state], dim=1)
        # states = state
        # embed the states sequence
        state_emb = self.obs_encoder(states)
        # encode the current time step infor given the sigma value
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, 'b -> b 1')
        
        # t_embed = self.sigma_emb(sigmas)
        # if len(states.shape) == 3:
        #    sigmas = einops.rearrange(sigmas, 'b d -> b 1 d')
        # other embedding for signa
        emb_sigma = self.embed_t(sigmas)
        # if self.training:
        emb_sigma = einops.repeat(emb_sigma, 'b 1 d -> b (1 t) d', t=state.shape[1])
        # else:
        #    emb_sigma = emb_sigma[:, -1]
        # action_embed = self.action_encoder(action)
        #  combine the input with the positional encoding and feed it into the transformer
        # trans_input = torch.cat([t_embed, state_emb, action_embed], dim=1)
        # add the information about position of the input
        b, t, d = state.size()
        position_embeddings = self.pos_emb[
            :, :(t + self.goal_seq_len), :
        ]
        input_seq = state_emb + position_embeddings
        
        # next run it through the transformer architecture
        output_seq = self.seqTransEncoder(input_seq) 
        # use the mlp to output the next action only the last output is needed
        # context = einops.rearrange(output_seq[:, -1], 'b d -> b 1 d')
        # if self.training:
        context = output_seq[:, -(action.shape[1]):]
        #else:
        #    context = output_seq[:, -1]
        # if self.training:
        output = self.mlp(torch.concat([emb_sigma, action, context], dim=-1))
        # else:
        #    action = action[:, -1]
        #    output = self.mlp(torch.concat([emb_sigma, action, context], dim=-1))
        # output = self.mlp(output_seq[:, -(action.shape[1]):])
        if return_variance:
            return output, output.flatten(1).mean(1)
        else:
            return output
    
    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob)# .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
    

class GCTransformerTimeScoreNetwork2(nn.Module):
    
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            latent_dim: int,
            embed_fn: DictConfig,
            ff_dim: int,
            mlp: DictConfig,
            cond_mask_prob: float, 
            goal_seq_len: int,
            obs_seq_len: int,
            num_trans_layers: int = 1,
            num_heads: int = 4,
            dropout: int = 0,
            activation: str = "ReLU",
            device: str = 'cuda'
    ):
        super(GCTransformerTimeScoreNetwork2, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.drop = nn.Dropout(dropout)
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.activation = activation
        self.ff_dim = ff_dim
        self.cond_mask_prob = cond_mask_prob
        #  Gaussian random feature embedding layer for time
        self.embed = hydra.utils.call(embed_fn)
        self.time_embed_dim = embed_fn.time_embed_dim
        # get positional encoding for transformer
        self.obs_seq_len = obs_seq_len
        self.goal_seq_len = goal_seq_len
        seq_size = goal_seq_len + obs_seq_len + 1
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, self.latent_dim))
        # self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        # next we need linear models to transform the inputs to the same dimension
        self.obs_encoder = InputEncoder(self.obs_dim, self.latent_dim)
        #
        self.t_encoder = InputEncoder(self.obs_dim, self.latent_dim)
        # we also need one for the noisy action to put it same latent dimensions as the observation sequence
        self.action_encoder = InputEncoder(self.action_dim, self.latent_dim)
        # define the transformer layer and stack it n times
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=num_trans_layers)
        
        # adapt the mlp config
        mlp.input_dim = mlp.input_dim + 1
        self.mlp = hydra.utils.instantiate(mlp)
        self.act = return_activiation_fcn(activation)

    def forward(self, state, action, goal, sigma, return_variance=False):
        
        # given the probability in self.cond_mask_prob we randomly zero out the goal during training
        if self.training:
            goal = self.mask_cond(goal)
        # combine state vectors to a sequence
        states = torch.cat([state, goal], dim=1)
        # states = state
        # encode the current time step infor given the sigma value
        t = sigma.log() / 4 
        t_embed = self.embed(t)
        # t_embed = self.embed(t)
        t_embed = einops.rearrange(t_embed, 'b d -> b 1 d')
        #  combine the input with the positional encoding and feed it into the transformer
        # add the information about position of the input
        # state_obs = self.sequence_pos_encoder(state_emb) 
        # get the action embedding
        state_emb = self.drop(self.obs_encoder(states))
        action_embed = self.drop(self.action_encoder(action))
        # get the input sequence
        position_embeddings = self.pos_emb[
            :, :(self.obs_seq_len + self.goal_seq_len), :
        ]
        input_seq = state_emb + position_embeddings
        # we want to predict the action of the last state 
        action_embed = action_embed + position_embeddings[:, -1, :]
        
        input_seq = torch.cat([t_embed, state_emb, action_embed], dim=1)
        # next run it through the transformer architecture
        output_seq = self.seqTransEncoder(input_seq)
        # use the mlp to output the next action only the last output is needed
        
        output = self.mlp(output_seq[:, -(action.shape[1]):], t)
        if return_variance:
            return output, output.flatten(1).mean(1)
        else:
            return output
    
    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob)# .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond


class GCTransformerTimeScoreNetwork3(nn.Module):
    # Variant, where we switch state goal encoding
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            latent_dim: int,
            embed_fn: DictConfig,
            ff_dim: int,
            mlp: DictConfig,
            cond_mask_prob: float, 
            goal_seq_len: int,
            obs_seq_len: int,
            num_trans_layers: int = 1,
            num_heads: int = 4,
            dropout: int = 0.1,
            activation: str = "ReLU",
            device: str = 'cuda'
    ):
        super(GCTransformerTimeScoreNetwork3, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.activation = activation
        self.ff_dim = ff_dim
        self.cond_mask_prob = cond_mask_prob
        # method returns an sigma embedding depemnding on the chosen type 'GaussianFourier' 'FourierFeatures' or 'Sinusoidal'
        self.sigma_emb = nn.Linear(1, self.latent_dim)
        self.embed_t = hydra.utils.call(embed_fn)
        self.time_embed_dim = embed_fn.time_embed_dim
        #  Gaussian random feature embedding layer for time
        # get positional encoding for transformer
        # self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.obs_seq_len = obs_seq_len
        self.goal_seq_len = goal_seq_len
        seq_size = goal_seq_len + obs_seq_len
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, self.latent_dim))
        # next we need linear models to transform the inputs to the same dimension
        self.obs_encoder = nn.Linear(self.obs_dim, self.latent_dim)
        # we also need one for the noisy action to put it same latent dimensions as the observation sequence
        self.action_encoder = nn.Linear(self.action_dim, self.latent_dim)
        # define the transformer layer and stack it n times
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer,
            num_layers=num_trans_layers
        )
        
        self.mlp = hydra.utils.instantiate(mlp)
        self.act = return_activiation_fcn(activation)

    def forward(self, state, action, goal, sigma, return_variance=False):
        
        # given the probability in self.cond_mask_prob we randomly zero out the goal during training
        if self.training:
            goal = self.mask_cond(goal)
        # combine state vectors to a sequence
        states = torch.cat([goal, state], dim=1)
        # states = state
        # embed the states sequence
        state_emb = self.drop(self.obs_encoder(states))
        # encode the current time step infor given the sigma value
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, 'b -> b 1')
        t_embed = self.sigma_emb(sigmas)
        if len(states.shape) == 3:
            t_embed = einops.rearrange(t_embed, 'b d -> b 1 d')
        # other embedding for signa
        emb_sigma = self.embed_t(sigmas)
        
        
        # action_embed = self.action_encoder(action)
        #  combine the input with the positional encoding and feed it into the transformer
        # trans_input = torch.cat([t_embed, state_emb, action_embed], dim=1)
        # add the information about position of the input
        position_embeddings = self.pos_emb[
            :, :(self.obs_seq_len + self.goal_seq_len), :
        ]
        input_seq = state_emb + position_embeddings
        
        input_seq = torch.cat([t_embed, input_seq], dim=1)
        # next run it through the transformer architecture
        output_seq = self.seqTransEncoder(input_seq) 
        # use the mlp to output the next action only the last output is needed
        # context = einops.rearrange(output_seq[:, -1], 'b d -> b 1 d')
        context = output_seq[:, -(action.shape[1]):]
        output = self.mlp(torch.concat([emb_sigma, action, context], dim=-1))
        # output = self.mlp(output_seq[:, -(action.shape[1]):])
        if return_variance:
            return output, output.flatten(1).mean(1)
        else:
            return output
    
    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob)# .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond


