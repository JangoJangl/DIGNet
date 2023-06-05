import logging
import math 
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops


logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
        self, 
        n_embd: int,
        n_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int,
        ):
        super().__init__()
        assert n_embd % n_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_heads

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    

class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self, 
        n_embd: int,
        n_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int,
        
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            block_size,
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    

class DiffusionGPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(
        self,
        state_dim: int,
        device: str,
        action_dim: int,
        goal_conditioned: bool,
        embed_dim: int,
        embed_pdrob: float,
        attn_pdrop: float,
        resid_pdrop: float,
        n_layers: int,
        n_heads: int,
        goal_seq_len: int,
        obs_seq_len: int,
        sigma_vocab_size: int
    ):
        super().__init__()
        self.device = device
        self.goal_conditioned = goal_conditioned
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + 2 * obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len + 1
        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim
        # transformer
        self.blocks = nn.Sequential(
            *[Block(
                embed_dim,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                block_size
            ) for _ in range(n_layers)]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(embed_dim)
        
        self.block_size = block_size
        
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        # we need another embedding for the sigma
        self.sigma_emb = nn.Linear(1, embed_dim) 
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        # action pred module 
        self.action_pred = nn.Sequential(
            nn.Linear(embed_dim, 100), 
            nn.GELU(),  
            nn.Linear(100, self.action_dim)
        )
        
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, DiffusionGPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def forward(
        self, 
        states,
        actions, 
        goals,
        sigmas,
        uncond: Optional[bool] =False
    ):  
        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        if len(sigmas.shape) == 1:
            sigmas = einops.rearrange(sigmas, 'b -> b 1')
        # get the sigma embedding
        emb_t = self.sigma_emb(sigmas)
        if emb_t.shape[0] != states.shape[0]:
            emb_t = einops.repeat(emb_t, 'b d -> (b n) d', n=states.shape[0])
        
        # sigmas = einops.rearrange(sigmas, 'b -> b 1')
        if len(states.shape) == 3:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
        
        # define the total length of the input sequence
        seq_length = self.goal_seq_len + 1 + t*2
        # get the beginning of the state action pairs
        '''if not uncond:
            second_half_idx = self.goal_seq_len + 1 
        else:
            second_half_idx = 1 '''
        second_half_idx = self.goal_seq_len + 1 
        # forward the GPT model
        if uncond:
            goals = torch.zeros_like(goals).to(self.device)
        
        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        goal_embed = self.tok_emb(goals)
        action_embed = self.action_emb(actions)
        
        # if not uncond:
        position_embeddings = self.pos_emb[
        :, :(t + self.goal_seq_len), :
        ]  # each position maps to a (learnable) vector
        # note, that the goal states are at the beginning of the sequence since they are available 
        # for all states s_1, .., s_t otherwise the masking would not make sense
        goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:, :])
        # the action get the same position embedding as the related states 
        action_x = self.drop(action_embed + position_embeddings[:, self.goal_seq_len:, :])
        
        # now for the complicated part
        # we need to stack the input in the following order:
        # [sigma_emb, s_g1, .., sg_n, s_1, a_1, s_2, a_2, ..., s_n, a_n]
        # first stack actions and states in the way: [s_1, a_1, s_2, a_2, ..,]
        sa_seq = torch.stack([state_x, action_x], dim=1
                            ).permute(0, 2, 1, 3).reshape(b, 2*t, self.embed_dim)
        
        # next we stack everything together 
        input_seq = torch.cat([emb_t, goal_x, sa_seq], dim=1)
        
        # Note we need to also adept the action masks 
        x = self.blocks(input_seq)
        x = self.ln_f(x)
        
        # now we want the last half of the output
        x = x[:, second_half_idx:, :]
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # we need to check this for inference and adapt the max seq len accord
        if x.size()[1] < 2*self.obs_seq_len:
            x_len = int(x.size()[1]/2)
            x = x.reshape(b, x_len, 2, self.embed_dim).permute(0, 2, 1, 3)
        else:
            x = x.reshape(b, self.obs_seq_len, 2, self.embed_dim).permute(0, 2, 1, 3)
        # get the outputs related to the actions
        action_outputs = x[:, 1]
        pred_actions = self.action_pred(action_outputs)

        return pred_actions

    def get_params(self):
        return self.parameters()



class DiffusionGPT2(DiffusionGPT):
    '''
    Variant which only learns to predict the last action of the sequence
    '''
    def __init__(self, state_dim: int, device: str, action_dim: int, embed_dim: int, embed_pdrob: float, attn_pdrop: float, resid_pdrop: float, n_layers: int, n_heads: int, goal_seq_len: int, obs_seq_len: int, sigma_vocab_size: int):
        super().__init__(state_dim, device, action_dim, embed_dim, embed_pdrob, attn_pdrop, resid_pdrop, n_layers, n_heads, goal_seq_len, obs_seq_len, sigma_vocab_size)
        
    
    def forward(
        self, 
        states,
        actions, 
        goals,
        sigma,
        uncond: Optional[bool] =False
    ):  
        b, t, dim = states.size()
        
        # get the sigma embedding
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, 'b -> b 1')
        emb_t = self.sigma_emb(sigmas)
        if len(states.shape) == 3:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
        
        # define the totoal length of the input sequence
        seq_length = self.goal_seq_len + 1 + t*2
        # get the beginning of the state action pairs
        second_half_idx = self.goal_seq_len + 1 
        # forward the GPT model
        if uncond:
            goals = torch.zeros_like(goals).to(self.device)
        
        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        goal_embed = self.tok_emb(goals)
        action_embed = self.action_emb(actions)
        
        position_embeddings = self.pos_emb[
            :, :(t + self.goal_seq_len), :
        ]  # each position maps to a (learnable) vector
        # note, that the goal states are at the beginning of the sequence since they are available 
        # for all states s_1, .., s_t otherwise the masking would not make sense
        goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:, :])
        # the action get the same position embedding as the related states 
        action_x = self.drop(action_embed + position_embeddings[:, -1, :])
        
        # next we stack everything together 
        input_seq = torch.cat([emb_t, goal_x, state_x, action_x], dim=1)
        
        # Note we need to also adept the action masks 
        x = self.blocks(input_seq)
        x = self.ln_f(x)
        
        # now we want the last output of the sequence
        action_outputs = x[:, -1, :]
        action_outputs = einops.rearrange(action_outputs, 'b d -> b 1 d')
        pred_actions = self.action_pred(action_outputs)

        return pred_actions


class DiffusionGPT3(DiffusionGPT):
    
    def __init__(self, state_dim: int, device: str, action_dim: int, embed_dim: int, embed_pdrob: float, attn_pdrop: float, resid_pdrop: float, n_layers: int, n_heads: int, goal_seq_len: int, obs_seq_len: int, sigma_vocab_size: int):
        super().__init__(state_dim, device, action_dim, embed_dim, embed_pdrob, attn_pdrop, resid_pdrop, n_layers, n_heads, goal_seq_len, obs_seq_len, sigma_vocab_size)
        
    
    