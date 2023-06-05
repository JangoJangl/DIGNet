import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from . import utils
# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
# code adapted from 
# https://github.com/GuyTevet/motion-diffusion-model/blob/55cd84e14ba8e0e765700913023259b244f5e4ed/model/cfg_sampler.py
class ClassifierFreeSampleModel(nn.Module):
    
    def __init__(self, model, cond_lambda: float=2):
        super().__init__()
        self.model = model  # model is the actual model to run
        # pointers to inner model
        self.cond_lambda = cond_lambda
        if self.cond_lambda == 1:
            self.cond = True
        else:
            self.cond = False

    def forward(self, state, action, goal, sigma, **extra_args):
        if self.cond:
            return self.model(state, action, goal, sigma)
        elif self.cond_lambda == 0:
            uncond_dict = {'uncond': True}
            out_uncond = self.model(state, action, goal, sigma, **uncond_dict)
            return out_uncond
        else:
            action = deepcopy(action)
            
            out = self.model(state, action, goal, sigma, **extra_args)
            uncond_dict = {'uncond': True}
            out_uncond = self.model(state, action, goal, sigma, **uncond_dict)
            
            return out_uncond + self.cond_lambda * (out - out_uncond)
    
    def get_params(self):
        return self.model.get_params()
    


class ClassifierGuidedSampleModel(nn.Module):
    
    def __init__(self, model, cond_func, cond_lambda: float=2):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.guide = cond_func
        # pointers to inner model
        self.cond_lambda = cond_lambda
        
    def forward(self, state, action, goal, sigma, cond_lambda=None, **extra_args):
        if cond_lambda is None:
            cond_lambda = self.cond_lambda
        pred_action = self.model(state, action, goal, sigma, **extra_args)
        with torch.enable_grad():
            a = pred_action.clone().requires_grad_(True)
            q_value = self.guide(state, pred_action, goal)
            grads = torch.autograd.grad(outputs=q_value, inputs=a, create_graph=True, only_inputs=True)[0].detach()
        
        return pred_action + cond_lambda * grads * utils.append_dims(sigma**2, action.ndim)
        
    def get_params(self):
        return self.model.get_params()