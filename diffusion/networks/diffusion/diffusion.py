
from matplotlib.backend_bases import MouseEvent
import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
from omegaconf import DictConfig
import hydra

from .utils import (cosine_beta_schedule,
                    linear_beta_schedule,
                    vp_beta_schedule,
                    extract,
                    Losses
)


# code adapted from https://github.com/twitter/diffusion-rl/blob/master/agents/diffusion.py
class Diffusion(nn.Module):

    def __init__(
        self,
        condition_dim: int,
        input_dim: int,
        model: DictConfig,
        beta_schedule: str,
        n_timesteps: int,
        loss_type: str,
        clip_denoised: bool,
        predict_epsilon=True,
        device: str = 'cuda',
        diffusion_x: bool = False,
        diffusion_x_M: int=32,
    ) -> None:
        super().__init__()
        self.device = device
        self.condition_dim = condition_dim
        self.input_dim = input_dim
        self.input_bounds = None
        # chose your beta style
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(n_timesteps).to(self.device)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(n_timesteps).to(self.device)
        elif beta_schedule == 'vp':
            self.betas = vp_beta_schedule(n_timesteps).to(self.device)

        self.model = hydra.utils.instantiate(model)
        self.n_timesteps = n_timesteps
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        # define alpha stuff
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]])
        # required for forward diffusion q( x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(self.device)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1).to(self.device)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).to(self.device)
        
        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20)).to(self.device)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).to(self.device)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod).to(self.device)

        # either l1 or l2
        self.loss_fn = Losses[loss_type]()

        self.diffusion_x = diffusion_x
        self.diffusion_x_M = diffusion_x_M
    
    def get_params(self):
        '''
        Helper method to get all model parameters
        '''
        return self.model.get_params()
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        '''
        if self.predict_epsilon, model output is (scaled) noise, which is applied to compute the
        value for x_{t-1} otherwise the model can output x_{t-1} directly
        otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """
        Computes the posterior mean and variance of the diffusion step at timestep t

        q( x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, grad: bool=True):
        '''
        Predicts the denoised x_{t-1} sample given the current diffusion model
        
        :param x:  noisy input
        :param t:  batch of timesteps 
        :param s:  the current condition batch
        :param grad:  bool, if the gradient should be computed

        :return: 
            - the model mean prediction 
            - the model log variance prediction 
            - the posterior variance 
        '''
        if grad:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model_frozen(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(self.min_input, self.max_input)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, grad: bool = True):
        """
        Generated a denoised sample x_{t-1} given the trained model and noisy sample x_{t}

        :param x:  noisy input
        :param t:  batch of timesteps 
        :param s:  the current condition batch
        :param grad:  bool, if the gradient should be computed

        :return:    torch.Tensor x_{t-1}
        """
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, grad=grad)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, condition, shape, verbose=False, return_diffusion=False):
        """
        Main loop for generating samples using the learned model and the inverse diffusion step
        
        :param condition: the current condition
        :param shape: the shape of the input samples [B, D]
        :param return diffusion: bool, if the complete diffusion chain should be returned or not 
        
        :return: either the predicted x_0 sample or a list with [x_{t-1}, .., x_{0}]
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)
        if return_diffusion: 
            diffusion = [x]

        # start the inverse diffusion process to generate the input conditioned on the noise
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, timesteps, condition)
            # if we want to return the complete chain add thee together
            if return_diffusion: 
                diffusion.append(x)

        if self.diffusion_x:
            # The sampling process runs as normal for T denoising timesteps. The denoising timestep is then fixed,
            # t = 0, and extra denoising iterations continue to run for M timesteps. The intuition behind this
            # is that samples continue to be moved toward higher-likelihood regions for longer.
            # https://openreview.net/pdf?id=Pv1GPQzRrC8

            timesteps = torch.full((batch_size,), 0, device=self.device, dtype=torch.long)
            for m in range(0, self.diffusion_x_M):
                x = self.p_sample(x, timesteps, condition)
                if return_diffusion:
                    diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def sample(self, condition, *args, **kwargs):
        """
        Main Method to generate inputs conditioned on the batch of condition
        
        :param condition: the current condition for the diffusion model
        
        :return: x_{0} the predicted inputs from the diffusion model
        """
        batch_size = condition.shape[0]
        shape = (batch_size, self.input_dim)
        input = self.p_sample_loop(condition, shape, *args, **kwargs)
        return input.clamp_(-self.max_input, self.max_input)
    
    def guided_p_sample(self, x, t, s, fun):
        '''
        Sample x_{t-1} from the model at the given timestep with additional conditioning
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param s: the current condition
        :param fun: the conditoning model, subclass from torch.nn.Model
        
        :return: x_{t-1}
        '''
        b, *_ = x.shape
        with torch.no_grad():
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # Involve Function Guidance
        a = model_mean.clone().requires_grad_(True)
        q_value = fun(s, a)
        # q_value = q_value / q_value.abs().mean().detach()  # normalize q
        grads = torch.autograd.grad(outputs=q_value, inputs=a, create_graph=True, only_inputs=True)[0].detach()
        return (model_mean + model_log_variance * grads) + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def guided_sample(self, condition: torch.Tensor, cond_fun, start: int=0.2, verbose=False, return_diffusion=False):
        """
        Generated diffusion samples conditioned on cond_fun and starts deffusion at 20% of initial timesteps
        for faster generating process
        
        :param condition: the current condition batch
        :param cond_fun: the condition function to guide samples
        :param start:  defines the starting timestep 0 refers to starting sampling from t and 0.2 means
                       starting sampling from at 20% of the inital time step for faster sampling
        :return_diffusion: bool to decide, if only x_0 or all samples should be returned from the looop
        
        :return: either the predicted x_0 sample or a list with [x_{t-1}, .., x_{0}]
        """
        batch_size = condition.shape[0]
        shape = (batch_size, self.input_dim)
        x = torch.randn(shape, device=self.device)
        i_start = self.n_timesteps * start

        if return_diffusion: diffusion = [x]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            if i <= i_start:
                x = self.guided_p_sample(x, timesteps, condition, cond_fun)
            else:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, condition)

            if return_diffusion: diffusion.append(x)

        x = x.clamp_(-self.max_input, self.max_input)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1).clamp_(-self.max_input, self.max_input)
        else:
            return x

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise=None):
        """
        Main Method to sample the forward diffusion start with random noise and get 
        the required values for the desired noisy sample at q(x_{t} | x_{0})
        at timestep t. The method is used for training only. 
        
        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start: torch.Tensor, condition: torch.Tensor, t: torch.Tensor, weights=1.0):
        """
        Computes the training loss of the diffusion model given a batch of data. At every
        training sample of the batch we generate noisy samples at different timesteps t
        and let the diffusion model predict the initial sample from the generated noisy one.
        The loss is computed by comparing the denoised sample from the diffusion model against
        the initial sample.
        
        :param x_start: the inital input samples
        :param condition:   the current condition batch
        :param t:       the batch of chosen timesteps
        :param weights: parameter to weight the individual losses
        
        :return loss: the total loss of the model given the input batch
        """
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, condition)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x: torch.Tensor, condition: torch.Tensor, weights=1.0):
        '''
        Computes the batch loss for the diffusion model
        '''
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, condition, t, weights)

    def forward(self, condition, *args, **kwargs):
        '''
        General forward method, which generates samples given the chosen condition
        '''
        return self.sample(condition, *args, **kwargs)

    def sample_t_middle(self, condition: torch.Tensor):
        """
        Fast generation of new samples, which only use 20% of the denoising steps of the true 
        denoising process
        """
        batch_size = condition.shape[0]
        shape = (batch_size, self.input_dim)
        

        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)

        t = np.random.randint(0, int(self.n_timesteps*0.2))
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, timesteps, condition, grad=(i == t))
        input = x
        return input.clamp_(-self.max_input, self.max_input)

    def sample_t_last(self, condition: torch.Tensor):
        """
        Generate denoised samples with all denoising steps 
        """
        batch_size = condition.shape[0]
        shape = (batch_size, self.input_dim)
        

        x = torch.randn(shape, device=self.device)
        cur_T = np.random.randint(int(self.n_timesteps * 0.8), self.n_timesteps)
        for i in reversed(range(0, cur_T)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            if i != 0:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, condition)
            else:
                x = self.p_sample(x, timesteps, condition)

        input = x
        return input.clamp_(-self.max_input, self.max_input)

    def sample_last_few(self, condition: torch.Tensor):
        """
        Return samples, that have not complelty denoised the data. The inverse diffusion process
        is stopped 5 timetsteps before the true denoising. 
        """
        batch_size = condition.shape[0]
        shape = (batch_size, self.input_dim)

        x = torch.randn(shape, device=self.device)
        nest_limit = 5
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            if i >= nest_limit:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, condition)
            else:
                x = self.p_sample(x, timesteps, condition)

        input = x
        return input.clamp_(-self.max_input, self.max_input)
    
    def ddim_sample(
        self,
        model,
        condition,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"]}
    
    def ddim_sample_loop(
        self,
        model,
        condition,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            condition,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
        ):
            final = sample
        return final["sample"]
