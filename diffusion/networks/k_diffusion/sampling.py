import math

from scipy import integrate
import torch
from torch import nn
from torchdiffeq import odeint
from tqdm.auto import trange, tqdm

from . import utils


def append_zero(action):
    return torch.cat([action, action.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """
    Constructs the noise schedule of Karras et al. (2022).
    """
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)


def to_d(action, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (action- denoised) / utils.append_dims(sigma, action.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """
    Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step.
    """
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_euler(
    model, 
    state, 
    action, 
    sigmas, 
    extra_args=None, 
    callback=None, 
    disable=None, 
    s_churn=0., 
    s_tmin=0., 
    s_tmax=float('inf'), 
    s_noise=1.
):
    """
    Implements a variant of Algorithm 2 (Euler steps) from Karras et al. (2022).
    Stochastic sampler, which combines a first order ODE solver with explicit Langevin-like "churn"
    of adding and removing noise. 
    Every update consists of these substeps:
    1. Addition of noise given the factor eps
    2. Solving the ODE dx/dt at timestep t using the score model 
    3. Take Euler step from t -> t+1 to get x_{i+1}
    
    In contrast to the Heun variant, this variant does not compute a 2nd order correction step
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(action) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            action = action + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(state, action, sigma_hat * s_in, **extra_args)
        d = to_d(action, sigma_hat, denoised)
        if callback is not None:
            callback({'x': action, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        action = action + d * dt
    return action


@torch.no_grad()
def sample_euler_ancestral(
    model, 
    state, 
    action, 
    sigmas, 
    extra_args=None, 
    callback=None, 
    disable=None, 
    eta=1.
):
    """
    Ancestral sampling with Euler method steps.
    
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(state, action, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': action, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(action, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        action = action + d * dt
        action = action + torch.randn_like(action) * sigma_up
    return action


@torch.no_grad()
def sample_heun(
    model, 
    state, 
    action, 
    sigmas, 
    extra_args=None, 
    callback=None, 
    disable=None, 
    s_churn=0., 
    s_tmin=0., 
    s_tmax=float('inf'), 
    s_noise=1.
):
    """
    Implements Algorithm 2 (Heun steps) from Karras et al. (2022).
    
    Stochastic sampler, which combines higher-order ODE wiht explicit Langevin-like "churn"
    of adding and removing noise. 
    Every update consists of two substeps:
    1. Addition of noise given the factor eps
    2. Solving the ODE dx/dt at timestep t
    3. Take Euler step from t -> t+1 to get x_{i+1}
    4. Apply 2nd order corecction
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(action) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)

        if gamma > 0:
            action= action+ eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        # compute the derivative dx/dt at timestep t
        denoised = model(state, action, sigma_hat * s_in, **extra_args)
        d = to_d(action, sigma_hat, denoised)

        if callback is not None:
            callback({'x': action, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        dt = sigmas[i + 1] - sigma_hat

        # if we are at the last timestep: use Euler method
        if sigmas[i + 1] == 0:
            # Euler method
            action= action+ d * dt
        else:
            # Heun's method
            action_2 = action+ d * dt
            denoised_2 = model(state, action_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d( action_2, sigmas[i + 1], denoised_2)
            # 2nd order correction
            d_prime = (d + d_2) / 2
            action= action+ d_prime * dt
    return action


@torch.no_grad()
def sample_dpm_2(model, state, action, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """
    A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022).
    
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(action) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            action = action + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(state, action, sigma_hat * s_in, **extra_args)
        d = to_d(action, sigma_hat, denoised)
        if callback is not None:
            callback({'action': action, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            dt = sigmas[i + 1] - sigma_hat
            action = action + d * dt
        else:
            # DPM-Solver-2 2nd order variant
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            action_2 = action + d * dt_1
            denoised_2 = model(state, action_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d( action_2, sigma_mid, denoised_2)
            action = action + d_2 * dt_2
    return action


@torch.no_grad()
def sample_dpm_2_ancestral(model, state, action, sigmas, extra_args=None, callback=None, disable=None, eta=1.):
    """
    Ancestral sampling with DPM-Solver inspired second-order steps.

    Ancestral sampling is based on the DDPM paper (https://arxiv.org/abs/2006.11239) generation process.
    Song et al. (2021) show that ancestral sampling can be used to improve the performance of DDPM for its SDE formulation.
    
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(state, action, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': action, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(action, sigmas[i], denoised)
        if sigma_down == 0:
            # Euler method
            dt = sigma_down - sigmas[i]
            action= action+ d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]
            action_2 = action+ d * dt_1
            denoised_2 = model(state, action_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d( action_2, sigma_mid, denoised_2)
            action= action+ d_2 * dt_2
            action= action+ torch.randn_like(action) * sigma_up
    return action


def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod
    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]


@torch.no_grad()
def sample_lms(
    model, 
    state, 
    action, 
    sigmas, 
    extra_args=None, 
    callback=None, 
    disable=None, 
    order=4
):  
    '''
    A linear multistep sampler.

    1. compute x_{t-1} using the current noise level 
    2. compute dx/dt at x_{t-1} using the current noise level 
    3. weight the dx/dt with the coefficients of the linear multistep method
    '''
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(state, action, sigmas[i] * s_in, **extra_args)
        d = to_d(action, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({'x': action, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        cur_order = min(i + 1, order)
        coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
        action = action + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return action


@torch.no_grad()
def log_likelihood(model, state, action, sigma_min, sigma_max, extra_args=None, atol=1e-4, rtol=1e-4):
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])
    v = torch.randint_like(action, 2) * 2 - 1
    fevals = 0
    def ode_fn(sigma, action):
        nonlocal fevals
        with torch.enable_grad():
            action= action[0].detach().requires_grad_()
            denoised = model(state, action, sigma * s_in, **extra_args)
            d = to_d(action, sigma, denoised)
            fevals += 1
            grad = torch.autograd.grad((d * v).sum(), action)[0]
            d_ll = (v * grad).flatten(1).sum(1)
        return d.detach(), d_ll
    action_min = action, action.new_zeros([action.shape[0]])
    t = action.new_tensor([sigma_min, sigma_max])
    sol = odeint(ode_fn, action_min, t, atol=atol, rtol=rtol, method='dopri5')
    latent, delta_ll = sol[0][-1], sol[1][-1]
    ll_prior = torch.distributions.Normal(0, sigma_max).log_prob(latent).flatten(1).sum(1)
    return ll_prior + delta_ll, {'fevals': fevals}


class PIDStepSizeController:
    """A PID controller for ODE adaptive step size control."""
    def __init__(self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, action):
        return 1 + math.atan(action- 1)

    def propose_step(self, error):
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept


class DPMSolver(nn.Module):
    """DPM-Solver. See https://arxiv.org/abs/2206.00927."""

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, state, action, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * action.new_ones([action.shape[0]])
        eps = (action - self.model(state, action, sigma, *args, **self.extra_args, **kwargs)) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, state, action, t, t_next, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', state, action, t)
        action_1 = action- self.sigma(t_next) * h.expm1() * eps
        return action_1, eps_cache

    def dpm_solver_2_step(self, state, action, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', state, action, t)
        s1 = t + r1 * h
        u1 = action - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', state, u1, s1)
        action_2 = action - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return action_2, eps_cache

    def dpm_solver_3_step(self, state, action, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', state, action, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = action - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', state, u1, s1)
        u2 = action - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', state, u2, s2)
        action_3 = action - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return action_3, eps_cache

    def dpm_solver_fast(self, state, action, t_start, t_end, nfe, eta=0., s_noise=1.):
        if not t_end > t_start and eta:
            raise ValueError('eta must be 0 for reverse sampling')

        m = math.floor(nfe / 3) + 1
        ts = torch.linspace(t_start, t_end, m + 1, device=action.device)

        if nfe % 3 == 0:
            orders = [3] * (m - 2) + [2, 1]
        else:
            orders = [3] * (m - 1) + [nfe % 3]

        for i in range(len(orders)):
            eps_cache = {}
            t, t_next = ts[i], ts[i + 1]
            if eta:
                sd, su = get_ancestral_step(self.sigma(t), self.sigma(t_next), eta)
                t_next_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t_next) ** 2 - self.sigma(t_next_) ** 2) ** 0.5
            else:
                t_next_, su = t_next, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', state, action, t)
            denoised = action- self.sigma(t) * eps
            if self.info_callback is not None:
                self.info_callback({'x': action, 'i': i, 't': ts[i], 't_up': t, 'denoised': denoised})

            if orders[i] == 1:
                action, eps_cache = self.dpm_solver_1_step(state, action, t, t_next_, eps_cache=eps_cache)
            elif orders[i] == 2:
                action, eps_cache = self.dpm_solver_2_step(state, action, t, t_next_, eps_cache=eps_cache)
            else:
                action, eps_cache = self.dpm_solver_3_step(state, action, t, t_next_, eps_cache=eps_cache)

            action= action+ su * s_noise * torch.randn_like(action)

        return action

    def dpm_solver_adaptive(self, state, action, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1.):
        if order not in {2, 3}:
            raise ValueError('order should be 2 or 3')
        forward = t_end > t_start
        if not forward and eta:
            raise ValueError('eta must be 0 for reverse sampling')
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        action_prev = action
        accept = True
        pid = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety)
        info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}

        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            eps_cache = {}
            t = torch.minimum(t_end, s + pid.h) if forward else torch.maximum(t_end, s + pid.h)
            if eta:
                sd, su = get_ancestral_step(self.sigma(s), self.sigma(t), eta)
                t_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t) ** 2 - self.sigma(t_) ** 2) ** 0.5
            else:
                t_, su = t, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', state, action, s)
            denoised = action - self.sigma(s) * eps

            if order == 2:
                action_low, eps_cache = self.dpm_solver_1_step(state, action, s, t_, eps_cache=eps_cache)
                action_high, eps_cache = self.dpm_solver_2_step(state, action, s, t_, eps_cache=eps_cache)
            else:
                action_low, eps_cache = self.dpm_solver_2_step(state, action, s, t_, r1=1 / 3, eps_cache=eps_cache)
                action_high, eps_cache = self.dpm_solver_3_step(state, action, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum( action_low.abs(), action_prev.abs()))
            error = torch.linalg.norm(( action_low - action_high) / delta) / action.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                action_prev = action_low
                action = action_high + su * s_noise * torch.randn_like( action_high)
                s = t
                info['n_accept'] += 1
            else:
                info['n_reject'] += 1
            info['nfe'] += order
            info['steps'] += 1

            if self.info_callback is not None:
                self.info_callback({'x': action, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error, 'h': pid.h, **info})

        return action, info


@torch.no_grad()
def sample_dpm_fast(model, state, action, sigma_min, sigma_max, n, extra_args=None, callback=None, disable=None, eta=0., s_noise=1.):
    """DPM-Solver-Fast (fixed step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_maactionmust not be 0')
    with tqdm(total=n, disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        return dpm_solver.dpm_solver_fast(state, action, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), n, eta, s_noise)


@torch.no_grad()
def sample_dpm_adaptive(model, state, action, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., return_info=False):
    """
    DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927.
    
    """
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_maactionmust not be 0')
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        action, info = dpm_solver.dpm_solver_adaptive(state, action, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise)
    if return_info:
        return action, info
    return action


@torch.no_grad()
def sample_dpmpp_2s_ancestral(model, state, action,sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1.):
    """
    Ancestral sampling with DPM-Solver++(2S) second-order steps.
    
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(state, action, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'action': action, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(action, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            action = action + d * dt
        else:
            # DPM-Solver-2++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * action - (-h * r).expm1() * denoised
            denoised_2 = model(state, x_2, sigma_fn(s) * s_in, **extra_args)
            action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised_2
        # Noise addition
        action = action + torch.randn_like(action) * s_noise * sigma_up
    return action


@torch.no_grad()
def sample_dpmpp_2m(model, state, action, sigmas, extra_args=None, callback=None, disable=None):
    """
    DPM-Solver++(2M). with second-order steps.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(state, action, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'action': action, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised_d
        old_denoised = denoised
    return action