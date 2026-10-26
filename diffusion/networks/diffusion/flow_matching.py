import torch
import torch.nn as nn
from omegaconf import DictConfig
import hydra

from helper import so3
from networks.diffusion.point_diffusion import Diffusion


class FlowMatching(nn.Module):
    """
    Conditional flow matching on SE(3) = SO(3) x R^3 as an alternative to the DDPM in point_diffusion.Diffusion.

    Path (as in DexGraspFlow / GraspMF):
        prior     R_0 ~ Haar(SO(3)),  p_0 ~ N(0, I)  (translation in units of trans_scale)
        path      R_t = R_0 Exp(t Log(R_0^T R_1)),  p_t = (1 - t) p_0 + t p_1
        target    w = Log(R_0^T R_1) (body frame, constant along the geodesic),  v = p_1 - p_0
    Sampling integrates the learned velocity with explicit Euler steps on the group.

    The network sees the pose as [p_t, rot6d(R_t)] (9 values, continuous) and predicts [v, w] (6 values).
    Grasps enter and leave in the se3_pseudo_log_map_representation layout [translation | rotation vector]
    (requires scaler: no_scale), so the agent, processor and metrics are unchanged.
    """

    def __init__(
        self,
        embedder: DictConfig,
        model: DictConfig,
        input_dim: int = 6,
        n_steps: int = 10,
        trans_scale: float = 0.1,
        rot_loss_weight: float = 1.0,
        time_sampling: str = 'uniform',
        device: str = 'cuda',
        **kwargs,
    ) -> None:
        super().__init__()
        assert input_dim == 6, "FlowMatching expects se3_pseudo_log_map_representation (grasp_dim 6)"
        self.device = device
        self.input_dim = input_dim
        self.embedder = hydra.utils.instantiate(embedder).to(device)
        self.model = hydra.utils.instantiate(model)
        self.n_steps = n_steps
        self.trans_scale = trans_scale  # metres per unit of the normalised translation
        self.rot_loss_weight = rot_loss_weight
        self.time_sampling = time_sampling

    def get_params(self):
        return list(self.embedder.parameters()) + list(self.model.parameters())

    def sample_time(self, n, dtype, device):
        if self.time_sampling == 'uniform':
            return torch.rand(n, dtype=dtype, device=device)
        if self.time_sampling == 'beta':
            # more weight close to the data (t -> 1), mixture as in GraspMF: 0.98 Beta(1.9, 1) + 0.02 U[0, 1]
            beta = torch.distributions.Beta(torch.tensor(1.9), torch.tensor(1.0)).sample((n,)).to(device, dtype)
            uniform = torch.rand(n, dtype=dtype, device=device)
            return torch.where(torch.rand(n, device=device) < 0.98, beta, uniform)
        raise ValueError(f"Unknown time_sampling '{self.time_sampling}'")

    def velocity(self, p, R, t, condition):
        """ Predicted [translation velocity, body angular velocity] at normalised translation p and rotation R """
        embedding = self.embedder.compute_features(condition, t, p * self.trans_scale)
        x = torch.cat([p, so3.matrix_to_rotation_6d(R)], dim=-1)
        # the time embedding was built for integer DDPM steps, scale the flow time accordingly
        return self.model(x, t * 1000, embedding)

    def loss(self, x: torch.Tensor, condition_pc: torch.Tensor, weights=1.0, gram_schmidt=False):
        batch_size = len(x)
        condition, num_objects = self.embedder(condition_pc)
        condition = Diffusion.repeat_condition(condition, num_objects, batch_size)

        p_1 = x[:, :3] / self.trans_scale
        R_1 = so3.exp_map(x[:, 3:])
        p_0 = torch.randn_like(p_1)
        R_0 = so3.random_rotation(batch_size, dtype=x.dtype, device=x.device)

        t = self.sample_time(batch_size, x.dtype, x.device)
        w_target = so3.log_map(R_0.transpose(-1, -2) @ R_1)
        v_target = p_1 - p_0
        R_t = R_0 @ so3.exp_map(t[:, None] * w_target)
        p_t = (1 - t[:, None]) * p_0 + t[:, None] * p_1

        pred = self.velocity(p_t, R_t, t, condition)
        loss = ((pred[:, :3] - v_target) ** 2).sum(-1) + self.rot_loss_weight * ((pred[:, 3:] - w_target) ** 2).sum(-1)
        return (loss * weights).mean()

    @torch.no_grad()
    def sample(self, condition, batch_size, n_steps=None, return_trajectory=False):
        n_steps = n_steps or self.n_steps
        dtype = torch.get_default_dtype()
        p = torch.randn(batch_size, 3, dtype=dtype, device=self.device)
        R = so3.random_rotation(batch_size, dtype=dtype, device=self.device)
        trajectory = []
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((batch_size,), k * dt, dtype=dtype, device=self.device)
            v = self.velocity(p, R, t, condition)
            p = p + dt * v[:, :3]
            R = R @ so3.exp_map(dt * v[:, 3:])
            if return_trajectory:
                trajectory.append(torch.cat([p * self.trans_scale, so3.log_map(R)], dim=-1))
        x = torch.cat([p * self.trans_scale, so3.log_map(R)], dim=-1)
        return (x, torch.stack(trajectory, dim=1)) if return_trajectory else x

    def forward(self, condition_pc, gram_schmidt=False, num_samples=1, *args, **kwargs):
        '''
        Generates num_samples grasps per point cloud, object-major [num_objects * num_samples, 6]
        '''
        condition, num_objects = self.embedder(condition_pc)
        batch_size = num_objects * num_samples
        condition = Diffusion.repeat_condition(condition, num_objects, batch_size)
        return self.sample(condition, batch_size, *args, **kwargs)
