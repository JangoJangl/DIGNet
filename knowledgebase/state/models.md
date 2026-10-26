---
title: Models
type: state
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [diffusion, ddpm, pointnet, vector-neurons, model]
related: ["[[state/architecture]]", "[[state/problems]]", "[[research/point-cloud-encoders]]", "[[research/grasp-generation-sota]]"]
sources: [diffusion/networks/diffusion/point_diffusion.py, diffusion/networks/diffusion/embedder_free_diffusion.py, diffusion/networks/diffusion/diffusion_models.py, diffusion/networks/vision_encoder/]
---

# Models

## Diffusion core (`point_diffusion.Diffusion`)

- DDPM adapted from Diffusion-QL (Wang et al. 2022). `n_timesteps` = 5 by default (10/100 in sweeps), `predict_epsilon: true`, L2 loss.
- Beta schedule: config says `cosine`, but until 2026-10-25 the code **always overwrote it with the VP schedule** (see [[state/problems#P-01 Beta schedule is always VP]]). Existing checkpoints were therefore trained with VP; set `beta_schedule: vp` to reproduce them.
- `clip_denoised`: the predicted x0 is clamped to [-1, 1] (`set_bounds`), which only makes sense for scaled representations. With `no_scale` translations are in metres (fine, objects are small) and so(3) log maps lie in [-pi, pi], so **rotations beyond 1 rad were clipped** during sampling (see [[state/problems#P-08 Clamping to plus minus one clips rotations]]). Since 2026-10-25 `set_bounds` widens the rotation bounds to pi for unscaled log map representations.
- Optional `diffusion_x` (extra denoising steps at t = 0) from Pearce et al. 2023.
- Sampling = ancestral DDPM (`p_sample_loop`); a DDIM sampler is present but broken (calls non-existing helpers).

## Denoiser (`DiffusionMLPNetwork`)

Residual MLP (default 6 layers x 512, Mish) over `concat(x_t, sinusoidal_t_embedding(16), condition)`. The condition is one global embedding per object.

## Embedders

| Config | Class | Output | Notes |
|---|---|---|---|
| `vnn_pointnet2` (default in `diffusion_model.yaml`) | `VNNPointnet2` -> `VNN_ResnetPointnet` | `embedding_dim` (multiple of 3) | Vector Neuron DGCNN-style encoder (Deng et al. 2021), SO(3) **equivariant** features flattened to a vector. The MLP on top is not equivariant |
| `pointnet2` | `Pointnet2` | `embedding_dim` | PointNet++ MSG (radius r, 2r, 4r) with global max pool |
| `hierarchical_pointnet2` | (config points to `Pointnet2`, class `pointnet2_hierarchical.Pointnet2`) | list of (xyz, features) per level | Intended: one abstraction level per diffusion step, local feature of the nearest point to the current grasp translation. Config is wrong, and the lookup only works for one object per batch |
| embedder-free | `diffusion_pointnet.Diffusion_Pointnet2` | grasp noise | PointNet++ is the denoiser itself: the object cloud is expressed in the frame of the noisy grasp and the network predicts the noise. Until 2026-10-25 the cloud was multiplied by the pose instead of its inverse ([[state/problems#P-05 Embedder-free model used the wrong transform]]) |
| `se3transformer` | external `se3_transformer-pytorch` | | Config only, parameters undefined |

## Token embedders, cross-attention denoiser, flow matching (2026-10-25)

| Config | Class | Output |
|---|---|---|
| `embedder: pointnet2_local` | `PointNet2Local` | 128 tokens (centroid xyz + feature), MSG radii r = 2 cm with radius-normalised offsets |
| `embedder: ptv3` | `PTv3Encoder` | tokens after 3 serialized poolings (1024 -> 128), 4 stages, channels 32 to 256 |
| `model: attention_diffusion_model` | DDPM + `CrossAttentionDenoiser` | grasp query attends to the `attention_k` tokens closest to the current (noisy) grasp position |
| `model: flow_matching_model` | `FlowMatching` + `CrossAttentionDenoiser` | velocity on SE(3), `flow_steps` Euler steps; see [[research/flow-matching-se3]] |

Token embedders implement `compute_features(condition, t, position)` by expressing the token positions relative to the current grasp position, so the same denoiser works for DDPM and flow matching. With `return_tokens: false` they return a global vector and work with the old MLP denoiser.

## Pre-embedding variant

`pre_embedding_grasp_diffusion_agent.py` + `object_representation/` train the diffusion model on frozen Point-MAE / PointNeXt embeddings computed offline. Not part of the main config path.

## Parameter count / compute

`float64` is the global default dtype (`torch.set_default_dtype(torch.float64)`), which makes every layer run in double precision, several times slower on consumer GPUs than float32/bf16 ([[state/problems#P-09 Global float64]]).
