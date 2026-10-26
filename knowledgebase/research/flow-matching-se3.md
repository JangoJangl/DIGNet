---
title: Flow matching on SE(3) for grasp generation
type: research
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [flow-matching, se3, generative-model, grasping]
related: ["[[state/ideas#I-04 Flow matching on SE(3)]]", "[[state/models]]", "[[research/grasp-generation-sota]]"]
confidence: medium
sources: ["https://arxiv.org/abs/2610.26962", "https://arxiv.org/abs/2608.26076", "https://arxiv.org/abs/2608.03295", "https://arxiv.org/abs/2210.02747"]
---

# Flow matching on SE(3) for grasp generation

## State of the art (checked 2026-10-25)

| Method | Year | Formulation | Steps | Notes |
|---|---|---|---|---|
| EquiGraspFlow | CoRL 2024 | SE(3)-equivariant conditional flow matching, VN backbone | ~20 | ACRONYM, baseline for the ones below |
| DexGraspFlow (arXiv 2610.26962) | 2026 | prior x0 ~ N(0, tau^2 I), R0 Haar (or around an init); path `R_t = R0 Exp(t Log(R0^T R1))`, `x_t = (1-t) x0 + t x1`; targets `Log(R0^T R1)`, `x1 - x0`; t ~ U(0, 1); masked MSE; explicit Euler on the group | 10 | dexterous hands, 95.3 % sim success, 91 % over 100 real trials |
| GraspMF, Lie group constrained MeanFlow (arXiv 2608.26076) | 2026 | product group SO(3) x R3, endpoint prediction X(H, s, t) -> H1, average velocity, flow-matching anchor + semigroup loss, t ~ 0.98 Beta(1.9, 1) + 0.02 U | 1 to 5 | ACRONYM 10 categories; T = 5: 87.4 % in-distribution / 71.7 % OOD sim success, 15.5 ms for 100 grasps (T = 1: 6.3 ms) vs 188 ms for EquiGraspFlow |
| GraspMeanFlow (arXiv 2608.03295) | 2026 | SE(3)-equivariant MeanFlow (average velocity over [s, t]), VN backbone from EquiGraspFlow | 1 | one step reaches the EMD of a 5-step SE(3) flow; up to +24.3 points grasp success in the few-step regime |

Base theory: flow matching (Lipman et al., ICLR 2023), rectified flow (Liu et al., ICLR 2023), Riemannian flow matching (Chen and Lipman, 2024).

## What is implemented (`networks/diffusion/flow_matching.py`)

DexGraspFlow-style conditional flow matching, the simplest of the above:

- Prior: translation `p0 ~ N(0, I)` in units of `trans_scale` = 0.1 m, rotation `R0` Haar uniform.
- Path: body-frame geodesic `R_t = R0 Exp(t w)`, `w = Log(R0^T R1)`, linear translation.
- Network input `[p_t, rot6d(R_t)]` (continuous, 9 values), output `[v, w]` (6 values). Time embedding as in the DDPM (sinusoidal on 1000 t).
- Loss `|v_hat - v|^2 + rot_loss_weight |w_hat - w|^2`, `time_sampling: uniform | beta` (beta = GraspMF mixture).
- Sampling: `flow_steps` explicit Euler steps on SE(3).
- Interface identical to the DDPM (`loss`, `forward(points, num_samples)`), input/output in `se3_pseudo_log_map_representation`, so agent, metrics and evaluation are unchanged. SO(3) helpers in `helper/so3.py` (robust log near pi, no Theseus).

## Possible optimisations, not implemented

1. **MeanFlow / few-step**: learn the average velocity u(s, t) with the semigroup loss (GraspMF) to sample with 1 to 2 network calls. Biggest latency win.
2. **Endpoint parameterisation** (predict H1 and derive the velocity, GraspMF) instead of predicting the velocity directly; usually more stable near t = 1, needs the 1/(1-t) clamp.
3. **Better integrators**: Heun / midpoint on the group, or adaptive steps; cheap to add in `sample`.
4. **Minibatch optimal transport coupling** between prior and data samples (straighter paths, fewer steps), per object as in GraspMeanFlow GMF-JVP.
5. **Narrower rotation prior** around a heuristic initial orientation (DexGraspFlow `tau_R`), e.g. approach towards the object centre, to reduce the search space for a 5-DoF arm.
6. **Equivariant backbone** (VN as in EquiGraspFlow / GraspMeanFlow) to remove rotation augmentation.
7. **Auxiliary heads**: SDF or contact prediction (GraspMF uses a clipped L1 SDF head) as extra supervision of the encoder.
8. **Guidance during sampling**: reachability or collision cost gradients added to the velocity (like classifier guidance) for the SO-101.
9. **Loss weighting** between translation and rotation tuned on validation recall; log both terms separately.
