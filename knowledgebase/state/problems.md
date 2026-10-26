---
title: Problem log
type: problem-log
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [problems, bugs, technical-debt]
related: ["[[state/ideas]]", "[[state/changelog]]", "[[state/training-and-evaluation]]"]
---

# Problem log

Severity: **high** = wrong results or blocks the goal, **medium** = large inefficiency or misleading output, **low** = cleanup. Status: `open`, `fixed` (with date), `wontfix`.

| ID | Problem | Severity | Status |
|---|---|---|---|
| P-01 | Beta schedule is always VP | high | fixed 2026-10-25 |
| P-02 | Learns only from full, canonical object clouds | high | partly fixed 2026-10-25 (augmentation) |
| P-03 | Batch size is forced to one grasp | high | fixed 2026-10-25 |
| P-04 | Evaluation sampled one grasp per object | high | fixed 2026-10-25 |
| P-05 | Embedder-free model used the wrong transform | high | fixed 2026-10-25 |
| P-06 | No negatives, no grasp scoring | high | open |
| P-07 | Metric ignores parallel jaw symmetry | medium | fixed 2026-10-25 (option) |
| P-08 | Clamping to plus minus one clips rotations | high | fixed 2026-10-25 |
| P-09 | Global float64 | medium | open |
| P-10 | Slow data loading and metrics | medium | partly fixed 2026-10-25 |
| P-11 | Stateful shared processor | medium | open |
| P-12 | Broken or stale configs | medium | open |
| P-13 | Hard-coded paths and CUDA | medium | partly fixed 2026-10-25 |
| P-14 | Control points and grasp frame do not match | low | open |
| P-15 | Representation discontinuities | medium | open |
| P-16 | Mixed scaling in the embedder-free model | medium | open |
| P-17 | EMA never updated | medium | fixed 2026-10-25 |
| P-18 | Legacy code and dead paths | low | open |
| P-19 | No physics-based evaluation | high | open |
| P-20 | Panda-only labels | high (for SO-101) | open |

## P-01 Beta schedule is always VP

`point_diffusion.py`, `diffusion.py`, `embedder_free_diffusion.py` set `self.betas` from the config and then unconditionally overwrote it with `vp_beta_schedule`. The `beta_schedule: 'cosine'` setting never had an effect. Fixed: the overwrite is removed and unknown names raise. Old checkpoints were trained with VP; use `beta_schedule: vp` to reproduce them.

## P-02 Learns only from full, canonical object clouds

The point cloud is sampled from the full mesh in the ACRONYM canonical orientation, centred at the mesh centre. Real observations are single-view, noisy, arbitrarily oriented and centred at the visible part. Without augmentation the model can not be expected to transfer. Partly fixed: `random_rotation`, `partial_view_prob` (hidden point removal from a random camera, re-centring), `jitter_sigma`. Still missing: rendered depth with a sensor model, table plane, clutter. See [[state/ideas#I-03 Train on rendered partial views]].

## P-03 Batch size is forced to one grasp

`Diffusion.loss` concatenated `num_objects` embeddings with `num_objects * grasps_per_object` noisy grasps, which only works if every object contributes exactly one grasp. The config therefore used `batch_size: 1, num_objects: 1`: one PointNet/VNN forward pass and one optimiser step per single grasp. Fixed: the embedding is repeated per grasp (`Diffusion.repeat_condition`), the embedder-free model repeats the cloud. New defaults 8 objects x 32 grasps. The hierarchical embedder still only supports one object.

## P-04 Evaluation sampled one grasp per object

`model(points)` generated one grasp per point cloud, so `coverage` and `diversity_loss` compared hundreds of ground truth grasps with a single prediction, and `gripper_check_loss` checked one grasp. The first test object was also skipped. Fixed: `num_eval_samples` (default 100) grasps per object, all objects up to `max_eval_objects`.

## P-05 Embedder-free model used the wrong transform

The embedder-free model is meant to show the object to the network from the gripper's point of view. It computed `H_t @ p` (gripper-to-object pose applied to object points) instead of `H_t^-1 @ p`. The network can still learn a function of the pose, but the local, gripper-centric inductive bias was lost. In addition `diffusion_x` passed the raw cloud instead of the transformed one and would crash. Fixed with `points_in_gripper_frame` (rigid inverse), covered by a unit test.

## P-06 No negatives, no grasp scoring

Only successful grasps are used. Generated grasps are never scored or ranked, so collisions and near misses go straight to execution. State-of-the-art systems (GraspGen, Contact-GraspNet, 6-DOF GraspNet) all add an evaluator. See [[state/ideas#I-06 Grasp discriminator trained on generator samples]].

## P-07 Metric ignores parallel jaw symmetry

The 20 control points are 10 left/right pairs. A grasp rotated 180 degrees about its approach axis swaps the fingers and appears ~10 cm away although it is the same physical grasp. Inflates both Chamfer directions and deflates coverage. Fixed as an option (`symmetric_metric`, default true). Training loss is unaffected (DDPM learns the multimodal distribution anyway).

## P-08 Clamping to plus minus one clips rotations

`set_bounds` fixed all bounds to [-1, 1] and the sampler clamps x0 and the final sample to them. With `scaler: no_scale` and log map rotations (range [-pi, pi]) every rotation larger than 1 rad was cut. Fixed: bounds widened to pi for unscaled log map representations. `max_scale`/`var_scale` runs are unaffected only if the scaling really maps into [-1, 1], which is not guaranteed for `var_scale`.

## P-09 Global float64

`training.py` sets `torch.set_default_dtype(torch.float64)`. Consumer GPUs run float64 at 1/32 to 1/64 of float32 throughput. Switch to float32 (and bf16 autocast for the encoder) after checking Theseus/PyTorch3D conversions in float32. See [[state/ideas#I-11 Float32 and mixed precision]].

## P-10 Slow data loading and metrics

`__load_data` loads every mesh at start-up (twice), `__getitem__` reloads mesh and npz and runs Open3D Poisson disk sampling on every access, `num_workers: 0`. The Chamfer metric used Python-level ball tree distances (fixed). Remaining fix: pre-sample point clouds (or cache meshes) offline, allow worker processes (see P-11).

## P-11 Stateful shared processor

One `DiffusionGraspProcessor` instance is shared by train and test datasets and stores `current_obj`, `inverse_representation_function`, `inverse_scaler` as attributes during `preprocess`. The agent reads `test_loader.dataset.processor.current_obj` after a batch. This breaks with `num_workers > 0` (the main process never sees worker state; the comment "4 gives cuda error" is related to CUDA tensors created in the dataset). Fix: return all per-sample data from `__getitem__`, keep tensors on CPU in the dataset, move to GPU in the agent.

## P-12 Broken or stale configs

- `agents/diffusion_grasp_agent_vnn.yaml` references embedder `vnn_pointnet_2` (does not exist).
- `embedder/hierarchical_pointnet2.yaml` targets the plain `Pointnet2` class and passes `diffusion_steps`, which it does not accept.
- `embedder/vnn_pointnet2.yaml` has `pointnet_radius: {pointnet_radius}` (missing `$`).
- `embedder/se3transformer.yaml` references undefined keys.
- `diffusion_grasp_graspnet_pre_embedding_main_config.yaml` references a missing agent config.
- `train_config.yaml` contains unused keys (`diffusion_model`, `latent_dim`, `epoch`, `diffusion_kde`).
- `evaluation.py` hard-codes the run path in the decorator.

## P-13 Hard-coded paths and CUDA

Dataset, cluster and log paths are hard-coded in configs, `evaluation.py`, the `Scaler` and processor defaults. Representation inverses, `start_transform` and the metric hard-coded `'cuda'` (fixed: they follow the input tensor's device). Remaining `.cuda()` calls in unused transforms, `graspnet_loss`, hierarchical encoder.

## P-14 Control points and grasp frame do not match

Grasps are shifted to the finger centre (+0.112 m along z), but `panda.npy` control points have z = 0.075 to 0.105 m, i.e. wrist frame coordinates. The metric is internally consistent but places the virtual fingers ~0.11 m beyond the real ones, which over-weights rotation errors. Decide on one frame and shift the control points by -0.11217 m (would change metric values, so do it together with a new baseline).

## P-15 Representation discontinuities

The default `se3_pseudo_log_map_representation` is discontinuous at rotation angle pi (axis flips), and quaternions have the q/-q double cover. Diffusion in such spaces creates artificial multimodality near the cut. The 6D representation is continuous but over-parameterised; Lie-group diffusion on SE(3) (Urain et al. 2023) or flow matching on SO(3) avoids the issue. See [[state/ideas#I-04 Flow matching on SE(3)]].

## P-16 Mixed scaling in the embedder-free model

With `var_scale`/`max_scale` the point cloud is scaled but `inv_fct(x)` returns unscaled metric poses, so `points_in_gripper_frame` mixes units. Only `no_scale` (the current default) is consistent.

## P-17 EMA never updated

`ExponentialMovingAverage` was created but `update` was commented out, so `use_ema` had no effect. Fixed: updated after each optimiser step when enabled, evaluation uses EMA weights, `ema_state_dict.pth` stored with the checkpoint.

## P-18 Legacy code and dead paths

k-diffusion / EDM agents, samplers, GPT denoisers, `eval_model*.py`, `wandb_test.py`, the DDIM sampler (calls undefined helpers), `predict` (uses non-existing `self.embedder`), visualisation methods referencing `self.embedder`, `get_train_val_split_iterator` (`np.arrange`). Around half of `diffusion/` is unused by grasping. Remove or move to `legacy/` to make the project readable.

## P-19 No physics-based evaluation

There is no simulation check whether generated grasps actually lift the object. ACRONYM itself was labelled in FleX; Isaac Lab can run thousands of grasp trials in parallel. See [[state/ideas#I-08 Simulation based grasp success evaluation]] and [[research/simulation-isaac-lab]].

## P-20 Panda-only labels

All labels are for the Franka Panda gripper (8 cm stroke, symmetric parallel jaw). The SO-101 has a different, smaller jaw gripper (one moving jaw) and a 5-DoF arm, so neither the grasp distribution nor reachability transfers directly. See [[research/lerobot-so101]] and [[research/roadmap-sim-to-real-so101]].
