---
title: Idea log
type: idea-log
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [ideas, improvements]
related: ["[[state/problems]]", "[[research/roadmap-sim-to-real-so101]]", "[[research/grasp-generation-sota]]"]
---

# Idea log

Effort: S < 0.5 day, M 0.5 to 2 days, L 2 to 5 days. Priority reflects the goal of a point cloud grasp generator trained in simulation and deployed on an SO-101.

| ID | Idea | Effort | Priority | Status |
|---|---|---|---|---|
| I-01 | Multi-grasp batching and many-sample evaluation | S | must | done 2026-10-25 |
| I-02 | SO(3) and partial-view augmentation | S | must | done 2026-10-25 |
| I-03 | Train on rendered partial views | M | must | done 2026-10-25 (code, not yet trained) |
| I-04 | Flow matching on SE(3) | M | should | done 2026-10-25 (code, not yet trained) |
| I-05 | Stronger point cloud encoder | M | should | done 2026-10-25 (code, not yet trained) |
| I-06 | Grasp discriminator trained on generator samples | M | must | open |
| I-07 | Scene context and collision filtering | M | should | open |
| I-08 | Simulation based grasp success evaluation | M | must | open |
| I-09 | SO-101 gripper grasp dataset from simulation | L | must | open |
| I-10 | Reachability-aware sampling for a 5-DoF arm | M | must | open |
| I-11 | Float32 and mixed precision | S | should | open |
| I-12 | Modernise the environment | M | should | open |
| I-13 | Few-step sampling | S | could | open |
| I-14 | Closed-loop refinement policy in Isaac Lab | L | could | open |
| I-15 | Clean-up and packaging | M | should | open |
| I-16 | Use the visibility classes of the partial view dataset | M | should | open |
| I-17 | Minibatch OT coupling for flow matching | S | should | open |
| I-18 | Joint set generation of grasps | M | could | open |
| I-19 | Fourier features for the grasp input | S | could | open |

## I-01 Multi-grasp batching and many-sample evaluation

Done, see [[state/changelog]]. Next: tune `num_objects` x `batch_size` for GPU memory, report coverage at several thresholds (1, 2, 3, 5 cm).

## I-02 SO(3) and partial-view augmentation

Done as options (`random_rotation`, `partial_view_prob`, `jitter_sigma`). Recommended first experiment: `random_rotation: true, partial_view_prob: 0.5, jitter_sigma: 0.002` vs baseline, evaluated on full and on partial test clouds (add a `partial` flag to the test set).

## I-03 Train on rendered partial views

**Implemented 2026-10-25:** `helper/rendering.py` (`DepthRenderer`, Open3D ray casting, axial depth noise growing with depth^2, pixel and edge dropout), `partial_mode: render` and `eval_partial_view` in the processor. Open: table plane, clutter, segmentation noise, Isaac Lab tiled cameras, see [[research/sim-to-real#Implemented renderer and next steps]].

Original idea: replace hidden point removal by depth images rendered from the mesh (pyrender / Open3D raycasting / Isaac Lab cameras) with a realistic sensor noise model (edge dropout, depth-dependent noise, missing thin parts), plus a table plane. The object cloud then comes from segmentation of the depth image, exactly as at test time.

## I-04 Flow matching on SE(3)

**Implemented 2026-10-25:** `networks/diffusion/flow_matching.py`, config `agents/model=flow_matching_model`. Design and further optimisations: [[research/flow-matching-se3]].

Original idea: replace the 5-step DDPM by rectified flow / conditional flow matching: R3 for translation and geodesic interpolation on SO(3) for rotation (as in FlowMatching-for-SE(3) / EquiGraspFlow / GraspLDM-style setups). Removes representation discontinuities (P-15), allows 2 to 10 integration steps, and simplifies the code (no beta schedule). Keep the DDPM as a baseline.

## I-05 Stronger point cloud encoder

**Implemented 2026-10-25:** `PointNet2Local` (radius-normalised MSG, 128 local tokens), `PTv3Encoder` (compact Point Transformer V3), `CrossAttentionDenoiser` (grasp query attends to the k nearest tokens). Configs `agents/model/embedder=pointnet2_local|ptv3`, `agents/model=attention_diffusion_model`. Optimisations: [[research/point-cloud-encoders#Implemented encoders and optimisations]].

Original idea: the encoder sees ~1024 points with PointNet++ or a small VN network. Options: Point Transformer V3 (used by GraspGen), PointNeXt (already vendored in `object_representation/openpoints`), or a frozen self-supervised encoder (Sonata, Point-MAE) with a small adapter. Also add **local** conditioning: cross-attention from the grasp token to point features (the hierarchical embedder tried this with a nearest-neighbour lookup).

## I-06 Grasp discriminator trained on generator samples

Train a classifier `D(points, grasp) -> success probability` on ACRONYM positives and negatives **plus samples from the trained generator** labelled by geometry or simulation ("on-generator training", GraspGen 2025). At inference sample N = 200 grasps, score, keep the top-k. Biggest single quality gain in recent literature.

## I-07 Scene context and collision filtering

Condition on (or post-filter against) the scene cloud: table plane and neighbouring objects. Cheap version: reject grasps whose gripper mesh intersects the scene cloud (`helper/collision.py` already has signed distance checks, vectorise them with a voxel/occupancy grid).

## I-08 Simulation based grasp success evaluation

Load the test meshes in Isaac Lab, spawn the free-floating gripper at generated grasps, close, lift and shake, record success. Thousands of parallel environments make this minutes per checkpoint. Gives the metric that matters (success rate) and labels for I-06. See [[research/simulation-isaac-lab]].

## I-09 SO-101 gripper grasp dataset from simulation

Generate an ACRONYM-style dataset for the SO-101 jaw gripper: antipodal sampling on (a subset of) ACRONYM / Objaverse meshes scaled to SO-101 sized objects, verification in Isaac Lab with the SO-101 gripper asset, store in the same npz format so the DIGNet loader works unchanged. NVIDIA's GraspDataGen (Isaac Lab based) is an alternative starting point.

## I-10 Reachability-aware sampling for a 5-DoF arm

The SO-101 has 5 arm joints, so most 6-DoF grasp poses are not reachable exactly (one rotational degree is constrained by the base/shoulder geometry). Options: filter samples by IK feasibility, condition the generator on the robot base position, or apply classifier guidance with an IK residual during sampling.

## I-11 Float32 and mixed precision

See [[state/problems#P-09 Global float64]]. Expect several times faster training on consumer GPUs.

## I-12 Modernise the environment

Python 3.11, current PyTorch with CUDA 12, replace Theseus with PyTorch3D or `roma` for Lie group ops, `uv`/`pip` based install, optional dependencies (Open3D only for visualisation). Needed to share code with Isaac Lab and LeRobot.

## I-13 Few-step sampling

DDIM or DPM-Solver for DDPM, or consistency distillation, to keep sampling of 200 grasps under 50 ms.

## I-14 Closed-loop refinement policy in Isaac Lab

Use DIGNet grasps as pre-grasp targets and train an RL (PPO) residual policy in Isaac Lab for the final approach and closing with point cloud / wrist camera observations (teacher-student distillation). Only worth it after the open-loop pipeline works.

## I-16 Use the visibility classes of the partial view dataset

The notebook `notebooks/build_partial_acronym_dataset.ipynb` labels every grasp per view as both / one / no contact visible. Uses: (1) report generator recall separately per class (grasps on hidden surfaces are much harder), (2) condition the generator or discriminator on the visibility class, (3) restrict training targets to class 2 + 1 for a conservative real-robot model, (4) analyse how many successful grasps are reachable from the visible side only.

## I-17 Minibatch OT coupling for flow matching

Per object, assign the noise samples to the target grasps by a Hungarian or Sinkhorn matching on an SE(3) distance before computing the flow targets (multisample flow matching). This gives straighter paths and fewer Euler steps, without changing the model or the inference. See [[research/pose-representations#3. Several poses per forward pass?]].

## I-18 Joint set generation of grasps

Denoise K grasps of one object together: self-attention between the grasp tokens plus cross-attention to the point tokens, trained on random K-subsets of the ground truth. Compare it with independent sampling plus I-17 at a fixed budget (K = 16, 64, 200) on coverage and precision. Watch for mode collapse. See [[research/pose-representations]].

## I-19 Fourier features for the grasp input

Use Fourier features of the translation (frequencies covering 1 mm to 20 cm) and of the 6D rotation entries as the denoiser input instead of the raw values. This is cheap and helps millimetre precision in MLPs.

## I-15 Clean-up and packaging

Move legacy code to `legacy/`, a single `pyproject.toml`, CLI entry points `dignet-train`, `dignet-eval`, `dignet-infer` (point cloud file in, ranked grasps out), CI running `tests/`.
