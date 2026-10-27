---
title: Repository state overview
type: moc
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [index, state]
related: ["[[state/architecture]]", "[[state/data-pipeline]]", "[[state/unified-datasets]]", "[[state/sim-grasp-validation]]", "[[state/models]]", "[[state/training-and-evaluation]]", "[[state/configuration-and-experiments]]", "[[state/environment-and-dependencies]]", "[[state/problems]]", "[[state/ideas]]", "[[state/changelog]]"]
sources: [README.md, diffusion/README.md]
---

# Repository state overview

DIGNet learns a **conditional generative model of 6-DoF parallel-jaw grasps** (Franka Panda gripper) given an **object point cloud**. A DDPM-style diffusion model denoises a grasp pose vector (translation + rotation parameterisation) conditioned on a point cloud embedding (PointNet++, Vector-Neuron PointNet, or a gripper-centric "embedder-free" PointNet++). Training data are the successful grasps of the ACRONYM dataset (ShapeNetSem meshes, Panda gripper, physics-verified grasps).

## Snapshot (git `main`, last commit `4dd6336 installation info`)

- Language / stack: Python 3.8, PyTorch 2.0.1 + CUDA 11.7, Hydra 1.2 configs, Weights & Biases logging, trimesh / Open3D for geometry, Theseus for Lie groups, PyTorch3D for rotation conversions.
- Entry points: `diffusion/training.py` (train), `diffusion/evaluation.py` (evaluate / visualise, hard-coded run path), SLURM sweep generators `architecture_search.bash`, `params_search.bash`, `job_template.sh`.
- Size: about 430 source files, of which a large share is vendored or legacy code (OpenPoints, k-diffusion / EDM agents from an earlier play-data project, Contact-GraspNet tools).
- Status: research prototype from 2023. The core pipeline runs on a CUDA machine with the dataset at hard-coded paths. Several bugs limit training efficiency and make the reported metrics hard to interpret, see [[state/problems]].

## Notes in this section

| Note | Content |
|---|---|
| [[state/architecture]] | Module map and data flow from mesh to grasp |
| [[state/data-pipeline]] | ACRONYM loading, preprocessing, representations, scaling, augmentation |
| [[state/unified-datasets]] | ACRONYM, GraspGen and GraspNet-1Billion in one format, full vs partial clouds, converters and loader |
| [[state/sim-grasp-validation]] | Grasp success in simulation: Isaac Lab 3 / Newton, Franka and SO-101 flying grippers |
| [[state/models]] | Diffusion core, denoiser networks, point cloud embedders |
| [[state/training-and-evaluation]] | Training loop, metrics, evaluation scripts |
| [[state/configuration-and-experiments]] | Hydra configs, sweeps, what was searched |
| [[state/environment-and-dependencies]] | Install, versions, hard-coded paths |
| [[state/problems]] | Problem log with IDs and status |
| [[state/ideas]] | Improvement ideas with effort and priority |
| [[state/changelog]] | What was changed in the code on 2026-10-25 and how it was tested |
