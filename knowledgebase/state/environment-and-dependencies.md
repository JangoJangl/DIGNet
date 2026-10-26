---
title: Environment and dependencies
type: state
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [setup, dependencies]
related: ["[[state/problems]]", "[[research/simulation-isaac-lab]]"]
sources: [diffusion/install.sh, environment.yaml, README.md]
---

# Environment and dependencies

## Pinned stack (`environment.yaml`)

Python 3.8.16, PyTorch 2.0.1 (CUDA 11.7), PyTorch3D 0.7.4, PyG 2.3.1, Hydra 1.2.0, equiformer-pytorch 0.3.2, Open3D, trimesh, Theseus, wandb, scikit-learn, acronym_tools (editable install from NVlabs/acronym), ALR SimulationFramework (editable).

## Install

`sh diffusion/install.sh` creates a conda env, installs PyTorch, PyTorch3D, Hydra, then pip packages, then `pip install -e` for `acronym` and `SimulationFramework` found under `~/repos`. Theseus often needs a source build (README hint).

## Issues

- Python 3.8 and CUDA 11.7 are end of life; current Isaac Lab needs Python 3.11 and CUDA 12.x, so DIGNet and Isaac Lab cannot share one environment without an upgrade. See [[state/ideas#I-12 Modernise the environment]].
- Many absolute paths to the original cluster (`/home/temp_store/jandl/...`, `/home/i53/student/jandl/...`) in configs, `evaluation.py`, `Scaler` default, `processor` default.
- `diffusion/README.md` and `install.sh` still describe the earlier play-data project (relay kitchen, BeT, mujoco).
- `pretrained/` holds a PointNeXt config/log but no weights; `object_representation/models/pointnext/checkpoints/` holds two PointNeXt checkpoints used only by the pre-embedding variant.
- `torch.cuda` is assumed in several helpers (`.cuda()` calls in `helper/transform.py` centre/bbox transforms, `helper/losses.graspnet_loss`, hierarchical encoder). The main path was made device agnostic on 2026-10-25.
