---
title: Change log
type: changelog
status: active
created: 2026-10-25
updated: 2026-10-26
tags: [changelog]
related: ["[[state/problems]]", "[[state/ideas]]"]
sources: [tests/test_core.py]
---

# Change log

## 2026-10-26 Grasp validation in Isaac Lab 3 / Newton

| Change | Files |
|---|---|
| Export of validation predictions (or dataset grasps) to the mesh frame, fixed number of grasps per object | `sim_eval/export_grasps.py`, `sim_eval/episodes.py` |
| Flying Franka Hand and SO-101 gripper assets (URDF, CoACD collision parts), Panda to SO-101 grasp transfer | `sim_eval/grippers.py` |
| Object assets (mass, CoACD convex USD, convex hull, SDF) | `sim_eval/objects.py` |
| Isaac Lab 3 kit-less Newton script: close, load in six directions, success 1/0 per grasp | `sim_eval/run_newton.py` |
| Chunked evaluation, summary, wandb logging (`sim/success_franka`, `sim/success_so101`), Hub checkpoint watcher | `sim_eval/evaluate.py`, `sim_eval/report.py`, `sim_eval/watch_checkpoints.py` |

Verification: `tests/test_sim_eval.py` (10 tests, CPU; the SO-101 asset test needs `SO101_DIR`). `run_newton.py` has not run on a GPU yet. See [[state/sim-grasp-validation]].

## 2026-10-25 (5) Resumable training

`GraspDiffusionAgent.save_training_state` / `resume_training` (model, optimizer, EMA, step, best metric, RNG), config keys `checkpoint_every_n_steps`, `resume`, `wandb.run_id`; pushed to the Hub with the other checkpoints. Test: `tests/test_logging.py::test_training_state_roundtrip_and_resume` (35 tests pass).

## 2026-10-25 (4) Unified datasets and a 20 M parameter flow matching model

| Change | Files |
|---|---|
| Converters ACRONYM / GraspGen / GraspNet-1Billion → one format, full and partial clouds, preview figures, optional Hub upload | `dataset_tools/` ([[state/unified-datasets]]) |
| Training loader for the unified format (mixes several datasets, e.g. full + partial) | `diffusion/environments/diffusion_grasp/unified_dataset.py`, `workspaces/unified_grasp_manager.yaml`, `DiffusionGraspProcessor.preprocess_cloud` |
| `flow_pn2_20m` config: PointNet2Local (`pointnet2_width: 3`, 3.5 M) + cross-attention denoiser (5 x 512, 16.8 M) + SE(3) flow matching, float32, AdamW, EMA | `diffusion/configs/flow_pn2_20m.yaml`, `pointnet2_local.py` (`width`) |
| `dtype` config key (default float64 as before) | `training.py`, `visualize_dataset.py`, `train_config.yaml` |
| `Scaler` works without the ACRONYM statistics files (needed by no_scale on other datasets); PyTorch3D import optional in `helper/representation.py` | `helper/scaler.py`, `helper/representation.py` |
| Test figures in `test_outputs/index.html` (converter previews, model output after a short training) | `tests/conftest.py`, `tests/test_datasets.py`, `tests/synthetic_data.py` |

Verification: 34 tests pass on CPU. The short training (small model, 300 CPU steps, full + partial synthetic ACRONYM) checks the whole path from converter to loader to flow matching to plots; it does not produce a trained model.

## 2026-10-25 (3) Logging, Hugging Face checkpoints, visualisation

| Change | Files |
|---|---|
| Evaluation metrics are logged to wandb only when they are computed (before: stale values on every step); all logs carry `step` | `grasp_diffusion_agent.py` |
| Parameter counts per sub-module (`params/embedder`, `params/model`, `params/total`) in the wandb run summary | `grasp_diffusion_agent.py` |
| Validation plots at `visualize_at` fractions of training (default 25/50/75/100 %): ground truth (blue) vs. sampled (green) grasps on the first `num_vis_objects` test objects, as `val_grasps` image and interactive `val_3d/object_i` point clouds, PNG also under `visualizations/` in the run directory | `grasp_diffusion_agent.py`, `helper/plotting.py` |
| Final evaluation and `final/` checkpoint at the end of training | `grasp_diffusion_agent.py` |
| Checkpoints, `.hydra` config and plots pushed to a Hugging Face model repo (`hub.repo_id`, private by default) on every new best and at the end, under `runs/<wandb run name>` | `helper/hub.py`, agent configs, `train_config.yaml`, `evaluation_config.yaml` |
| Dataset sample plots (preprocessed and augmented, as the model sees them) | `diffusion/visualize_dataset.py`, `dataset_vis` block in `train_config.yaml` |
| Sample viewer cell (grasps coloured by visibility class) | `notebooks/build_partial_acronym_dataset.ipynb` (section 6b) |
| Tests | `tests/test_logging.py` |

```bash
export HF_TOKEN=...   # write token, or huggingface-cli login
python diffusion/training.py hub.repo_id=<user>/dignet-grasp visualize_at=[0.25,0.5,0.75,1.0] num_vis_objects=4
python diffusion/visualize_dataset.py dataset_vis.split=train dataset_vis.indices=[0,7,42] partial_view_prob=1.0 partial_mode=render
```

Verification: `PYTHONPATH=.:diffusion pytest tests` 25 passed on CPU; the agent test trains 8 steps with flow matching, checks the four plots at 25/50/75/100 %, the `final/` checkpoint and the best/final uploads against a fake `HfApi` (nothing uploaded). Flow matching overfit sanity check (one cloud, two grasp modes, 800 steps, CPU): median error 9 mm / 4 deg, samples split 52/48 between the modes. `visualize_dataset.py` was not run on real ACRONYM data.

## 2026-10-25 (2) Encoders, flow matching, rendered partial views, visibility dataset

| Change | Files | Idea |
|---|---|---|
| PointNet++ with radius-normalised offsets and 128 local tokens | `diffusion/networks/vision_encoder/pointnet2_local.py`, `configs/agents/model/embedder/pointnet2_local.yaml` | [[state/ideas#I-05 Stronger point cloud encoder]] |
| Compact Point Transformer V3 encoder (z-order / transposed z-order serialisation, patch attention, shuffled orders, kNN xCPE, serialized pooling) | `diffusion/networks/vision_encoder/ptv3.py`, `embedder/ptv3.yaml` | [[state/ideas#I-05 Stronger point cloud encoder]] |
| Cross-attention denoiser (grasp query, k nearest tokens, global token) | `diffusion/networks/diffusion/attention_denoiser.py`, `configs/agents/model/attention_diffusion_model.yaml` | [[state/ideas#I-05 Stronger point cloud encoder]] |
| SE(3) conditional flow matching (Haar rotation prior, body-frame geodesic path, Euler on the group) | `diffusion/networks/diffusion/flow_matching.py`, `helper/so3.py`, `configs/agents/model/flow_matching_model.yaml` | [[state/ideas#I-04 Flow matching on SE(3)]] |
| Depth rendering with sensor noise, `partial_mode: render`, `eval_partial_view` | `helper/rendering.py`, processor, workspace config, `train_config.yaml` | [[state/ideas#I-03 Train on rendered partial views]] |
| `DiffusionMLPNetwork` honours `output_dim` (default unchanged) | `diffusion/networks/diffusion/diffusion_models.py` | |
| Notebook that builds a partial-view dataset with visibility classes and uploads it to the Hugging Face Hub | `notebooks/build_partial_acronym_dataset.ipynb` | [[state/ideas#I-16 Use the visibility classes of the partial view dataset]] |
| Tests | `tests/test_models.py` | |

Usage examples:

```bash
# DDPM, PointNet++ tokens, cross-attention denoiser
python diffusion/training.py agents/model=attention_diffusion_model
# same with Point Transformer V3
python diffusion/training.py agents/model=attention_diffusion_model agents/model/embedder=ptv3
# flow matching (needs representation=se3_pseudo_log_map_representation, scaler=no_scale, the defaults)
python diffusion/training.py agents/model=flow_matching_model flow_steps=10
# train on rendered partial views, evaluate on rendered partial views
python diffusion/training.py partial_view_prob=0.7 partial_mode=render eval_partial_view=true random_rotation=true
```

Verification: `PYTHONPATH=.:diffusion pytest tests` 22 passed on CPU (both new models x both token encoders compose from the Hydra configs, train one step and sample; SO(3) exp/log round trip incl. angles near pi; serialized attention order handling; renderer only sees camera-facing faces of a box). The notebook ran end to end on a synthetic two-object ACRONYM layout (contacts land on the box faces, visible contacts lie ~3 mm from the observed cloud, hidden ones ~27 mm). Not verified: training on real ACRONYM data on a GPU.

## 2026-10-25 Algorithm and pipeline fixes

Code changes are uncommitted in the working tree of `main` so they can be reviewed with `git diff`.

| Change | Files | Problem / idea |
|---|---|---|
| Beta schedule honoured, unknown names raise | `diffusion/networks/diffusion/{point_diffusion,diffusion,embedder_free_diffusion}.py` | [[state/problems#P-01 Beta schedule is always VP]] |
| Several grasps per object in one batch (embedding repeated per grasp), `forward(..., num_samples=k)` | `point_diffusion.py`, `embedder_free_diffusion.py` | [[state/problems#P-03 Batch size is forced to one grasp]] |
| Evaluation samples `num_eval_samples` grasps per object, no skipped object, optional `max_eval_objects` | `grasp_diffusion_agent.py`, agent configs, `train_config.yaml`, `evaluation_config.yaml` | [[state/problems#P-04 Evaluation sampled one grasp per object]] |
| Vectorised Chamfer metric (identical values), optional symmetric finger assignment | `helper/losses.py` | [[state/problems#P-07 Metric ignores parallel jaw symmetry]], [[state/problems#P-10 Slow data loading and metrics]] |
| Embedder-free model: object cloud in the gripper frame via the rigid inverse; `diffusion_x` path fixed | `embedder_free_diffusion.py` | [[state/problems#P-05 Embedder-free model used the wrong transform]] |
| Rotation bounds pi for unscaled log map representations | `grasp_diffusion_agent.py` (`set_bounds`) | [[state/problems#P-08 Clamping to plus minus one clips rotations]] |
| EMA updated and used for evaluation, saved as `ema_state_dict.pth` | `grasp_diffusion_agent.py` | [[state/problems#P-17 EMA never updated]] |
| Training augmentation: random SO(3) rotation, simulated partial view (hidden point removal + re-centring), point jitter | `helper/augmentation.py` (new), `diffusion_grasp_processor.py`, `dataloader.py`, workspace config, `train_config.yaml` | [[state/problems#P-02 Learns only from full, canonical object clouds]] |
| Device agnostic representation inverses and start transform | `helper/representation.py`, `helper/transform.py` | [[state/problems#P-13 Hard-coded paths and CUDA]] |
| `mean_var_scale` fixed (operator precedence and indexing) | `helper/scaler.py` | |
| `get_current` used a non-existing attribute | `dataloader.py` | |
| Lazy PyTorch3D import, unused scikit-learn import removed in metrics | `helper/losses.py` | |
| New default batch 8 objects x 32 grasps | `train_config.yaml` | [[state/ideas#I-01 Multi-grasp batching and many-sample evaluation]] |
| CPU unit tests | `tests/test_core.py` | |

### Behaviour changes to be aware of

- Training now uses the cosine schedule from the config (before: VP). Old checkpoints need `beta_schedule: vp`.
- Metric values with `symmetric_metric: true` and 100 samples per object are **not comparable** with 2023 numbers.
- Augmentation is off by default.

### Verification

- `PYTHONPATH=.:diffusion pytest tests`: 12 passed on CPU (PyTorch 2.x, Python 3.13), including a train-step + sampling smoke test of `point_diffusion.Diffusion` with 2 objects x 4 grasps.
- Not verified: a full training run on ACRONYM (no GPU or dataset available during the change). The dataloader/processor changes were checked by reading, not executed, because they need ACRONYM, Theseus and Open3D.

- 2026-10-27: sim grasp validation ran on the GPU for the first time; calibration results and next levers in [[state/sim-grasp-validation]].
