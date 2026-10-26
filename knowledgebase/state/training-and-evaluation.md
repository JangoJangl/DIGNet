---
title: Training and evaluation
type: state
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [training, evaluation, metrics]
related: ["[[state/models]]", "[[state/problems]]", "[[state/changelog]]"]
sources: [diffusion/training.py, diffusion/agents/diffusion_agents/grasp_diffusion_agent.py, helper/plotting.py, helper/hub.py, diffusion/visualize_dataset.py, helper/losses.py, helper/collision.py, diffusion/evaluation.py]
---

# Training and evaluation

## Training loop (`GraspDiffusionAgent.train_agent`)

- Step based (`max_train_steps`, default 100k), Adam lr 5e-4, weight decay 1e-5, no LR schedule, no gradient clipping.
- Every `eval_every_n_steps` (4000) the agent evaluates on the test loader and stores `model_state_dict.pth` in the Hydra run directory when `diversity + accuracy` improves.
- `wandb.log` is called every step with the last evaluation metrics.
- EMA helper existed but was never updated; since 2026-10-25 `use_ema: true` updates it every `update_ema_every_n_steps` and evaluation runs with the EMA weights.
- Batch: `num_objects` objects x `batch_size` grasps per object. Before 2026-10-25 only `1 x 1` worked (one grasp per PointNet forward pass), see [[state/problems#P-03 Batch size is forced to one grasp]]. New defaults: 8 objects x 32 grasps.

## Metrics (`evaluate`)

All distances use the Panda finger control points (mean Euclidean distance between corresponding points of two grasps):

| Name in code | Direction | Meaning |
|---|---|---|
| `diversity_loss` | GT -> predictions | Mean distance from every ground truth grasp to its nearest generated grasp (recall / mode coverage) |
| `accuracy_loss` | predictions -> GT | Mean distance from every generated grasp to its nearest ground truth grasp (precision) |
| `coverage` | GT -> predictions | Fraction of ground truth grasps with a generated grasp closer than 3 cm |
| `gripper_check_loss` | | For the first 5 test objects: fraction of (up to 100) generated grasps that collide with the point cloud or have no points between the fingers (signed distance on trimesh meshes) |

Known issues (all fixed on 2026-10-25 unless noted):

- Only **one grasp per test object** was sampled (the forward pass produced one sample per point cloud), so coverage and diversity mostly measured a single grasp ([[state/problems#P-04 Evaluation sampled one grasp per object]]). Now `num_eval_samples` (default 100).
- The first test object was skipped (`if i == 0: continue`).
- The Chamfer metric used scikit-learn ball trees with a Python callback metric: very slow. Replaced by a vectorised, chunked torch implementation with identical values (unit-tested against the naive loop).
- A parallel-jaw grasp flipped by 180 degrees about the approach axis is physically the same grasp but was counted as ~10 cm away. New option `symmetric_metric` (default true) takes the minimum over both finger assignments. **Numbers are not comparable with runs before this change.**
- Not fixed: there is no physics-based success rate. The metrics compare against the ACRONYM ground truth set only. See [[state/ideas#I-08 Simulation based grasp success evaluation]].

## Logging, checkpoints, visualisation

- wandb: `train_loss` every step, evaluation metrics every `eval_every_n_steps` and at the end, parameter counts in the run summary.
- Validation plots at `visualize_at` (fractions of `max_train_steps`), EMA weights if enabled: `val_grasps` (two views per object) and `val_3d/object_i`.
- Checkpoints: best Chamfer in the run directory, last in `final/`. With `hub.repo_id` both plus `.hydra/*.yaml` and the plots go to the Hugging Face Hub (`helper/hub.py`, failures only log a warning). Loading back: `huggingface_hub.hf_hub_download(repo_id, "runs/<run>/final/model_state_dict.pth")`.
- Resume: the full training state (model, optimizer, EMA, step, best metric, RNG states) goes to `<run>/last/training_state.pth` every `checkpoint_every_n_steps` (default: every evaluation) and to `final/`, and is pushed to the Hub with the checkpoints. Continue with `resume=<folder or file>` or `resume=<hf_repo_id>:runs/<run>/last/training_state.pth`; `wandb.run_id=<id>` continues the same wandb curves. The dataloader order is not restored (a fresh shuffle).
- Data inspection: `diffusion/visualize_dataset.py` (see [[state/changelog]]).

## Evaluation script

`diffusion/evaluation.py` has the run path hard-coded in the `@hydra.main` decorator and mostly calls visualisation methods; `evaluate.sh` loops over run folders. `eval_model.py` and `eval_model_debug.py` belong to the earlier block-push project and do not work with grasping.

## Tests (new)

`tests/test_core.py`, run with `PYTHONPATH=.:diffusion pytest tests` (CPU, no dataset needed): augmentation geometry, hidden point removal, Chamfer metric vs naive loop, symmetric metric, condition repetition, gripper-frame transform, and an end-to-end train + sample smoke test of `point_diffusion.Diffusion` with several grasps per object.
