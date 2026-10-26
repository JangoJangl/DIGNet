---
title: Configuration and experiments
type: state
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [hydra, config, experiments]
related: ["[[state/training-and-evaluation]]", "[[state/problems]]"]
sources: [diffusion/configs/, architecture_search.bash, params_search.bash, job_template.sh, diffusion/eval_table.csv]
---

# Configuration and experiments

## Hydra layout

```
diffusion/configs/
├── train_config.yaml                 # main config (defaults: agents=diffusion_grasp_agent, workspaces=diffusion_grasp_graspnet_manager)
├── evaluation_config.yaml            # same keys, evaluation: True
├── diffusion_grasp_graspnet_pre_embedding_main_config.yaml  # pre-embedding variant (references a missing agent config)
├── agents/
│   ├── diffusion_grasp_agent.yaml    # -> model: diffusion_model (point_diffusion + vnn_pointnet2)
│   ├── diffusion_grasp_agent_vnn.yaml  # references embedder "vnn_pointnet_2" which does not exist
│   ├── embedderfree_diffusion_grasp_agent.yaml
│   ├── diffusion_loss/{edm,ve,vp}_loss.yaml   # legacy
│   └── model/{diffusion_model, embedder_free_diffusion_model, ...}.yaml + embedder/*.yaml
└── workspaces/diffusion_grasp_graspnet_manager.yaml   # dataset + processor
```

Important keys in `train_config.yaml`: `representation` + `grasp_dim` (must match, see table in [[state/data-pipeline]]), `pc_dim` (points), `embedding_dim` (multiple of 3 for VNN), `n_timesteps`, `pointnet_radius`, `scaler`, `batch_size` (grasps per object), `num_objects`, augmentation keys (`random_rotation`, `partial_view_prob`, `jitter_sigma`), evaluation keys (`num_eval_samples`, `max_eval_objects`, `symmetric_metric`). Dataset paths `root_path`, `data_path`, `h5_path` and wandb entity are user specific.

## Sweeps (SLURM)

- `architecture_search.bash`: representation x {max_scale, no_scale} x radius {0.01, 0.05, 0.1} x steps {10, 100} x points {1024, 4096}. 96 jobs, group `final_param_search`.
- `params_search.bash`: seeds {26, 42, 69} x agent {diffusion_grasp_agent, diffusion_grasp_agent_vnn} with `se3_log_map_representation`.
- `job_template.sh`: BwUniCluster `gpu_4` partition, copies `acronym.tar.gz` to `$TMP`.

## Results in the repository

Only `diffusion/eval_table.csv` (one placeholder row) and the 2023 Hydra output folders in `outputs/`. No result table or checkpoint is committed, so earlier findings can not be reproduced from the repo alone. Because of [[state/problems#P-01 Beta schedule is always VP]], [[state/problems#P-04 Evaluation sampled one grasp per object]] and [[state/problems#P-08 Clamping to plus minus one clips rotations]], earlier sweep conclusions should be re-checked before relying on them.
