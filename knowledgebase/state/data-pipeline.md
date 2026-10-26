---
title: Data pipeline
type: state
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [data, acronym, point-cloud, augmentation]
related: ["[[state/architecture]]", "[[state/problems]]", "[[research/grasp-generation-sota]]"]
sources: [diffusion/environments/diffusion_grasp/dataloader.py, diffusion/processors/diffusion_grasp_processor.py, helper/representation.py, helper/scaler.py, helper/sampler.py, helper/transform.py, helper/augmentation.py]
---

# Data pipeline

## Dataset

- **ACRONYM** (NVIDIA, 2021): 17.7M simulated Panda grasps on 8872 ShapeNetSem objects, each labelled successful/failed by a FleX shake test. DIGNet uses the per-object `grasps/*.h5` (mesh path and scale) plus `mesh_contacts_graspnet/*.npz` produced by `tools/create_contact_infos.py` (keys `grasp_transform`, `successful`, `contact_points`, ...).
- Meshes must be made watertight (Manifold) following the ACRONYM instructions.
- `object_statistics.npz` and `grasp_statistics.npz` in the dataset root are required by the `Scaler` even when `scaler: no_scale` (computed with `dataloader.py` `__main__` / `get_statistics`).

## Splits and items (`GraspnetAcronymDataset`)

- File list split with `random_split(files, [0.98, 0.02], seed 42)` in `get_train_val`.
- `__load_data` loads **every** mesh once at construction to filter objects with more than `subbatch_size` successful grasps and to collect size statistics. This is slow for thousands of objects and runs twice (train and test).
- Train item `i` = (object, the `i`-th chunk of `subbatch_size` successful grasps). Test item = (object, **all** successful grasps).
- Only successful grasps are used. Failed grasps (about half of ACRONYM) are discarded.

## Processing (`DiffusionGraspProcessor.preprocess`)

1. `Transform.start_transform`: centre the mesh at its vertex mean, move grasps to the finger centre frame.
2. Sample `num_points` points (`poisson_disk_sample` via Open3D, falls back to random vertices). Normals and colours are not supported by the Poisson path.
3. **Augmentation (new, training only)**: optional simulated single-view partial cloud (hidden point removal from a random camera, re-centred on the visible points), optional uniform SO(3) rotation of points and grasps, optional Gaussian point jitter. Controlled by `partial_view_prob`, `random_rotation`, `jitter_sigma` in `train_config.yaml`. Defaults are off, so behaviour is unchanged unless enabled.
4. Grasp representation (`helper/representation.py`):

| Name | Dim | Notes |
|---|---|---|
| `se3_pseudo_log_map_representation` (default) | 6 | translation + so(3) log map. Discontinuous at rotation angle pi |
| `se3_log_map_representation` | 6 | full se(3) log map, translation is coupled to rotation |
| `rotation_6d_representation` | 9 | translation + first two rotation columns (continuous, Zhou et al. 2019) |
| `trans_quat_representation` | 7 | translation + quaternion, double cover q = -q |
| `se3_representation` | 16 | flattened 4x4 |

5. Scaling (`helper/scaler.py`): `no_scale` (default), `max_scale`, `var_scale`, `mean_var_scale` (fixed on 2026-10-25). Scale factors are indexed into a statistics vector with hand-chosen indices per representation.

## Rendered partial views (2026-10-25)

`partial_mode: render` replaces hidden point removal with a ray cast depth image (`helper/rendering.py`, Open3D `RaycastingScene`): 160x120 px, 58 degree field of view, camera 0.35 to 0.7 m from the object, axial noise 1 mm + 2 mm x depth^2, 2 % pixel dropout, 50 % dropout at depth edges. The cloud is resampled to `pc_dim` points and re-centred on the visible points; grasps are shifted accordingly. `eval_partial_view: true` applies the same partial rendering (without other augmentation) to the test set. Rendered clouds carry no normals.

## Partial view dataset with visibility classes

`notebooks/build_partial_acronym_dataset.ipynb` renders `VIEWS_PER_OBJECT` views per ACRONYM object and stores per view: camera-frame partial cloud, camera pose, up to 500 balanced grasps (successful and failed), the two finger contacts per grasp (ray cast from the Panda finger pads), a per-contact visibility flag (frustum + line of sight), the distance of each contact to the observed cloud, and the visibility class (2 both, 1 one, 0 no contact visible, -1 no contact on the mesh). Object-level hash split 90/5/5, saved with `datasets.save_to_disk`, optional `push_to_hub` with the user's token.

## Inverse path

`processor.postprocess` = inverse scaler then inverse representation, giving 4x4 poses in the centred object frame. The **diffusion model also calls it inside the denoising loop** (`self.inv_fct`) to get the current grasp translation (embedder lookup) or the full pose (embedder-free model).

## Gaps for real-world use

- Point clouds are sampled from the **complete** mesh surface, in the canonical ACRONYM orientation, centred at the true object centre. A real depth camera gives a partial, noisy, arbitrarily oriented view whose centre differs from the mesh centre. The new augmentation options address this; partial views are still synthetic (no sensor model, no table, no clutter).
- No scene context (table plane, neighbouring objects), so collisions with the support surface cannot be learned.
