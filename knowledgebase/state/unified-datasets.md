---
title: Unified grasp datasets (ACRONYM, GraspGen, GraspNet-1Billion)
type: state
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [data, datasets, acronym, graspgen, graspnet, partial-view]
related: ["[[state/data-pipeline]]", "[[state/changelog]]", "[[state/training-and-evaluation]]", "[[research/sim-to-real]]"]
sources: [dataset_tools/common.py, dataset_tools/sources.py, dataset_tools/mesh_views.py, dataset_tools/build_dataset.py, dataset_tools/preview.py, diffusion/environments/diffusion_grasp/unified_dataset.py, diffusion/configs/workspaces/unified_grasp_manager.yaml, diffusion/configs/flow_pn2_20m.yaml]
---

# Unified grasp datasets

## Which source gives which cloud type

| Source | What it is | Full clouds | Partial clouds |
|---|---|---|---|
| ACRONYM | 8872 meshes, 17.7 M simulated Panda grasps (h5 per object) | yes, surface sampling of the mesh | yes, rendered depth views with sensor noise ([[state/data-pipeline]]) |
| GraspGen (`nvidia/PhysicalAI-Robotics-GraspGen`, CC-BY 4.0) | 8515 Objaverse-LVIS objects, 2000 grasps per object and gripper (Franka Panda, Robotiq 2F-140, suction), WebDataset shards `grasp_data/<gripper>/shard_XXX.tar` with `<uuid>.grasps.json` | yes (meshes from Objaverse via the repo's `download_objaverse.py`, `map_uuid_to_path.json`) | yes, rendered like ACRONYM |
| GraspNet-1Billion | 190 real cluttered scenes, 256 RGB-D images each (Kinect, RealSense), 88 object models with dense grasp labels | yes, from the 88 object models and their object level labels (no scene collisions) | **natively partial**: object points cut from the real depth image with the segmentation mask, scene collision labels applied |

Suction grasps are not converted (no antipodal contacts).

## Format (one row = one observation of one object)

`points` [N, 3]; `grasps` [K, 4, 4] in the same frame. Convention of GraspGen = ACRONYM: origin at the gripper base link, approach +z, closing along x. `tcp_depth` (Panda 0.11217, Robotiq 0.195) moves the origin to the finger centre, which is what DIGNet's processor uses. Other fields:
- Grasp labels: `success`, `quality` (GraspNet: 1.1 - friction coefficient) and `width`.
- Contacts: `contacts` [K, 2, 3] (fingers closed on the mesh by ray casting), `contact_visible`, `contact_cloud_distance`.
- `visibility_class` (2 / 1 / 0 contacts visible, -1 no contact).
- Bookkeeping: `split` (per object by hash; GraspGen and GraspNet use their official splits) and `camera_pose` (camera → object frame, identity for full clouds).

- Full clouds use the object (mesh) frame. Partial clouds use the camera frame (OpenCV).
- Training (`unified_dataset.py`) centres every cloud on its points, so both types look like real observations.
- Visibility: rendered views use line of sight plus the frustum. Real GraspNet views count a contact as visible if an observed point lies within 5 mm, which also accounts for occlusion by other objects.

## Conversion details and assumptions to check on real data

- **GraspNet → our convention.** GraspNet's frame has x = approach, y = closing and origin at the target point; the finger tips are `depth` further along the approach. Converted with `R_ours = R_gn @ [[0,0,1],[1,0,0],[0,1,0]]`. The Panda base is placed `tcp_depth` behind the finger tips (`dataset_tools/sources.py: graspnet_to_ours`, unit tested). Widths above the Panda stroke (8 cm) are dropped.
- **GraspNet segmentation.** The code assumes the segmentation label value is object id + 1, which is how graspnetAPI's masks are commonly used. Check this on the first converted scene with `preview.png`: the grasps have to sit on the object points.
- **GraspGen frames.** The code assumes the transforms are given in the frame of the mesh scaled by `object.scale`, as in ACRONYM. Check with `preview.png` after converting one shard.
- Grasps per observation: `--max-grasps` (default 500), with half of them successful where possible.

## Commands

```bash
python -m dataset_tools.build_dataset --source acronym --cloud full    --root $ACRONYM --output $OUT/acronym_full --preview 6
python -m dataset_tools.build_dataset --source acronym --cloud partial --root $ACRONYM --output $OUT/acronym_partial --views 4 --preview 6
python -m dataset_tools.build_dataset --source graspgen --cloud full --root $GRASPGEN --object-root $OBJAVERSE --output $OUT/graspgen_full
python -m dataset_tools.build_dataset --source graspnet1b --cloud partial --root $GRASPNET --camera kinect --ann-stride 16 --output $OUT/graspnet_partial
python -m dataset_tools.preview $OUT/acronym_partial --num 6 --out check.png
# 20 M parameter PointNet++ + flow matching on full + partial ACRONYM
cd diffusion && python training.py --config-name flow_pn2_20m "unified.paths=[$OUT/acronym_full,$OUT/acronym_partial]"
```

`--push-to-hub <repo>` uploads a converted dataset (private by default). Mind the licences: ACRONYM and GraspNet-1Billion are not redistributed on the Hub by their authors.

## Verification status

All six converters were tested on synthetic copies of the original layouts (`tests/synthetic_data.py`, `tests/test_datasets.py`). The checks: finger centres of successful grasps lie on the object, contacts are found, full clouds see all contacts, and single views miss some. Preview figures are written to `test_outputs/index.html`. Nothing has been run on the real downloads yet.
