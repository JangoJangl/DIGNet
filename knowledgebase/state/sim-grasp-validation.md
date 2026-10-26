---
title: Grasp validation in simulation (Isaac Lab 3, Newton)
type: state
status: active
created: 2026-10-26
updated: 2026-10-26
tags: [simulation, isaac-lab, newton, evaluation, so101, franka]
related: ["[[research/simulation-isaac-lab]]", "[[state/ideas#I-08 Simulation based grasp success evaluation]]", "[[state/unified-datasets]]", "[[research/lerobot-so101]]"]
sources: [sim_eval/README.md, sim_eval/run_newton.py, sim_eval/grippers.py, sim_eval/export_grasps.py, sim_eval/objects.py, sim_eval/report.py, tests/test_sim_eval.py]
---

# Grasp validation in simulation

Implements [[state/ideas#I-08 Simulation based grasp success evaluation]]: grasps predicted on the validation split are executed with a flying Franka Hand and a flying SO-101 gripper; per grasp success is 1 if the object is still in the gripper, else 0. Usage: `sim_eval/README.md`.

**Status 2026-10-26:** code and CPU tests done (geometry, transfer, export, reports). The Isaac Lab script has not run on a GPU yet; the first step is the calibration with dataset grasps.

## Pipeline

1. `export_grasps.py` (DIGNet environment): for each validation observation (full and partial clouds) the model samples grasps; they are mapped back to the mesh frame (`grasp_mesh = camera_pose @ T(centre) @ pred @ T_z(-tcp_depth)`, undoing `preprocess_cloud`). The same object gets exactly `grasps_per_object` grasps, drawn from all its views. `dataset` mode exports the dataset grasps with their labels instead.
2. `run_newton.py` (Isaac Lab environment, one process per chunk of objects): one environment per grasp.
3. `evaluate.py` merges the chunks into `results.csv` and `summary.json`; `watch_checkpoints.py` does 1 to 3 for every new Hub checkpoint and logs `sim/success_<gripper>` to wandb.

## Simulation protocol

- Backend: Isaac Lab 3.0 kit-less, Newton with the MuJoCo-Warp solver (`newton_mjwarp`), the solver settings of Isaac Lab's Franka lift task (implicitfast, Newton collision pipeline, 2 substeps) at dt 1/240 s.
- The gripper root is fixed in its environment; the object is placed relative to it (the grasp pose). Gravity is off and replaced by an inertial load on the object, which is what lifting and shaking the gripper does in the gripper frame.
- Phases: settle 0.1 s open (object moving > 5 mm = collision with the open gripper, counts as failure), close 0.6 s with limited effort, then 6 x 0.4 s load `m g` along ±z (pull out / push in), ±x (closing axis), ±y (along the pads).
- Success: centre of mass moved < 2 cm relative to the gripper since closing and the fingers did not close completely.
- Objects: mass from the volume at 400 kg/m^3, clipped to 0.02 to 1 kg. Collision by CoACD convex decomposition (default, cached in `~/.cache/dignet_sim`), or `convex_hull` / `sdf`.
- Friction 1.0 on all shapes (Newton needs friction authored, an unset value is 0).

All of these are assumptions to check with the calibration run (dataset grasps: simulated success of label 1 should be high, of label 0 low).

## Grippers

| | Franka Hand | SO-101 |
|---|---|---|
| Asset | URDF from `gripper/models/panda_gripper` (hand.stl, finger.stl), already in the grasp convention frame | gripper links of `so101_new_calib.urdf` (TheRobotStudio SO-ARM100), CoACD collision parts |
| Actuation | 2 prismatic fingers, 0 to 0.04 m, 70 N each | 1 revolute jaw against a fixed jaw, -0.17 to 1.75 rad, 1.5 N m (guess, measure) |
| Grasp transfer | identity | see below |

SO-101 geometry (measured on the URDF meshes, frame `gripper_frame_link`): approach +z, the jaw tips end at z ≈ +0.005, pad centre used at z = -0.010, fixed jaw inner surface at x = +0.002, the moving jaw opens towards -x; its inner surface at the pad depth goes from x = -0.008 (q = -0.15) to x = -0.072 (q = 0.75). Above 0.75 rad the jaw tip leaves the pad depth, so the usable opening is about 7.4 cm.

Transfer of a Panda grasp: same approach and closing axes, the Panda pad centre depth (0.1034 behind the base) maps to the SO-101 pad centre, the object surface on the +x side of the Panda closing line (ray cast on the mesh) is put 2 mm inside the fixed jaw, and the moving jaw is opened 8 mm beyond the -x surface. Objects wider than the opening are still simulated (they fail) and flagged with `fits = 0`, so `sim/success_so101` and `sim/success_fits_so101` are both reported.

## Open points

- First GPU run: `run_newton.py` against the real Isaac Lab 3.0.0 install (API checked against the v3.0.0 sources, not executed).
- Calibration of friction, effort, contact settings against the dataset labels.
- SO-101 servo torque and the jaw geometry of the printed parts.
- Performance: the 6 GB laptop GPU limits the environments per chunk; CoACD of new objects takes about 10 s each (cached).
