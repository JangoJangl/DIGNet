---
title: Sim-to-real transfer for point cloud grasping
type: research
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [sim-to-real, domain-randomization, depth, calibration]
related: ["[[research/simulation-isaac-lab]]", "[[research/lerobot-so101]]", "[[research/roadmap-sim-to-real-so101]]", "[[state/data-pipeline]]"]
confidence: medium
sources: ["https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html"]
---

# Sim-to-real transfer for point cloud grasping

Point clouds have a much smaller visual gap than RGB, so for an open-loop grasp generator the main gaps are **geometry of the observation**, **calibration**, and **execution** on the robot.

## Gap sources and countermeasures

| Gap | Effect | Countermeasure |
|---|---|---|
| Partial view | model trained on full clouds sees half an object | train on rendered single views; `partial_view_prob` augmentation as a first step ([[state/problems#P-02 Learns only from full, canonical object clouds]]) |
| Depth noise, missing data | holes on thin/shiny parts, flying pixels at edges, depth-dependent noise | sensor noise model in rendering (edge dropout, Gaussian noise growing with depth, random holes), jitter augmentation, statistical outlier removal at test time, optionally learned stereo |
| Segmentation | mask errors add table or neighbour points | train with mask erosion/dilation noise; at test time SAM 2 + depth-based table plane removal (RANSAC) |
| Object frame / centring | real centre unknown | centre on the visible-point mean in training and test (done by the new augmentation) |
| Orientation | real objects lie in arbitrary poses | `random_rotation` augmentation or equivariant model |
| Camera calibration | grasp lands centimetres off | intrinsics from the camera, extrinsics by hand-eye calibration (ArUco/ChArUco board on the gripper, or a marker on the table and the arm touching known points); verify by projecting the gripper model into the image |
| Robot kinematics | SO-101 servos have backlash and link compliance; calibration offsets | LeRobot calibration, measure TCP error at several poses, add a small pre-grasp offset and approach along the grasp z-axis |
| Actuation dynamics | sim gripper closes differently | measure closing speed/force; NVIDIA uses SAGE (per-joint sim-real gap analysis) and GapONet (neural actuator model) for the SO-101 with kp 100, kd 2 at 50 Hz |
| Physics labels | friction and mass differ | randomise friction (e.g. 0.4 to 1.2), object mass, gripper force during label generation, keep only robust grasps (success under most randomisations) |

## Domain randomisation reference (NVIDIA SO-101 workshop, image-based policy)

Per reset: dome light exposure (-4 to 3), colour temperature 2500 to 9500 K, random HDRI, external camera position offset ±2 cm (x, y) and ±1 cm (z), small camera rotation offsets, mat yaw ±0.1, rack pose randomised. Physics randomisation is mentioned but not implemented in the shown code. For a depth-only grasp model the camera pose and depth noise randomisation matter, lighting does not.

## Validation ladder

1. Sim, full cloud, Panda (reproduce DIGNet baseline with the fixes).
2. Sim, rendered partial cloud, Panda: measures the partial-view gap.
3. Sim, rendered partial cloud, SO-101 gripper, free-floating: label quality.
4. Sim, full SO-101 arm with IK and approach: reachability and collisions.
5. Real, fixed known objects, measured success rate over N trials per object.

## Implemented renderer and next steps

`helper/rendering.py` (2026-10-25) ray casts the mesh with Open3D (Embree) from a random camera (0.35 to 0.7 m, 58 degree field of view, 160x120 px, elevation >= -0.2) and applies an axial noise model `sigma = 1 mm + 2 mm * z^2`, 2 % random dropout and 50 % dropout at depth discontinuities. Used for training (`partial_mode: render`), evaluation (`eval_partial_view`) and the visibility dataset notebook.

Possible optimisations, not implemented:
- Calibrate the noise parameters on the real camera: record a flat board at several distances (axial noise vs depth) and a few objects (edge and hole statistics).
- Simulate the active stereo pipeline (IR pattern + stereo matching, e.g. as in DREDS / simulated-stereo work) for realistic holes on dark, shiny and transparent surfaces.
- Render a table plane and neighbouring objects, then segment with a noisy mask (erode/dilate, merge with neighbours) to mimic SAM 2 errors.
- Use the real camera intrinsics and resolution, and the real camera pose distribution relative to the SO-101 workspace instead of a sphere around the object.
- Move rendering to Isaac Lab tiled cameras (thousands of views per second on the GPU) once the Isaac Lab data engine exists ([[research/simulation-isaac-lab]]).
- Cache rendered views offline (the visibility dataset notebook already does this) instead of rendering inside `__getitem__`.

