---
title: Simulation with Isaac Lab
type: research
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [simulation, isaac-lab, isaac-gym, data-generation, rl]
related: ["[[research/sim-to-real]]", "[[research/lerobot-so101]]", "[[research/roadmap-sim-to-real-so101]]", "[[state/ideas#I-08 Simulation based grasp success evaluation]]", "[[state/ideas#I-09 SO-101 gripper grasp dataset from simulation]]"]
confidence: medium
sources: ["https://github.com/isaac-sim/IsaacLab", "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/download.html", "https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-a-robot-arm/02-so101-robot.html", "https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html"]
---

# Simulation with Isaac Lab

## Isaac Gym vs Isaac Lab

- **Isaac Gym (Preview 4)** was NVIDIA's standalone GPU-parallel RL simulator. It is deprecated and no longer developed; new projects should not start on it.
- **Isaac Lab** (formerly Orbit) is the successor: a robot learning framework on Isaac Sim with GPU-parallel environments, tiled camera rendering (RGB, depth, segmentation for thousands of envs), domain randomisation through event managers, RL wrappers (RSL-RL, skrl, rl-games, Stable-Baselines3) and imitation learning tooling.
- Versions (checked 2026-10-25, confidence medium): Isaac Sim **6.2.0** is listed as the latest release (October 2026). NVIDIA's 2026 Isaac Lab tutorials link Isaac Lab **3.0** documentation and use the **Newton** physics engine (MuJoCo-Warp solver) for an SO-101 task. The NVIDIA SO-101 sim-to-real workshop pins Isaac Sim 5.1.0 + Isaac Lab 2.3.0 + LeRobot 0.4.3; the Seeed LeIsaac guide used Isaac Sim 4.5 + Isaac Lab 2.1. Pick one consistent pin per project and record it here.
- Hardware: an RTX GPU is required (the NVIDIA workshop recommends Ada or Blackwell). Isaac Lab runs on Linux and Windows; Python 3.11 for recent versions, so DIGNet's Python 3.8 environment must be upgraded or kept separate ([[state/ideas#I-12 Modernise the environment]]).
- Alternatives: **ManiSkill3** (SAPIEN, very fast GPU rendering, permissive licence), **MuJoCo / MJX / MuJoCo Playground**, **Genesis**. All can play the same three roles below; Isaac Lab is preferred here because of the existing SO-101 assets (Isaac Lab `SO101_CFG`, LeIsaac) and the NVIDIA SO-101 tutorials.

## Three ways to use simulation for point cloud grasping

### A. Data engine (recommended core)

1. Object set: ACRONYM/ShapeNetSem meshes (already used) or Objaverse subsets, rescaled to what the SO-101 jaw can hold (roughly 1 to 6 cm grasp width, light objects; measure the real gripper).
2. Grasp candidates: antipodal sampling on the mesh (contact pairs with surface normals inside the friction cone), oriented for the SO-101 gripper frame.
3. Verification: thousands of parallel envs, each with a free-floating SO-101 gripper (no arm) at a candidate pose: close, lift, shake, check the object is still between the jaws. Store successes and failures in DIGNet's npz format.
4. Observations: render single-view depth from randomised cameras with the tiled camera, convert to object clouds with the segmentation mask, apply depth noise models.

This directly yields the SO-101 dataset ([[state/ideas#I-09 SO-101 gripper grasp dataset from simulation]]) and negatives for a discriminator ([[state/ideas#I-06 Grasp discriminator trained on generator samples]]).

### B. Evaluator

Same free-floating gripper setup, but the candidates come from a DIGNet checkpoint. Gives the success rate per checkpoint and on-generator labels for the discriminator ([[state/ideas#I-08 Simulation based grasp success evaluation]]). With the full arm loaded, also checks reachability and collisions of the approach.

### C. Policy learning

- **RL with privileged teacher, point cloud student**: PPO teacher with object pose / state, then DAgger/behaviour cloning into a student that sees point clouds or a wrist camera. NVIDIA's 2026 Isaac Lab SO-101 tutorial uses exactly this teacher (sim state) to student (wrist image + joint state) distillation for a vial-placing task.
- **Imitation learning from simulated teleoperation**: LeIsaac collects SO-101 leader-arm demonstrations in Isaac Lab and converts them to LeRobot datasets; policies (GR00T N1.5/N1.6 in the guides; LeRobot ACT/Diffusion Policy/SmolVLA in general) are trained on them and co-trained with real data.
- For this project, C is a stretch goal ([[state/ideas#I-14 Closed-loop refinement policy in Isaac Lab]]): a learned closed-loop approach/grasp policy that starts from a DIGNet grasp proposal.

## Practical notes

- Gripper contact: use convex decomposition or SDF collision for objects, tune contact offset and friction; small grippers and light objects are sensitive to solver settings.
- Measure the real SO-101 jaw geometry and stroke and update the USD/URDF; the default asset may not match the printed parts.
- Keep the verification protocol (close force, lift height, shake amplitude) fixed and documented, because success labels depend on it.
