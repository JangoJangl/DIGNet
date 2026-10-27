---
title: Research overview
type: moc
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [index, research]
related: ["[[research/grasp-generation-sota]]", "[[research/point-cloud-encoders]]", "[[research/flow-matching-se3]]", "[[research/pose-representations]]", "[[research/simulation-isaac-lab]]", "[[research/sim-to-real]]", "[[research/lerobot-so101]]", "[[research/roadmap-sim-to-real-so101]]", "[[research/references]]"]
---

# Research overview

**Goal:** a point cloud based 6-DoF grasp generator, trained with data and evaluation from GPU-parallel simulation (Isaac Lab, the successor of Isaac Gym), deployed on a real SO-101 arm with a depth camera, using DIGNet as the starting point and LeRobot for robot I/O and data handling.

## Notes

| Note | Question it answers |
|---|---|
| [[research/grasp-generation-sota]] | What is the state of the art for learned grasping from full point clouds, partial point clouds and RGB-D, and where does DIGNet stand? |
| [[research/point-cloud-encoders]] | Which encoder should replace PointNet++ / VN-PointNet? |
| [[research/flow-matching-se3]] | How does flow matching on SE(3) work for grasps, what is implemented, what can be optimised? |
| [[research/pose-representations]] | Which pose representations do grasp generators and VLAs use, are poses embedded or tokenised, and should several grasps be generated jointly? |
| [[research/simulation-isaac-lab]] | Isaac Gym vs Isaac Lab, Isaac Lab 3 with Newton without Isaac Sim, versions, and the three ways to use simulation for grasp learning |
| [[research/sim-to-real]] | What closes the gap between simulated point clouds and a real depth camera on a low-cost arm? |
| [[research/lerobot-so101]] | What LeRobot and the SO-101 provide, and their constraints (5-DoF, small jaw gripper) |
| [[research/roadmap-sim-to-real-so101]] | A concrete 2 to 10 day plan, with tiers, milestones and deliverables |
| [[research/references]] | Paper and repository list |

## Main conclusions

1. **Generator + discriminator + filters** is the dominant recipe for 6-DoF parallel-jaw grasping in 2025-2026 (GraspGen, Contact-GraspNet lineage). DIGNet has only the generator. The fastest quality win is a discriminator trained on generator samples ([[state/ideas#I-06 Grasp discriminator trained on generator samples]]).
2. **Training on partial, sensor-like point clouds** is mandatory for real use; full-object training (DIGNet today) does not transfer to single-view depth.
3. **Use Isaac Lab, not Isaac Gym.** Isaac Gym Preview is deprecated. For grasping, simulation is most valuable as a **data engine** (grasp labelling for the SO-101 gripper, rendered partial depth with domain randomisation) and as an **evaluator** (success rate), rather than for end-to-end RL.
4. **SO-101 constraints dominate the design**: 5 arm DoF (not every 6-DoF pose reachable), a small one-sided jaw gripper, servo backlash and compliance. Grasp generation must be filtered or conditioned on reachability, and labels must come from the SO-101 gripper, not the Panda.
5. LeRobot is the natural layer for teleoperation, calibration, data recording and real-robot control of the SO-101; Isaac Lab + LeIsaac provide an SO-101 digital twin.
