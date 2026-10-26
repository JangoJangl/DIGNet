---
title: LeRobot and the SO-101 arm
type: research
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [lerobot, so101, hardware, teleoperation]
related: ["[[research/simulation-isaac-lab]]", "[[research/sim-to-real]]", "[[research/roadmap-sim-to-real-so101]]", "[[state/ideas#I-10 Reachability-aware sampling for a 5-DoF arm]]"]
confidence: medium
sources: ["https://github.com/huggingface/lerobot", "https://huggingface.co/docs/lerobot/envhub_leisaac", "https://wiki.seeedstudio.com/simulate_soarm101_by_leisaac/", "https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html"]
---

# LeRobot and the SO-101 arm

## SO-101

- Open-source low-cost arm (TheRobotStudio + Hugging Face), successor of SO-100. **Five arm joints + one gripper joint** (shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper), Feetech STS3215 serial bus servos. Used as leader (teleoperation) / follower pair.
- Gripper: two fingers where **one jaw moves against a fixed jaw** (a rotating jaw, not a symmetric parallel gripper). The grasp frame therefore is not centred between two symmetric fingers, and the 180 degree finger flip symmetry used for the Panda metric ([[state/problems#P-07 Metric ignores parallel jaw symmetry]]) does not hold exactly.
- Small payload and reach (tens of centimetres; measure for your build, confidence low on exact numbers). Servo backlash and plastic links give millimetre to centimetre level TCP errors, larger than with industrial arms.
- **5-DoF consequence:** an arbitrary 6-DoF grasp pose is generally not reachable; the approach direction is constrained to (roughly) the vertical plane through the base axis, with wrist roll free. Grasp generation must be filtered or conditioned on reachability ([[state/ideas#I-10 Reachability-aware sampling for a 5-DoF arm]]). A practical subset: top-down and tilted approaches in the arm plane, with any roll.
- Cameras: kits commonly add a wrist camera; NVIDIA's workshop uses an external Logitech C920 plus the wrist camera for image policies. For this project a **depth** camera (e.g. RealSense D405/D435, Orbbec) is needed; a fixed external camera is easier to calibrate, a wrist depth camera gives closer, less occluded views.

## LeRobot

- Hugging Face library for real-world robot learning: hardware drivers (SO-100/SO-101, Koch, LeKiwi, ...), calibration (`lerobot-calibrate`), teleoperation and recording (`lerobot-record`), the LeRobotDataset format on the Hugging Face Hub, training (`lerobot-train`) and evaluation of policies: ACT, Diffusion Policy, VQ-BeT, TD-MPC, SmolVLA, pi0 / pi0-FAST / pi0.5, GR00T N1.x (list from the docs sidebar, confidence medium).
- Versions seen on 2026-10-25: 0.4.1 (LeIsaac EnvHub page), 0.4.3 (NVIDIA workshop). Check the current release before pinning.
- **EnvHub / LeIsaac** (LightwheelAI): Isaac Lab tasks with the SO-101 (`LeIsaac-SO101-PickOrange-v0`, `-LiftCube-v0`, `-CleanToyTable-v0`, bi-arm cloth folding), teleoperation in sim with the real leader arm, conversion of sim demos to LeRobot datasets. Installed with `pip install 'leisaac[isaaclab] @ git+https://github.com/LightwheelAI/leisaac.git#subdirectory=source/leisaac'`.
- NVIDIA "Train an SO-101 Robot From Sim-to-Real" workshop (repo `isaac-sim/Sim-to-Real-SO-101-Workshop`): vial-to-rack task, domain randomisation, co-training with real data, Cosmos augmentation, SAGE + GapONet actuator modelling, GR00T N1.6 fine-tuning.

## How DIGNet fits

LeRobot policies are end-to-end image policies; DIGNet is a geometric grasp proposer. They combine well:

1. **Modular pipeline (main path):** depth camera -> segmentation -> object cloud -> DIGNet grasps -> discriminator + IK/reachability filter -> motion via IK on the SO-101 URDF -> send joint targets through the LeRobot SO-101 follower driver. LeRobot provides the robot I/O, calibration and episode recording for evaluation videos and datasets.
2. **Grasp-conditioned policy (stretch):** use DIGNet proposals as an extra conditioning input (target pose) for an ACT/Diffusion Policy trained on teleoperated or sim-generated demos; or as automatic demonstration generator in sim (scripted grasps from DIGNet, recorded as LeRobot datasets) for policy training.

## Kinematics tooling

SO-101 URDF/MJCF is published with the hardware repository and in Isaac Lab / LeIsaac assets. IK options: `placo` (used by LeRobot examples for SO-100 end-effector control, confidence medium), `ikpy`, `pinocchio`. Because of 5-DoF, solve position + approach-axis IK and let roll follow, or use a weighted IK that tolerates errors in the unreachable rotational direction.
