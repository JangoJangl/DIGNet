---
title: State of the art in learned 6-DoF grasping
type: research
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [grasping, point-cloud, rgbd, diffusion, flow-matching, sota]
related: ["[[research/point-cloud-encoders]]", "[[state/models]]", "[[state/ideas]]", "[[research/references]]"]
confidence: medium
sources: ["https://arxiv.org/abs/2507.13097", "https://github.com/NVlabs/GraspGen"]
---

# State of the art in learned 6-DoF grasping

Confidence: GraspGen facts were checked against the paper and repository on 2026-10-25 (high). Other entries come from known literature up to mid 2026 and are marked where uncertain; verify numbers before quoting them.

## Problem families

| Family | Input | Output | Typical methods |
|---|---|---|---|
| Object-centric generative | segmented object point cloud (full or partial) | set of 6-DoF grasps for that object | 6-DOF GraspNet (VAE + evaluator), SE(3)-DiffusionFields, GraspLDM, EquiGraspFlow, GraspGen, **DIGNet** |
| Scene-level dense prediction | single-view scene point cloud | per-point grasp (contact + approach + width + score) | Contact-GraspNet, GSNet/graspness, AnyGrasp, EconomicGrasp, M2T2 |
| Image-based (RGB-D) | RGB-D image | planar (4-DoF) or 6-DoF grasps | GG-CNN, GR-ConvNet (planar); GraspNet-1Billion baselines (6-DoF) |
| End-to-end policies | RGB(-D) images + proprioception (+ language) | joint/EE actions | Diffusion Policy, ACT, VLAs (pi0 / pi0.5, GR00T N1.x, SmolVLA, GraspVLA) |

## Object-centric generative models (closest to DIGNet)

- **6-DOF GraspNet** (Mousavian et al., ICCV 2019): CVAE grasp sampler on object point clouds + learned evaluator + gradient-based refinement. Established the sampler/evaluator split. Trained in FleX on partial clouds.
- **SE(3)-DiffusionFields** (Urain et al., ICRA 2023): energy-based model on SE(3), Langevin sampling directly on the Lie group, trained on ACRONYM; used jointly with motion planning costs. Shows that diffusion on the group avoids representation issues (cf. [[state/problems#P-15 Representation discontinuities]]).
- **GraspLDM** (Barad et al., 2024, confidence medium): latent diffusion over grasp poses conditioned on partial object clouds, ACRONYM.
- **EquiGraspFlow** (Lim et al., CoRL 2024, confidence medium): SE(3)-equivariant conditional flow matching for 6-DoF grasps; equivariance replaces rotation augmentation.
- **Grasp Diffusion Network / "learning to grasp on SE(3)" variants** (2024-2025, confidence low): diffusion on SE(3) from partial clouds with a scoring head.
- **GraspGen** (Murali et al., NVIDIA, arXiv 2507.13097, 2025, confidence high):
  - Diffusion Transformer generator over grasp poses conditioned on an object point cloud encoder (PointNet++; PTv3 option in the repository), plus an efficient **discriminator** that scores and filters samples.
  - **On-generator training**: the discriminator is trained on samples of the trained generator, so it learns exactly the failure modes the generator produces.
  - Simulated dataset of more than 53 million grasps (paper) / 57 million (README) on 8515 Objaverse-XL (LVIS) objects for three grippers: Franka Panda, Robotiq 2F-140, a 30 mm suction cup. Dataset under CC BY 4.0.
  - Claims: outperforms prior methods in simulation on singulated objects across grippers, state of the art on the **FetchBench** benchmark (+17%), real-robot results with noisy observations, robust to partial vs complete clouds, 21x less memory than prior art, ~20 Hz before TensorRT. Scene demo with collision filtering.
  - Custom grippers need retraining with a gripper description and a grasp dataset; data generation for pinch grippers goes through NVIDIA **GraspDataGen**, which uses Isaac Lab. A follow-up "GraspGenX" covers more grippers.
  - **Implication for DIGNet:** DIGNet already has the generator half on the same ACRONYM-style data. Adding the on-generator discriminator, partial-cloud training and a stronger encoder brings it to the same recipe; the differentiator can be the SO-101 gripper, reachability-aware generation, and an end-to-end low-cost real deployment.

## Scene-level methods on partial clouds / RGB-D

- **Contact-GraspNet** (Sundermeyer et al., ICRA 2021): per-point grasp prediction on single-view scene clouds, trained on ACRONYM table-top scenes rendered from 10k synthetic scenes. DIGNet's `tools/` folder is derived from this repository, so its scene generation and rendering code can be reused for partial-view training data.
- **GraspNet-1Billion** (Fang et al., CVPR 2020): real RGB-D benchmark (RealSense and Kinect), 97k images, 88 objects, analytic labels. Standard for scene-level 6-DoF detection.
- **GSNet / graspness** (Wang et al., ICCV 2021), **AnyGrasp** (Fang et al., T-RO 2023): dense grasp detection with temporal association, strong real-world baseline (SDK licence restricted).
- **EconomicGrasp** (ECCV 2024, confidence medium): cheaper supervision for dense 6-DoF detection on GraspNet-1Billion.
- **M2T2** (Yuan et al., CoRL 2023): transformer predicting contact grasps and placements in clutter, trained in simulation, zero-shot real transfer.
- **FetchBench** (2024): simulated benchmark for fetching objects from cluttered shelves, used by GraspGen to compare full pipelines.

## RGB-D specific considerations

- Consumer depth (RealSense D4xx, Orbbec) loses thin, dark, shiny and transparent parts; noise grows with distance. Short-range sensors (e.g. RealSense D405) suit small arms with close working distances.
- Learned stereo (e.g. **FoundationStereo**, NVIDIA 2025, confidence medium) and depth completion produce cleaner clouds than on-sensor depth and are a cheap upgrade for the perception front end.
- Object segmentation: **SAM 2** (promptable, video) or text-prompted detectors + SAM give the object mask that turns a scene cloud into an object cloud, which is the input DIGNet expects.

## Flow matching vs diffusion for grasps

Conditional flow matching / rectified flow (Lipman et al. 2023, Liu et al. 2023) trains a velocity field along straight paths and integrates in few steps; on SE(3) it can interpolate geodesically on SO(3). It is now common for grasp and action generation because of fewer sampling steps and simpler training. 2026 grasp-specific variants (DexGraspFlow, GraspMF, GraspMeanFlow) are summarised in [[research/flow-matching-se3]]; DIGNet now has an SE(3) flow matching model ([[state/ideas#I-04 Flow matching on SE(3)]]).

## Where DIGNet stands

| Capability | DIGNet (2023) | 2025-2026 SOTA |
|---|---|---|
| Generator | DDPM, 5 steps, MLP denoiser | Diffusion Transformer / flow matching on SE(3) |
| Encoder | PointNet++ / VN-PointNet, global vector | PTv3 / PointNet++ with local attention |
| Input | full object cloud, canonical pose | partial, noisy, arbitrary pose, clutter |
| Scoring | none | discriminator trained on generator samples |
| Data | ACRONYM Panda (successful only) | tens of millions of sim grasps, several grippers, negatives |
| Evaluation | Chamfer / coverage vs dataset | simulated success rate + real robot |
