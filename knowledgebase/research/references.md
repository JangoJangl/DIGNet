---
title: References
type: reference
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [references]
related: ["[[research/grasp-generation-sota]]"]
confidence: medium
---

# References

Links marked (checked) were opened on 2026-10-25. Others are well-known references from memory; verify identifiers before citing.

## Grasp generation and detection

- GraspGen: Murali et al., "GraspGen: A Diffusion-based Framework for 6-DOF Grasping with On-Generator Training", 2025. https://arxiv.org/abs/2507.13097 , https://github.com/NVlabs/GraspGen (checked)
- ACRONYM: Eppner, Mousavian, Fox, "ACRONYM: A Large-Scale Grasp Dataset Based on Simulation", ICRA 2021. https://arxiv.org/abs/2011.09584 , https://github.com/NVlabs/acronym
- 6-DOF GraspNet: Mousavian, Eppner, Fox, ICCV 2019. https://arxiv.org/abs/1905.10520
- Contact-GraspNet: Sundermeyer et al., ICRA 2021. https://arxiv.org/abs/2103.14127 , https://github.com/NVlabs/contact_graspnet
- SE(3)-DiffusionFields: Urain et al., ICRA 2023. https://arxiv.org/abs/2209.03855
- GraspLDM: Barad et al., 2024 (latent diffusion for 6-DoF grasps).
- EquiGraspFlow: Lim et al., CoRL 2024 (SE(3)-equivariant flow matching for grasps).
- M2T2: Yuan et al., CoRL 2023 (multi-task masked transformer for pick and place).
- GraspNet-1Billion: Fang et al., CVPR 2020; IJRR 2023. https://graspnet.net (checked)
- AnyGrasp: Fang et al., T-RO 2023. https://arxiv.org/abs/2212.08333
- TOSC: Wu, Shi, Cai, "Task-Oriented Shape Completion for Open-World Dexterous Grasp Generation from Partial Point Clouds", AAAI 2026. https://arxiv.org/abs/2601.05499 (checked; dexterous hands, flow matching, partial clouds)

## Generative modelling

- DexGraspFlow: "Topology-Aware SE(3) Flow Matching for Cross-Embodiment Dexterous Grasp Generation", 2026. https://arxiv.org/abs/2610.26962 (checked)
- GraspMF: "Fast Generative Grasping via Lie Group-Constrained MeanFlow", 2026. https://arxiv.org/abs/2608.26076 (checked)
- GraspMeanFlow: "SE(3)-Equivariant MeanFlow for Few-Step 6-DoF Grasp Generation", 2026. https://arxiv.org/abs/2608.03295 (checked, abstract and method start)
- Riemannian flow matching: Chen and Lipman, ICLR 2024.

- DDPM: Ho et al., NeurIPS 2020.
- EDM: Karras et al., NeurIPS 2022. https://arxiv.org/abs/2206.00364
- Flow matching: Lipman et al., ICLR 2023. https://arxiv.org/abs/2210.02747
- Rectified flow: Liu et al., ICLR 2023. https://arxiv.org/abs/2209.03003
- Diffusion-QL (code base of DIGNet's DDPM): Wang et al., 2022. https://arxiv.org/abs/2208.06193
- Diffusion Policy: Chi et al., RSS 2023. https://arxiv.org/abs/2303.04137

## Point cloud encoders

- PointNet++: Qi et al., NeurIPS 2017.
- Vector Neurons: Deng et al., ICCV 2021. https://arxiv.org/abs/2104.12229
- PointNeXt: Qian et al., NeurIPS 2022.
- Point Transformer V3: Wu et al., CVPR 2024. https://arxiv.org/abs/2312.10035 (checked)

## Pose representations, action tokenisation, set generation

- Zhou et al., On the Continuity of Rotation Representations in Neural Networks, CVPR 2019. https://arxiv.org/abs/1812.07035
- Pertsch et al., FAST: Efficient Action Tokenization for VLA Models, 2025. https://arxiv.org/abs/2501.09747 (checked)
- Kim et al., OpenVLA-OFT: Fine-Tuning VLA Models, Optimizing Speed and Success, 2025. https://arxiv.org/abs/2502.19645 (checked)
- Motion-Centric Action Frames for VLAs (MCF-Proto), 2026. https://arxiv.org/abs/2605.11809 (checked)
- Xu et al., Dexterous Grasp Transformer (set prediction), CVPR 2024. https://arxiv.org/abs/2404.18135 (checked)
- Group Diffusion: cross-sample collaboration, 2025. https://arxiv.org/abs/2512.10954 (checked)
- Song et al., Implicit Grasp Diffusion, CoRL 2024 (PMLR 270). https://proceedings.mlr.press/v270/song25b.html (checked)
- GraspGen-X: Cross-Embodiment 6-DOF Diffusion-based Grasping, 2026. https://arxiv.org/abs/2606.00998 (checked)
- Pooladian et al., Multisample Flow Matching, ICML 2023; Tong et al., Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport, 2023.

## Perception

- SAM 2: Ravi et al., 2024. https://arxiv.org/abs/2408.00714
- FoundationStereo: Wen et al., NVIDIA, CVPR 2025.

## Simulation, sim-to-real, robot software

- Isaac Lab: https://github.com/isaac-sim/IsaacLab
- Isaac Sim download page (6.2.0, October 2026): https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/download.html (checked)
- NVIDIA, "Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac": https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html (checked)
- NVIDIA Isaac Lab tutorial, SO-101 robot: https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-a-robot-arm/02-so101-robot.html (checked)
- LeRobot: https://github.com/huggingface/lerobot ; LeIsaac EnvHub docs: https://huggingface.co/docs/lerobot/envhub_leisaac (checked)
- Seeed Studio, simulate SO-ARM101 with LeIsaac: https://wiki.seeedstudio.com/simulate_soarm101_by_leisaac/ (checked)
- ManiSkill3: https://github.com/haosulab/ManiSkill
