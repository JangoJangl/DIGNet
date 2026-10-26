# Contact-Anchored Grasp Diffusion

A 6-DOF parallel-jaw grasp generator that anchors one contact on an observed depth point and uses diffusion to generate the rest of the grasp.

## Core idea

- Every grasp has two contact points, p1 and p2.
- **p1 is fixed** to a point from the depth map, so grasps cannot drift away from the observed object.
- **Diffusion generates the rest**: the second contact p2, the rotation about the p1–p2 axis, and the grasp depth.
- Because diffusion can produce several answers per anchor, the model can represent multiple valid grasps at the same point. Contact-GraspNet regresses a single grasp per point and cannot.

## Diffusion space

The model learns a conditional distribution per anchor point, not one joint distribution over the whole depth map:

    p(Δp2, θ, d | p1, scene features)

Anchors are treated as a batch dimension, like B in GraspGen.

| Variable | Dims | Notes |
|---|---|---|
| Δp2 = p2 − p1 | 3 | Expressed in a local frame built from p1's surface normal. Typically close to −width · normal. |
| Rotation about the p1–p2 axis | 2 | (cos θ, sin θ), which avoids the jump at ±π. Only 1 DOF is left, so a 3D rotation vector would be redundant. |
| Grasp depth d | 1 | How far the fingers reach along the approach direction. Optional: can be fixed as in Contact-GraspNet. |
| **Total** | **5–6** | Per anchor, with no redundant dimensions. |

The grasp pose follows from these values. The center is the midpoint of p1 and p2, shifted by d along the approach direction. The closing axis is p2 − p1, and θ gives the approach direction. The width is |p2 − p1|.

## Option A: Perpendicular contact at p1 (2 DOF)

**Assumption:** the closing axis is perpendicular to the surface at p1, i.e. it is aligned with p1's surface normal n.

This fixes the direction of p2, so only two values are left to generate:

| Variable | DOF | Notes |
|---|---|---|
| Width w | 1 | p2 = p1 − w · n |
| Rotation θ about n | 1 | Encode as (cos θ, sin θ), which is 2 numbers but 1 DOF |
| Grasp depth d | 0 | Fixed |
| **Total** | **2** | |

**Advantages**
- Very small diffusion space, so it trains fast and needs few denoising steps.
- Strong geometric prior: a perpendicular contact is the antipodal ideal, so grasps are stable by construction.
- Easy to label: for each observed point, keep the dataset grasps whose closing axis lies within a small angle of that point's normal.

**Risks**
- **Depends on normal quality.** Normals from noisy single-view depth are unreliable, especially at edges.
  - Mitigation: estimate normals over a larger neighborhood, or have the backbone predict them.
- **Excludes valid grasps.** Grasps that are stable only through friction, where the axis is inside the friction cone but not on the normal, are lost. This matters most on curved or inclined surfaces.
  - Mitigation: add an optional small 2D tilt of the axis relative to n, bounded by the friction cone. This gives 4 DOF in total.
- **p2 may land off the surface.** p2 is not guaranteed to touch the object, especially when the back side is occluded.
  - Mitigation: the discriminator filters these grasps, and the shape-aware encoder from Option B helps place them correctly.

## Option B: Encoder pretrained on shape completion

**Idea:** replace or initialize the backbone with an encoder pretrained to reconstruct complete shapes from partial point clouds. Example methods are masked-point reconstruction (Point-MAE style) and point-cloud completion (PoinTr style). The encoder then already "knows" what the hidden back side of an object looks like, which is exactly what is needed to place p2 and the width w.

**Pretraining**
- Training pairs: a rendered partial view as input and the full mesh point cloud as target.
- Data source: the same object sets as the grasp data (e.g. Objaverse), so there are no extra labeling costs.
- Loss: Chamfer distance between the predicted and the complete point cloud.

**Use in the grasp model**

| Variant | Description | When to use |
|---|---|---|
| Frozen | Train only the grasp heads | Little grasp data, fast training |
| Fine-tuned | Unfreeze the encoder with a lower learning rate | Best accuracy |
| Auxiliary | Keep the completion decoder and train it jointly with the grasp heads | Regularizes features; the completed cloud can also be used to check p2 |

**Advantages**
- Better inference of occluded geometry, which helps p2, w and d.
- The pretraining needs no grasp labels, so it scales with any 3D object dataset.
- The encoder can be reused for every gripper.

**Risks**
- Completion features may not focus on what matters for grasping, such as contact regions and thin parts.
  - Mitigation: use the fine-tuned or auxiliary variant rather than freezing the encoder.
- Domain gap between the noise in rendered pretraining views and real depth.
  - Mitigation: use the same noise and segmentation augmentation as in grasp training.

## Architecture

1. **Input**: depth map, converted to a point cloud and masked to the target object (e.g. with SAM2).
2. **Backbone**: a point-cloud encoder (e.g. PointTransformerV3) produces a feature for every point. It can optionally be pretrained on shape completion (Option B).
3. **Anchor sampling**: K anchors (about 1–2K) chosen on the mask with farthest point sampling, instead of using every pixel.
4. **Diffusion head**: one small network shared by all anchors. It is conditioned on the anchor's feature and position and denoises (Δp2, cos θ, sin θ, d) in about 10 DDPM steps, or (w, cos θ, sin θ) under Option A. Several samples can be drawn per anchor.
5. **Graspability head**: a per-point score for whether any valid grasp exists at that anchor. Diffusion always outputs something, so this head is needed to say "no grasp" at edges, the middle of large flat faces, or points near the table.
6. **Discriminator**: an MLP on the anchor feature plus the generated grasp, outputting a success probability. It is trained **on-generator** as in GraspGen: run the generator on the training objects, label its grasps in simulation, and train on those labels.
7. **Post-processing**: each grasp can appear twice (with p1 and p2 swapped), so remove duplicates or apply NMS, then send the top-K grasps to the motion planner.

## Losses

| Component | Loss |
|---|---|
| Diffusion head | DDPM noise MSE: ‖ε − φ(t, x̃, f_p1)‖² on (Δp2, cos θ, sin θ, d) |
| Graspability head | Binary cross-entropy. Positive if a successful grasp contacts the point within a few mm. |
| Discriminator | Binary cross-entropy on simulated success labels of the generator's own grasps |

## Training data

- Start from a grasp dataset with success and failure labels (e.g. ACRONYM or the GraspGen dataset).
- For each successful grasp, compute its two contacts with the object mesh.
- Assign each grasp to the observed points near its contacts, in **both** directions (p1↔p2). Every observed point then holds a set of grasps, which is the multimodal target for diffusion.
- Points with no grasp nearby are graspability negatives.
- Render partial views from random cameras, with segmentation noise added, for sim-to-real robustness.

## Strengths

- No outlier grasps far from the object, since p1 always lies on observed geometry.
- Handles multimodality per point, unlike Contact-GraspNet.
- Less sensitive to translation scale, because Δp2 is relative to p1 (no κ normalization needed).
- Fits single-view depth naturally: only p1 has to be visible, and p2 can be on the hidden back side.
- Small per-sample diffusion space (5–6 dims), so few denoising steps are needed.

## Limitations

- **Parallel-jaw grippers only.** The contact-pair representation does not fit suction grippers and fits adaptive grippers such as the Robotiq-2F-140 poorly.
- **p2 is often occluded**, so it has to be inferred from shape.
- **Compute grows with the number of anchors × samples × steps**, which is why anchors are subsampled.
- **Fixing the grasp depth** (if d is dropped) reduces the reachable grasp set.

## Related work to check

- **Contact-GraspNet** (Sundermeyer et al., ICRA 2021): dense per-point contact grasps, regressed with no diffusion.
- **M2T2** (Yuan et al., CoRL 2023): a transformer extension of the contact-point representation.
- **Implicit Grasp Diffusion** (Song et al., CoRL 2024): appears to combine dense prediction with diffusion sampling. Likely the closest prior work.
- **GraspGen** (Murali et al., 2025): object-level SE(3) diffusion and the source of on-generator discriminator training.
