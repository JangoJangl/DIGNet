---
title: Point cloud encoders
type: research
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [point-cloud, encoder, equivariance]
related: ["[[state/models]]", "[[state/ideas#I-05 Stronger point cloud encoder]]", "[[research/grasp-generation-sota]]"]
confidence: medium
---

# Point cloud encoders

| Encoder | Year | Key property | Fit for DIGNet |
|---|---|---|---|
| PointNet++ (MSG) | 2017 | hierarchical ball-query grouping, simple, robust | current baseline; global feature only loses local geometry needed for precise grasps |
| DGCNN | 2019 | dynamic kNN graph, edge convolutions | base of the VN encoder in the repo |
| Vector Neurons (VN-DGCNN) | 2021 | exact SO(3) equivariance by 3D vector features | in repo (`vnn_pointnet2.py`); equivariance is lost because the MLP denoiser on top is not equivariant. Either use invariant features + express the grasp in a canonical frame, or keep an equivariant head |
| PointNeXt | 2022 | modernised PointNet++ training recipe, strong at small cost | vendored in `object_representation/openpoints`; easy upgrade |
| Point Transformer V3 (PTv3) | 2024 | serialised (space-filling curve) attention, fast, state of the art on segmentation | used as a GraspGen backbone option; strong but heavier dependency (spconv / flash-attention) |
| Sonata / self-supervised PTv3 (confidence medium) | 2025 | self-supervised 3D features | frozen features + small adapter for low-data SO-101 fine-tuning |
| Point-MAE | 2022 | masked autoencoding pre-training | already wrapped in `object_representation/models/pc_mae` |

## Recommendations

1. Keep PointNet++ as baseline but return **per-point features** and let the denoiser attend to them (cross-attention from the noisy grasp token to the k nearest points of the current grasp position). This is what the hierarchical embedder attempted and what Transformers do natively.
2. For arbitrary object orientations prefer **augmentation + a stronger non-equivariant encoder** (simplest, what GraspGen does) over strict equivariance, unless data is tiny.
3. Input size: 1024 to 2048 points after voxel downsampling (5 mm) is enough for objects SO-101 can grasp; normalise by centring on the visible-point mean (matches the new `partial_view` augmentation).

## Implemented encoders and optimisations

Implemented on 2026-10-25 (see [[state/models]]). Both encoders return tokens `[B, M, 3 + C]` and are used with the `CrossAttentionDenoiser`.

### PointNet2Local (`networks/vision_encoder/pointnet2_local.py`)

Two MSG set abstraction levels (512 -> 128 centroids), offsets divided by the ball radius (PointNeXt), linear + LayerNorm projection to `token_dim`. Kept deliberately small.

Possible optimisations, not implemented:
- CUDA kernels for farthest point sampling and ball query (`pointnet2_ops` / PyTorch3D `sample_farthest_points`, `ball_query`): the Python FPS loop is the main cost on GPU and very slow on CPU.
- PointNeXt InvResMLP blocks (residual, separable MLPs) and a deeper stage stack; PointNeXt reports large gains from the training recipe (label smoothing, AdamW, cosine schedule, height/normal features) more than from architecture.
- Normals or a learned per-point embedding as input features (`use_normals: true`, needs normals from the depth image at test time).
- More tokens (256) for large objects, or keep both levels (512 and 128) as tokens.

### PTv3Encoder (`networks/vision_encoder/ptv3.py`)

Kept from Point Transformer V3 (Wu et al., CVPR 2024, arXiv 2312.10035, confidence high for the items below):
- serialisation of the cloud along space filling curves, here z-order and transposed z-order (PTv3 uses Z, TZ, Hilbert, TH),
- attention inside non-overlapping patches of the serialised sequence instead of kNN attention; PTv3 shows accuracy rising with patch size up to 1024 (ScanNet mIoU 75.0 at 16 to 77.3 at 1024),
- a different curve per block, shuffled per forward pass ("shuffle order"),
- a conditional positional encoding before every attention (PTv3 xCPE = sparse convolution with skip connection; RPE was dropped because it cost 26 % of the forward time),
- pre-norm blocks with LayerNorm, BatchNorm in pooling (PTv3: BN in pooling cannot be replaced by LN), MLP ratio 4.

Simplified here: dense tensors with a fixed point count, pooling merges `stride` consecutive points of the z-order sequence (PTv3: grid pooling with x2 grid multipliers), xCPE replaced by a kNN max-pool point convolution, encoder only (no decoder), defaults scaled down for 1024-point object clouds (channels 32 to 256, depths 1-1-2-1; PTv3 default 64 to 512 with depths 2-2-6-2, heads 4 to 32, drop path 0.3).

Possible optimisations, not implemented:
- Use the official implementation (Pointcept / `PointTransformerV3` repository) with spconv xCPE, Hilbert curves, flash attention and packed variable-size batches. PTv3 reports 3.3x faster and 10x less memory than PTv2.
- Self-supervised pre-trained weights (Sonata, 2025) and fine-tune only an adapter on grasp data.
- Drop path / stochastic depth (PTv3 uses 0.3) once the model is large enough to overfit.
- `torch.compile` and bf16 autocast for the attention blocks (works once the global float64 default is removed, [[state/problems#P-09 Global float64]]).
- Use the kNN of the first stage for a local refinement head at grasp contacts.

### Cross-attention denoiser (`networks/diffusion/attention_denoiser.py`)

One query token from `[x_t, time embedding]` plus a max-pooled global token, 2 cross-attention + MLP layers over the k = 32 tokens nearest to the current grasp position.

Possible optimisations: self-attention between grasp queries of the same object (set-level generation as in GraspGen's transformer), FiLM / adaLN time conditioning instead of concatenation, attention over all tokens with a distance bias instead of a hard top-k, querying tokens at both predicted contact points instead of the grasp centre.
