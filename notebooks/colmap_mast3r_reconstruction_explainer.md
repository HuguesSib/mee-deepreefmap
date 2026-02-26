# COLMAP + MASt3R Reconstruction — Deep Dive

---

## TL;DR

This notebook merges two independently-drifted reconstruction passes (forward and backward transects of the same reef) into a single, globally-consistent 3D point cloud.

The core idea: **replace COLMAP's classical SIFT features with MASt3R's learned dense features**, use **DINOv2 image retrieval** to find which forward frames overlap with backward frames (loop closures), then let **COLMAP's bundle adjustment** jointly optimize all camera poses across both passes. The final step re-projects MASt3R depth maps through the corrected poses to produce a unified, deduplication-cleaned point cloud.

**Dependency chain:**

```
GoPro frames (EUCM fisheye)
    → Undistort to pinhole
        → DINOv2 descriptors (image retrieval: fw↔bw pair selection)
            → MASt3R inference (dense pixel correspondences per pair)
                → COLMAP database (cameras, images, keypoints, matches)
                    → COLMAP geometric verification (essential matrix filtering)
                        → COLMAP incremental mapping (bundle adjustment)
                            → Optimized C2W poses
                                → MASt3R depth re-integration
                                    → Merged, deduped point cloud (.ply)
```

---

## Why Two Passes?

The existing pipeline ([`reconstruct.py`](../src/reconstruct.py)) processes video segments independently. A reef transect filmed in both directions gives two separately-drifted reconstructions. Each pass has good local consistency but arbitrary global scale and drift, so they cannot simply be concatenated — their coordinate frames are unrelated. The notebook solves this by treating both passes as a single SfM problem.

---

## Theoretical Background

### Camera Models

**EUCM (Enhanced Unified Camera Model)** is used by GoPro Hero 10. It's a generalization of the unified projection model for wide-angle/fisheye lenses:

```
Parameters: [fx, fy, cx, cy, α, β]
```

A 3D point `(X, Y, Z)` is first projected onto a unit sphere with a correction controlled by `β`, then perspective-projected onto the image plane with a shift `α` on the Z-axis. This two-stage model can represent everything from perspective (α=0, β=1) to orthographic to extreme fisheye in a single, differentiable formula. The advantage over polynomial distortion models (OpenCV) is that the mapping is invertible in closed form.

**COLMAP requires a pinhole model** (or its standard distortion variants). So the first thing we must do is *undistort* the images: compute where each output pixel in a flat, distortion-free image maps to in the original fisheye image, and interpolate. The resulting intrinsics are just `[fx, fy, cx, cy]`.

---

### DUSt3R and MASt3R

**DUSt3R** (Grounding Image 3D Structure from Pointmaps — NAVER/INRIA, 2024) is a Vision Transformer trained to take an *uncalibrated image pair* and produce, for each image, a dense **pointmap** — a `(H, W, 3)` array where each pixel holds an estimated 3D coordinate in a canonical coordinate frame anchored to image 0. The key innovation is framing stereo reconstruction as a regression problem rather than a geometric pipeline, enabling it to handle arbitrary baselines, overlapping fields of view, and even monocular scenes.

**MASt3R** (Matching and Stereo 3D Reconstruction — NAVER/INRIA, 2024) extends DUSt3R by also outputting per-pixel **local feature descriptors** (`desc`) alongside the pointmaps. This lets you:
1. Extract dense correspondences between any two images with a simple nearest-neighbor search in descriptor space.
2. Get a confidence score (`desc_conf`) for each correspondence.

This replaces the classical SIFT + FLANN/BF matcher pipeline with a geometry-aware learned matcher — correspondences are implicitly consistent with the predicted 3D structure.

The model used here is the largest publicly available checkpoint:
```
naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
```
- **ViT-Large**: 307M parameter Vision Transformer encoder
- **BaseDecoder**: lighter cross-attention decoder
- **512**: operates on 512-pixel images
- **catmlpdpt**: uses concatenated MLP for depth and point prediction
- **metric**: trained with metric (absolute scale) depth supervision

---

### DINOv2 for Image Retrieval

**DINOv2** (Meta AI, 2023) is a self-supervised Vision Transformer trained via a knowledge-distillation objective on ~142M curated images. Its `[CLS]` token produces a powerful global image descriptor that is excellent for image retrieval and place recognition — similar images in appearance produce nearby embeddings in cosine distance.

Here it's used as a fast, cheap **retrieval stage** to avoid running MASt3R on all O(N²) pairs. Instead:
1. Extract a 384-dim descriptor per frame.
2. Compute a cross-pass cosine similarity matrix (N_fw × N_bw).
3. For each forward frame, search only within a time-prior window in the backward sequence, pick the best match if above a similarity threshold.

The time prior encodes the assumption that the backward pass is roughly a reversal of the forward pass: frame `i` in the forward pass most likely corresponds to frame `N_bw - 1 - (i/N_fw * N_bw)` in the backward pass. Searching ±150 frames around this center handles drift.

---

### COLMAP Structure-from-Motion

**COLMAP** is the de facto standard academic SfM pipeline (Schönberger & Frahm, CVPR 2016). Its incremental mapping works as follows:

1. **Feature extraction**: Find keypoints and compute descriptors per image (SIFT by default). *We replace this with MASt3R.*
2. **Feature matching**: Find putative correspondences between image pairs. *We provide these directly.*
3. **Geometric verification**: For each pair with matches, COLMAP fits an **essential matrix** (E) or **fundamental matrix** (F) using RANSAC. Only geometrically consistent matches (inliers under epipolar constraint) are kept. This is the `verify_matches` call.
4. **Incremental reconstruction**:
   - Pick the best initial pair (good overlap, large baseline).
   - Estimate the relative pose via the essential matrix decomposition.
   - Triangulate 3D points from correspondences.
   - Register new images by solving **Perspective-n-Point (PnP)**: given N known 3D points visible in a new image, find the camera pose.
   - After each new image, run **bundle adjustment** — a nonlinear least-squares optimization (usually Levenberg-Marquardt via Ceres Solver) minimizing reprojection error over all camera poses and 3D point positions simultaneously.

**Prior poses** (from the DA3/legacy SfM) are injected as soft constraints (`prior_q*`, `prior_t*`), which can help COLMAP initialize better but are refined during BA.

---

### Bundle Adjustment

Bundle adjustment jointly optimizes:
- Camera extrinsics (pose) for all images
- Camera intrinsics (optional — focal length here)
- 3D point positions

by minimizing the sum of squared reprojection errors:

```
E = Σ_ij ||π(Rᵢ Pⱼ + tᵢ, Kᵢ) - pᵢⱼ||²
```

where `π` is the projection function, `Pⱼ` is the 3D point, `pᵢⱼ` is the observed 2D keypoint, and `Kᵢ, Rᵢ, tᵢ` are the camera intrinsics, rotation, and translation. This is what eliminates drift: by tying both passes together via cross-pass loop closure matches, BA "pulls" the two trajectories into a single consistent coordinate frame.

---

### The Pair Selection Strategy

**Sequential (log-window) pairs** connect frames at offsets [5, 10, 20, 40]:
- **Close pairs (5, 10)**: high overlap → many robust matches → accurate but poor triangulation (shallow baseline)
- **Wide pairs (20, 40)**: lower overlap → fewer matches → but good parallax for triangulation

Both are needed: close pairs initialize the match graph reliably; wide pairs give COLMAP the geometry it needs to triangulate well.

**Cross-pass pairs** link the fw and bw trajectories. These are the *loop closures* — the same reef location seen from opposite directions. Without them, the two passes would remain disconnected subgraphs in COLMAP's scene graph and BA couldn't merge them.

---

### Point Cloud Post-Processing

After re-projecting MASt3R depth maps with COLMAP-optimized poses:

**Voxel downsampling**: Space is divided into a regular 3D grid (voxel size ≈ scene_extent / 1600). Within each occupied voxel, one representative point is kept. This removes redundancy without losing coverage and makes subsequent processing tractable.

**Statistical outlier removal**: For each point, the mean distance to its 20 nearest neighbors is computed. Points where this distance exceeds 2 standard deviations above the global mean are removed. This eliminates floating noise points that arise from low-confidence MASt3R predictions.

---

## Cell-by-Cell Walkthrough

---

### Cell 1 — Marimo setup
```python
import marimo as mo
```
Boilerplate for the [Marimo](https://marimo.io) reactive notebook runtime. Marimo re-executes only downstream cells when an upstream cell changes (DAG-based reactivity), which is especially useful here where early cells (COLMAP mapping) are slow and you don't want to re-run them just to tweak visualization.

---

### Cell 2 — Path setup & imports
```python
MAST3R_REPO_PATH, DUST3R_PATH, SRC_PATH → sys.path
```
MASt3R and DUSt3R use absolute imports internally (e.g. `from dust3r.inference import ...`), so both repos must be on `sys.path`. DUSt3R is vendored *inside* the MASt3R submodule. The local `src/` directory is also added for `sfm.inverse_warp`.

---

### Cell 3 — Configure paths
```python
output_fw_dir, output_bw_dir, colmap_workdir
```
Sets the input directories (outputs of the existing per-pass pipeline) and the working directory for all COLMAP artifacts. Both passes were reconstructed separately by `reconstruct.py` and their outputs live in sibling directories.

---

### Cell 4 — Load reconstruction outputs

**What's loaded:**
- `poses_{fw,bw}.npy`: shape `[N, 4, 4]` — camera-to-world (C2W) SE(3) matrices, one per frame
- `intrinsics_{fw,bw}.npy`: shape `[N, 6]` — per-frame EUCM parameters `[fx, fy, cx, cy, α, β]`
- `frames_{fw,bw}`: sorted lists of JPEG paths

The poses come from the DA3 streaming pipeline (or legacy SfM). They define where each camera was in *that pass's* local coordinate frame. Both fw and bw frames are separate videos processed independently, so their world frames are unrelated.

---

### Cell 5 — EUCM → Pinhole undistortion (Step 0)

**Why:** MASt3R was trained on standard perspective images. More critically, COLMAP's geometric verification and bundle adjustment use a pinhole camera model — feeding it fisheye images would cause RANSAC and triangulation to fail.

**How:** `rectify_eucm` (from `src/sfm/inverse_warp.py`) builds an inverse warp map: for each output pixel `(u, v)` in the pinhole image, it back-projects through the pinhole model to get a 3D ray, then forward-projects through the EUCM model to find where in the original fisheye image that ray lands. The image is then sampled there via bilinear interpolation (essentially `grid_sample`).

**Intrinsics used:** The mean over all per-frame intrinsics. Since GoPro intrinsics are essentially fixed (they vary only with zoom which is constant), this is valid.

**Output naming:** `fwd_NNNNNN.jpg` / `bwd_NNNNNN.jpg` — the 6-digit zero-padded local frame index, with a prefix to distinguish the two passes when both live in the same directory.

---

### Cell 6 — Load MASt3R model

```python
model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
```

The **asymmetric** architecture uses the same ViT-Large encoder for both images, but applies the cross-attention decoder in a single direction: image 0 acts as the "anchor" reference frame. Outputs:
- `pts3d`: dense pointmap in image-0 coordinate frame
- `conf`: confidence per pixel
- `desc`: local feature descriptor per pixel
- `desc_conf`: confidence in the descriptor

Loading from HuggingFace Hub downloads the ~400MB checkpoint on first call.

---

### Cell 7 — DINOv2 descriptor extraction

```python
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
```

**Model used:** `dinov2_vits14` — the ViT-Small variant with 14×14 patch size. Smaller and faster than ViT-Large, but the `[CLS]` token is still an excellent global descriptor for retrieval.

**Pre-processing:** resize to 224×224, center crop, normalize to ImageNet stats `(μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])`.

**Output:** A `(N, 384)` array of L2-normalizable descriptors, one per frame, for each pass.

Processing is batched (batch_size=32) and wrapped in `@torch.no_grad()` since we only need inference.

---

### Cell 8 — Cross-pass pair selection + sequential pair construction

**Cross-pass pairs (fw ↔ bw):**

```python
sim[i, j] = cosine_similarity(descs_fw[i], descs_bw[j])
```

For each forward frame `i` (sampled every `FW_STEP=3`), we find the backward frame `j` with maximum cosine similarity within a ±150-frame window centered at the time-reversed position. Only pairs with similarity ≥ 0.85 are kept.

The similarity matrix is visualized as a heatmap; matched pairs appear as red dots. A clean diagonal (or near-diagonal with some offset) indicates the two passes are well-aligned temporally. Gaps indicate sections with poor visual overlap or large appearance change (different lighting, sand clouds, etc.).

**Sequential pairs (log-window):**

For both passes, pairs at offsets `[5, 10, 20, 40]` are generated every 5 frames. This gives:
- Dense overlap for robust matching
- Wide baseline for good triangulation geometry

Pairs are stored as `(min_idx, max_idx)` tuples in a set to avoid duplicates.

**Visualization:** The histogram of cross-pair similarities helps calibrate `SIM_THRESH`. If you see many pairs below 0.85, the threshold may be too high and you'd miss loop closures.

---

### Cell 9 — Build COLMAP SQLite database

COLMAP stores all reconstruction data in a **SQLite database** with a fixed schema. This cell builds the database from scratch.

**Global index convention:**
```
fwd frame i  →  global index i
bwd frame j  →  global index N_fw + j
```
This allows all images to coexist in a single COLMAP DB with unique IDs.

**Tables created:**
- `cameras`: One shared PINHOLE camera (`model=1`), resolution from the rectified images, intrinsics `[fx, fy, cx, cy]`
- `images`: One row per frame — name, camera_id, and **prior pose** (C2W converted to W2C quaternion + translation for COLMAP convention)
- `keypoints`: Placeholder — filled by MASt3R matching cell
- `matches`: Placeholder — filled by MASt3R matching cell
- `two_view_geometries`: Filled by COLMAP's geometric verification

**Pose convention:** COLMAP stores **world-to-camera** (`W2C`). Our poses are **camera-to-world** (`C2W`). Conversion: `W2C = inv(C2W)`. The rotation is then converted from rotation matrix to quaternion in COLMAP's `[qw, qx, qy, qz]` format (note: scipy's `as_quat()` returns `[x, y, z, w]`, so components are reordered).

**Only frames participating in at least one pair** are registered — this avoids polluting the DB with frames that would never get matched.

---

### Cell 10 — MASt3R inference + match export

This is the most compute-intensive cell. For each image pair, MASt3R is run to produce dense correspondences.

**Image loading (`_load_view`):**
- Images are resized to fit within 512px (longest side), rounding to the nearest multiple of 16 (required by the ViT patch size of 14, but padded to 16 for safe tensor alignment).
- Normalized to `[-1, 1]` range.
- Wrapped in the DUSt3R view dictionary format.

**Inference:**
```python
result = dust3r_inference(batch_views, model, device, batch_size=4)
```
`dust3r_inference` handles batching pairs, stacking images, and returning the two heads (`pred1`, `pred2`) containing pointmaps and descriptors.

**Correspondence extraction:**
```python
corres = extract_correspondences_nonsym(descs0, descs1, conf0, conf1, subsample=8)
```
`extract_correspondences_nonsym` does nearest-neighbor matching in descriptor space (cosine similarity). `subsample=8` means only 1/8 of pixels are considered as query points (to save memory/time). Correspondences below `CONF_THR=1.5` are discarded.

**Ravel encoding:** Each keypoint `(x, y)` is encoded as a single integer `ravel_id = x + W * y`. This compact format is used throughout to avoid storing (x,y) pairs explicitly, and maps directly to the linear memory layout of the image tensor.

**Accumulation across batches:** For images that appear in multiple pairs, keypoints and matches are accumulated. Duplicate matches are removed with `np.unique`.

**Scaling back to original resolution:** MASt3R operates at 512px; COLMAP needs keypoints in the original (undistorted) image resolution. Scale factors `sx = W_orig / mast3r_W`, `sy = H_orig / mast3r_H` convert coordinates.

**DB export:**
- Keypoints are written as `(x, y)` float32 arrays per image.
- Matches are written as pairs of keypoint indices `(kp_idx_0, kp_idx_1)` using COLMAP's pair ID encoding: `pair_id = id_lo * 2147483647 + id_hi`.

---

### Cell 11 — COLMAP geometric verification + incremental mapping

**Step 1 — Geometric verification (`pycolmap.verify_matches`):**

COLMAP iterates over all match pairs and fits an **essential matrix** `E` using RANSAC. The essential matrix encodes the epipolar constraint between two calibrated cameras:

```
p'ᵀ E p = 0
```

where `p, p'` are normalized image coordinates. Points satisfying this (up to noise threshold) are *geometric inliers*. This step filters out MASt3R false matches that happen to be in descriptor space but violate epipolar geometry — typically texture-repetition confusions or low-confidence matches.

**Step 2 — Incremental mapping (`pycolmap.incremental_mapping`):**

Options used:
- `multiple_models=False`: Force a single connected reconstruction (don't split into sub-models)
- `ba_refine_focal_length=True`: Allow BA to refine `fx, fy` — useful since rectification may introduce small errors
- `ba_refine_principal_point=False`: Keep `cx, cy` fixed (usually well-determined)
- `ba_refine_extra_params=False`: No distortion params (pinhole model, already undistorted)

The mapper:
1. Selects the initial image pair (best overlap + baseline trade-off)
2. Decomposes the essential matrix into `(R, t)` with cheirality disambiguation
3. Triangulates initial 3D points
4. Registers remaining images via PnP + RANSAC
5. Runs **global bundle adjustment** periodically and at the end

Output: a `Reconstruction` object containing optimized `Image` objects (with `cam_from_world()` poses) and `Point3D` objects (triangulated 3D points with RGB color).

**Failure mode:** If too few cross-pass pairs passed geometric verification, the fw and bw sub-graphs may remain disconnected. COLMAP would then produce two sub-models and fail with `multiple_models=False`. Lowering `CONF_THR` or `SIM_THRESH` would help.

---

### Cell 12 — Extract and visualize optimized poses

**Pose extraction:** For each registered image, `img.cam_from_world()` returns the `W2C` rigid body transform as a `Rigid3d` object. Its `matrix()` gives the 3×4 `[R | t]` matrix. We embed it in 4×4 and invert to get C2W.

**Name-to-index parsing:** Image names like `fwd_000050.jpg` are parsed back to `(prefix, local_idx)` and mapped to `(opt_poses_fw, opt_poses_bw)` dictionaries.

**Visualization:**
- Left plot: optimized 3D trajectories for both passes, now in a *single* coordinate frame
- Right plot: triangulated sparse 3D point cloud colored by RGB, with trajectories overlaid

A good reconstruction shows the fw and bw trajectories tracking the same path (roughly mirrored), with the sparse cloud forming a coherent reef scene. Drift artifacts appear as the two trajectories diverging near their endpoints.

---

### Cell 13 — Depth re-integration with optimized poses

**Why re-integrate?** The original per-pass depth maps were projected using drifted poses from the initial pipeline. Now we have globally consistent poses from COLMAP. Re-projecting the same MASt3R depth estimates through the new poses gives a unified, overlap-free point cloud.

**How MASt3R depth is used:** For each sequential frame pair `(i, i+1)`, DUSt3R/MASt3R is run to get `pts3d` — the pointmap in camera-0's coordinate frame. Confident points (conf > 2.0, finite values) are transformed to world space:

```python
pts_world = C2W[:3, :3] @ pts_cam.T + C2W[:3, 3:4]
```

where `C2W` is now the COLMAP-optimized pose for that frame.

**Voxel downsampling:**
```python
voxel_size = scene_extent / 1600
```
Auto-scaled to scene size so the resolution adapts to the reef's physical extent (whether 10m or 100m).

**Statistical outlier removal:**
```python
pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
```
Removes isolated floating points from noisy MASt3R predictions in low-texture regions (sand, open water).

**Output:** `merged_optimized.ply` — a single Open3D point cloud containing the entire reef transect with globally consistent geometry.

---

## Data Flow Summary

```
poses.npy [N,4,4]  ──────────────────────────────────────────┐
intrinsics.npy [N,6]  ────────────────────────────────────────┤
frames/*.jpg  ──────────────────────────────────────────────────┤
                                                                 ▼
                                              Cell 5: EUCM undistortion
                                              images_rect/{fwd,bwd}_*.jpg
                                                                 │
                             ┌───────────────────────────────────┘
                             ▼                            ▼
               Cell 7: DINOv2 [N,384]       Cell 6: MASt3R ViT-Large loaded
                             │
                             ▼
               Cell 8: Pair selection
                  cross_pairs (fw↔bw)
                  seq_pairs (logwin)
                             │
                             ▼
               Cell 9: COLMAP DB
                  cameras, images w/ priors
                             │
                             ▼
               Cell 10: MASt3R inference
                  → keypoints, matches → DB
                             │
                             ▼
               Cell 11: COLMAP
                  geometric verification (RANSAC E-matrix)
                  incremental mapping (PnP + bundle adjustment)
                             │
                             ▼
               Cell 12: Optimized C2W poses (fw + bw unified frame)
                  + triangulated sparse point cloud
                             │
                             ▼
               Cell 13: MASt3R depth re-integration
                  per-pair depth → project via opt poses
                  voxel downsample + outlier removal
                             │
                             ▼
               merged_optimized.ply
```

---

## Key Design Decisions & Trade-offs

| Decision | Alternative | Reason |
|---|---|---|
| DINOv2 for retrieval | Run MASt3R on all pairs | O(N²) MASt3R pairs would be infeasible (~hours); DINOv2 is fast (seconds) |
| Log-window sequential pairs | Only adjacent pairs | Adjacent-only gives shallow parallax; wider gaps needed for triangulation |
| Mean intrinsics for undistortion | Per-frame intrinsics | GoPro zoom is fixed; mean is identical in practice and simpler |
| `CONF_THR=1.5` for MASt3R | Higher/lower | Too high → too few matches → COLMAP fails; too low → noisy matches mislead BA |
| `SIM_THRESH=0.85` for cross-pairs | Higher/lower | Trade-off between completeness (more loop closures) and precision (no wrong pairs) |
| `ba_refine_focal_length=True` | Fixed focal length | Small systematic error in undistortion is absorbed by focal length adjustment |
| Voxel size = extent/1600 | Fixed voxel size | Auto-adapts to any scene scale without manual tuning |
| MASt3R depth for final cloud | COLMAP triangulation | MASt3R gives dense depth; COLMAP triangulation is sparse |
