import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pathlib import Path
    import sys
    import numpy as np

    # Add mast3r repo to path so internal absolute imports work
    MAST3R_REPO_PATH = Path("/home/hugues/Documents/mee-deepreefmap/mast3r")
    if str(MAST3R_REPO_PATH) not in sys.path:
        sys.path.insert(0, str(MAST3R_REPO_PATH))

    # dust3r is nested inside the mast3r submodule
    DUST3R_PATH = MAST3R_REPO_PATH / "dust3r"
    if str(DUST3R_PATH) not in sys.path:
        sys.path.insert(0, str(DUST3R_PATH))

    # Add src to path for local imports
    SRC_PATH = Path("/home/hugues/Documents/mee-deepreefmap/src")
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))
    return Path, np


@app.cell
def _(Path):
    output_fw_dir_ = Path(
        "/home/hugues/Documents/mee-deepreefmap/output/israel-eilat/2024_Sunrise_fw"
    )

    output_bw_dir_ = Path(
        "/home/hugues/Documents/mee-deepreefmap/output/israel-eilat/2024_Sunrise_bw"
    )

    output_fw_dir = Path(
        "/home/hugues/Documents/mee-deepreefmap/output/israel-eilat/2024_Sunrise_fw_leg"
    )

    output_bw_dir = Path(
        "/home/hugues/Documents/mee-deepreefmap/output/israel-eilat/2024_Sunrise_bw_leg"
    )
    return output_bw_dir, output_fw_dir


@app.cell
def _(np, output_bw_dir, output_fw_dir):
    """
    Load reconstruction outputs.

    fwd_dir/poses.npy         → [N_fwd, 4, 4] C2W poses
    fwd_dir/intrinsics.npy    → [N_fwd, 6] EUCM intrinsics
    fwd_dir/frames/*.jpg      → N_fwd frame images

    bwd_dir/poses.npy         → [N_bwd, 4, 4] C2W poses
    bwd_dir/intrinsics.npy    → [N_bwd, 6] EUCM intrinsics
    bwd_dir/frames/*.jpg      → N_bwd frame images
    """
    poses_fw = np.load(output_fw_dir / "poses.npy")
    intrinsics_fw = np.load(output_fw_dir / "intrinsics.npy")
    frames_fw = sorted(output_fw_dir.glob("frames/*.jpg"), key=lambda p: int(p.stem))

    poses_bw = np.load(output_bw_dir / "poses.npy")
    intrinsics_bw = np.load(output_bw_dir / "intrinsics.npy")
    frames_bw = sorted(output_bw_dir.glob("frames/*.jpg"), key=lambda p: int(p.stem))

    print(f"Forward:  {len(frames_fw)} frames, poses {poses_fw.shape}, intrinsics {intrinsics_fw.shape}")
    print(f"Backward: {len(frames_bw)} frames, poses {poses_bw.shape}, intrinsics {intrinsics_bw.shape}")
    return frames_bw, frames_fw, intrinsics_fw, poses_bw, poses_fw


@app.cell
def _(output_bw_dir, output_fw_dir):
    """Load forward and backward TSDF point clouds."""
    import open3d as o3d

    fw_pcd = o3d.io.read_point_cloud(str(output_fw_dir / "point_cloud_semantic.ply"))
    bw_pcd = o3d.io.read_point_cloud(str(output_bw_dir / "point_cloud_semantic.ply"))
    print(f"fw: {len(fw_pcd.points):,} pts, bw: {len(bw_pcd.points):,} pts")
    return bw_pcd, fw_pcd, o3d


@app.cell
def _(bw_pcd, fw_pcd, o3d):

    """Register bw → fw via FPFH global registration + colored ICP.

    Both clouds are gravity-aligned (Z-up). Parameters auto-scale based on
    point cloud extent so the same code works for both legacy (relative depth)
    and DA3 (metric depth in metres).
    """
    import numpy as _np

    # Auto-scale: target ~1500-2000 points after coarse downsample
    extent = _np.asarray(fw_pcd.points).max(0) - _np.asarray(fw_pcd.points).min(0)
    scene_scale = float(extent.max())
    COARSE_VOX = scene_scale / 80   # ~80 bins along longest axis → ~1-2k pts
    print(f"Scene scale: {scene_scale:.3f}, coarse voxel: {COARSE_VOX:.4f}")

    def _preprocess(pcd, voxel):
        down = pcd.voxel_down_sample(voxel)
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
        return down, fpfh

    fw_d, fw_fpfh = _preprocess(fw_pcd, COARSE_VOX)
    bw_d, bw_fpfh = _preprocess(bw_pcd, COARSE_VOX)
    print(f"Coarse: fw {len(fw_d.points):,}, bw {len(bw_d.points):,}")

    # Step 1: RANSAC with FPFH feature matching
    _reg = o3d.pipelines.registration
    ransac = _reg.registration_ransac_based_on_feature_matching(
        bw_d, fw_d, bw_fpfh, fw_fpfh,
        mutual_filter=True,
        max_correspondence_distance=COARSE_VOX * 3,
        estimation_method=_reg.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            _reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            _reg.CorrespondenceCheckerBasedOnDistance(COARSE_VOX * 3),
        ],
        criteria=_reg.RANSACConvergenceCriteria(100_000, 0.999),
    )
    print(f"RANSAC: fitness={ransac.fitness:.4f}, RMSE={ransac.inlier_rmse:.6f}")

    # Step 2: multi-scale colored ICP refinement (coarse → fine)
    # 3 levels at 1/4, 1/8, 1/16 of coarse voxel
    T = ransac.transformation
    for factor in [4, 8, 16, 32, 64]:
        vox = COARSE_VOX / factor
        dist = vox * 4
        fw_v = fw_pcd.voxel_down_sample(vox)
        bw_v = bw_pcd.voxel_down_sample(vox)
        for p in (fw_v, bw_v):
            p.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))
        icp = _reg.registration_colored_icp(
            bw_v, fw_v, dist, T,
            _reg.TransformationEstimationForColoredICP(),
            criteria=_reg.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100),
        )
        T = icp.transformation
        print(f"ICP vox={vox:.4f}: fitness={icp.fitness:.4f}, RMSE={icp.inlier_rmse:.6f}")
    return T, icp


@app.cell
def _(bw_pcd, fw_pcd, icp, o3d, output_fw_dir):
    """Merge aligned point clouds and save."""
    import copy
    import numpy as _np

    bw_aligned = copy.deepcopy(bw_pcd)
    bw_aligned.transform(icp.transformation)

    # Dedup voxel: must be at least as large as the alignment residual,
    # otherwise duplicated structures from fw/bw survive as separate points
    _extent = _np.asarray(fw_pcd.points).max(0) - _np.asarray(fw_pcd.points).min(0)
    detail_vox = float(_extent.max()) / 1600  # ~1cm for DA3, ~0.001 for legacy
    dedup_vox = max(detail_vox, icp.inlier_rmse)
    merged = fw_pcd + bw_aligned
    merged = merged.voxel_down_sample(dedup_vox)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    print(f"Dedup voxel: {dedup_vox:.4f} (2× ICP RMSE)")

    out_path = output_fw_dir.parent / "merged_fwbw_rgb3.ply"
    o3d.io.write_point_cloud(str(out_path), merged)
    print(f"Merged: {len(merged.points):,} pts → {out_path}")
    return


@app.cell
def _(T, np, poses_bw, poses_fw):
    """Visualise trajectories before and after alignment."""
    import matplotlib.pyplot as _plt

    _poses_bw_aligned = np.array([T @ p for p in poses_bw])

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(12, 5),
                                        subplot_kw={"projection": "3d"})
    for ax, bw_p, title in [
        (_ax1, poses_bw, "Before alignment"),
        (_ax2, _poses_bw_aligned, "After alignment"),
    ]:
        ax.plot(*poses_fw[:, :3, 3].T, "b-", lw=1, label="fw")
        ax.plot(*bw_p[:, :3, 3].T, "r-", lw=1, label="bw")
        ax.set_title(title)
        ax.legend(fontsize=7)
    _plt.tight_layout()
    _plt.show()
    return


@app.cell
def _(frames_fw, intrinsics_fw, mo):
    """Rectification sanity check on frame index 0."""
    import io
    import numpy as _np
    import torch
    from PIL import Image as _PIL
    from sfm.inverse_warp import rectify_eucm

    _pil = _PIL.open(frames_fw[2]).convert("RGB")
    _arr = _np.array(_pil).astype(_np.float32) / 255.0   # H x W x 3
    _H, _W = _arr.shape[:2]

    # rectify_eucm expects batched tensors: [1, C, H, W]
    _img_t = torch.from_numpy(_arr).permute(2, 0, 1).unsqueeze(0)   # [1, 3, H, W]
    _mask_t = torch.ones(1, 1, _H, _W)
    _depth_t = torch.ones(1, 1, _H, _W)
    _intr_t = torch.from_numpy(intrinsics_fw[0].astype(_np.float32))  # [6]

    _rect_img, _rect_mask, _rect_depth = rectify_eucm(_img_t, _mask_t, _depth_t, _intr_t)

    # rect_img is [3, H, W] float32 numpy
    _rect_uint8 = (_rect_img.transpose(1, 2, 0).clip(0, 1) * 255).astype(_np.uint8)
    _pil_rect = _PIL.fromarray(_rect_uint8)
    _buf = io.BytesIO()
    _pil_rect.save(_buf, format="PNG")
    _buf.seek(0)

    mo.image(_buf.read())
    return (torch,)


@app.cell
def _(torch):
    """Load MASt3R model."""
    from mast3r.model import AsymmetricMASt3R

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    model.eval()
    print(f"MASt3R loaded on {device}")
    return device, model


@app.cell
def _(frames_bw, frames_fw, torch):
    """Extract DINOv2 global descriptors for visual pair matching."""
    import numpy as _np
    import torchvision.transforms as _T
    from PIL import Image as _PIL
    from tqdm import tqdm as _tqdm

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(_device).eval()
    _transform = _T.Compose([
        _T.Resize(224, interpolation=_T.InterpolationMode.BICUBIC),
        _T.CenterCrop(224),
        _T.ToTensor(),
        _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def _extract(paths, batch_size=32):
        descs = []
        for i in _tqdm(range(0, len(paths), batch_size), desc="DINOv2"):
            batch = torch.stack([
                _transform(_PIL.open(p).convert("RGB"))
                for p in paths[i : i + batch_size]
            ]).to(_device)
            descs.append(_dino(batch))
        return torch.cat(descs).cpu().numpy()

    descs_fw = _extract(frames_fw)
    descs_bw = _extract(frames_bw)
    print(f"DINOv2 descriptors: fw {descs_fw.shape}, bw {descs_bw.shape}")
    return descs_bw, descs_fw


@app.cell
def _(descs_bw, descs_fw, frames_bw, frames_fw, np):
    """Select fw-bw pairs using DINOv2 similarity + time prior.

    Strategy: subsample fw frames evenly, pick top-1 bw match per fw frame.
    This ensures coverage along the entire overlapping transect.
    """
    import matplotlib.pyplot as _plt

    # Cosine similarity
    _norms_fw = np.linalg.norm(descs_fw, axis=1, keepdims=True)
    _norms_bw = np.linalg.norm(descs_bw, axis=1, keepdims=True)
    sim = (descs_fw / _norms_fw) @ (descs_bw / _norms_bw).T  # [N_fw, N_bw]

    N_fw, N_bw = len(frames_fw), len(frames_bw)

    # Subsample fw frames evenly, then pick best bw match within time window
    FW_STEP = 5         # one pair every 5 fw frames → ~230 pairs for 1160 frames
    WINDOW = 100          # search ±100 bw frames around time-prior center
    SIM_THRESH = 0.9

    pairs_idx = []
    for i in range(0, N_fw, FW_STEP):
        j_center = int(round((1 - i / max(N_fw - 1, 1)) * (N_bw - 1)))
        j_lo = max(0, j_center - WINDOW)
        j_hi = min(N_bw, j_center + WINDOW + 1)
        j_best = j_lo + int(np.argmax(sim[i, j_lo:j_hi]))
        if sim[i, j_best] >= SIM_THRESH:
            pairs_idx.append((i, j_best))

    print(f"Selected {len(pairs_idx)} pairs (step={FW_STEP}, window={WINDOW}, sim>={SIM_THRESH})")

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(14, 5))
    _ax1.imshow(sim, aspect="auto", origin="lower", cmap="viridis")
    if pairs_idx:
        _fi_list, _bi_list = zip(*pairs_idx)
        _ax1.scatter(_bi_list, _fi_list, s=2, c="red", alpha=0.7)
    _ax1.set_xlabel("Backward frame index")
    _ax1.set_ylabel("Forward frame index")
    _ax1.set_title(f"DINOv2 cosine similarity (n={len(pairs_idx)} pairs)")
    _plt.colorbar(_ax1.images[0], ax=_ax1)

    _sims = [float(sim[i, j]) for i, j in pairs_idx]
    _ax2.hist(_sims, bins=50)
    _ax2.set_xlabel("Cosine similarity")
    _ax2.set_title("Selected pair similarity distribution")
    _plt.tight_layout()
    _plt.show()
    return (pairs_idx,)


@app.cell
def _(device, frames_bw, frames_fw, model, pairs_idx):
    """Sanity check: run MASt3R on one pair and display matches."""
    import matplotlib.pyplot as _plt
    import numpy as _np
    import torchvision.transforms as _T
    from PIL import Image as _PIL
    from dust3r.inference import inference as dust3r_inference
    from mast3r.colmap.database import get_im_matches
    from mast3r.fast_nn import extract_correspondences_nonsym

    TARGET_SIZE = 512
    CONF_THR = 1.0
    _ImgNorm = _T.Compose([_T.ToTensor(), _T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    def _load_view(path, idx):
        pil = _PIL.open(path).convert("RGB")
        W, H = pil.size
        scale = TARGET_SIZE / max(H, W)
        new_H = round(H * scale / 16) * 16                                           
        new_W = round(W * scale / 16) * 16    
        pil = pil.resize((new_W, new_H), _PIL.LANCZOS)
        return dict(
            img=_ImgNorm(pil)[None],
            true_shape=_np.int32([[new_H, new_W]]),
            idx=idx,
            instance=str(idx),
        )

    for _i in range(0, len(pairs_idx), 10):
        _fi, _bi = pairs_idx[_i]

        _view1 = _load_view(frames_fw[_fi], idx=0)
        _view2 = _load_view(frames_bw[_bi], idx=1)

    

        _result = dust3r_inference([(_view1, _view2)], model, device, batch_size=1, verbose=False)
        _pred1, _pred2 = _result["pred1"], _result["pred2"]

        # get_im_matches returns {(imidx0, imidx1): colmap_matches_array}
        _image_to_colmap = {0: {"colmap_imid": 0}, 1: {"colmap_imid": 1}}
        _im_keypoints = {0: {}, 1: {}}
        _im_matches = get_im_matches(
            _pred1, _pred2, [(_view1, _view2)],
            _image_to_colmap, _im_keypoints,
            conf_thr=CONF_THR, subsample=8, device=device,
        )
        _n_matches = sum(v.shape[0] for v in _im_matches.values())
    
        # if _n_matches < 100 :
        #    continue
        print(f"Test pair fwd[{_fi}] ↔ bwd[{_bi}]: {_n_matches} colmap matches")
        print(f"Keypoints im0: {len(_im_keypoints[0])}, im1: {len(_im_keypoints[1])}")

        # Visualise correspondences
        _descs = [_pred1["desc"][0], _pred2["desc"][0]]
        _confs = [_pred1["desc_conf"][0], _pred2["desc_conf"][0]]
        _corres = extract_correspondences_nonsym(
            _descs[0], _descs[1], _confs[0], _confs[1],
            device=device, subsample=8,
        )
        _conf_mask = _corres[2] >= CONF_THR
        _pts0 = _corres[0][_conf_mask].cpu().numpy()
        _pts1 = _corres[1][_conf_mask].cpu().numpy()

        _img0_arr = _np.array(_PIL.open(frames_fw[_fi]).convert("RGB"))
        _img1_arr = _np.array(_PIL.open(frames_bw[_bi]).convert("RGB"))
        _H0, _W0 = _img0_arr.shape[:2]
        _H1, _W1 = _img1_arr.shape[:2]
        _s0 = TARGET_SIZE / max(_H0, _W0)
        _s1 = TARGET_SIZE / max(_H1, _W1)
        _arr0 = _np.array(_PIL.fromarray(_img0_arr).resize((int(_W0 * _s0), int(_H0 * _s0))))
        _arr1 = _np.array(_PIL.fromarray(_img1_arr).resize((int(_W1 * _s1), int(_H1 * _s1))))

        _canvas_h = max(_arr0.shape[0], _arr1.shape[0])
        _canvas0 = _np.pad(_arr0, ((0, _canvas_h - _arr0.shape[0]), (0, 0), (0, 0)))
        _canvas1 = _np.pad(_arr1, ((0, _canvas_h - _arr1.shape[0]), (0, 0), (0, 0)))
        _canvas = _np.concatenate([_canvas0, _canvas1], axis=1)

        _fig, _ax = _plt.subplots(figsize=(14, 5))
        _ax.imshow(_canvas)
        _cmap = _plt.get_cmap("jet")
        _n_viz = min(10, len(_pts0))
        _idx = sorted(_np.random.choice(len(_pts0), _n_viz, replace=False) )
        for _k, _j in enumerate(_idx):
            _x0, _y0 = _pts0[_j]
            _x1, _y1 = _pts1[_j]
            _ax.plot([_x0, _x1 + _arr0.shape[1]], [_y0, _y1], "-",
                     color=_cmap(_k / max(_n_viz - 1, 1)), lw=0.8, alpha=0.7)
        _ax.set_title(f"MASt3R matches: {len(_pts0)} (fwd[{_fi}] ↔ bwd[{_bi}])")
        _ax.axis("off")
        _plt.tight_layout()
        _plt.show()
    
    return (
        TARGET_SIZE,
        dust3r_inference,
        extract_correspondences_nonsym,
        get_im_matches,
    )


@app.cell
def _(
    TARGET_SIZE,
    device,
    dust3r_inference,
    extract_correspondences_nonsym,
    frames_bw,
    frames_fw,
    get_im_matches,
    model,
    pairs_idx,
):
    """Full MASt3R inference on all selected pairs.

    Stores per-pair:
      - pts3d_1:      [H, W, 3] in cam1 frame  (fw view)
      - pts3d_2_cam2: [H, W, 3] in cam2 frame  (bw view, needed for bw-world projection)
      - conf1, conf2: [H, W] confidence maps
      - corres_pts0:  [K, 2] matched pixel coords in view1
      - corres_pts1:  [K, 2] matched pixel coords in view2
    """
    import numpy as _np
    import torchvision.transforms as _T
    from PIL import Image as _PIL
    from tqdm import tqdm

    _ImgNorm = _T.Compose([_T.ToTensor(), _T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def _load_view(path, idx):
        pil = _PIL.open(path).convert("RGB")
        W, H = pil.size
        scale = TARGET_SIZE / max(H, W)
        new_H = round(H * scale / 16) * 16
        new_W = round(W * scale / 16) * 16
        pil = pil.resize((new_W, new_H), _PIL.LANCZOS)
        return dict(
            img=_ImgNorm(pil)[None],
            true_shape=_np.int32([[new_H, new_W]]),
            idx=idx,
            instance=str(idx),
        )

    # Global index scheme: fwd frames 0..N_fwd-1, bwd frames N_fwd..N_fwd+N_bwd-1
    N_fwd = len(frames_fw)
    _view_cache = {}

    def _get_view(is_bw, local_idx):
        global_idx = (N_fwd + local_idx) if is_bw else local_idx
        if global_idx not in _view_cache:
            path = frames_bw[local_idx] if is_bw else frames_fw[local_idx]
            _view_cache[global_idx] = _load_view(path, global_idx)
        return _view_cache[global_idx]

    _pair_views = [(_get_view(False, fi), _get_view(True, bi)) for fi, bi in pairs_idx]

    # image_to_colmap: global_idx → {"colmap_imid": global_idx}
    image_to_colmap = {gidx: {"colmap_imid": gidx} for gidx in _view_cache}
    im_keypoints = {gidx: {} for gidx in _view_cache}
    all_colmap_matches = {}   # (imidx_lo, imidx_hi) → colmap_matches array

    # Store MASt3R 3D pointmaps for alignment estimation
    # Each entry: (fi, bi, pts3d_in_cam1, conf1) where pts3d are in view1's camera frame
    pair_pointmaps = []

    BATCH_SIZE = 4
    for _batch_start in tqdm(range(0, len(_pair_views), BATCH_SIZE), desc="MASt3R inference"):
        _batch = _pair_views[_batch_start: _batch_start + BATCH_SIZE]
        _result = dust3r_inference(_batch, model, device, batch_size=BATCH_SIZE, verbose=False)
        _pred1, _pred2 = _result["pred1"], _result["pred2"]

        _batch_matches = get_im_matches(
            _pred1, _pred2, _batch,
            image_to_colmap, im_keypoints,
            conf_thr=1.5, subsample=8, device=device,
        )
        for (idx0, idx1), matches_arr in _batch_matches.items():
            key = (min(idx0, idx1), max(idx0, idx1))
            if key in all_colmap_matches:
                all_colmap_matches[key] = _np.concatenate(
                    [all_colmap_matches[key], matches_arr], axis=0
                )
            else:
                all_colmap_matches[key] = matches_arr

        # Collect pointmaps and pixel correspondences for SIM3 alignment
        _b = _pred1["pts3d"].shape[0]
        for _k in range(_b):
            _pair_i = _batch_start + _k
            if _pair_i >= len(pairs_idx):
                break
            _fi, _bi = pairs_idx[_pair_i]
            _pts1      = _pred1["pts3d"][_k].cpu().numpy()   # [H, W, 3] in cam1 frame
            _pts2_cam2 = _pred2["pts3d_in_other_view"][_k].cpu().numpy()   # [H, W, 3] in cam2 frame
            _conf1     = _pred1["conf"][_k].cpu().numpy()    # [H, W]
            _conf2     = _pred2["conf"][_k].cpu().numpy()    # [H, W]

            # Dense pixel correspondences: view1 ↔ view2 (in their respective grids)
            _corres = extract_correspondences_nonsym(
                _pred1["desc"][_k], _pred2["desc"][_k],
                _pred1["desc_conf"][_k], _pred2["desc_conf"][_k],
                device=device, subsample=4,
            )
            # corres = (pts0 [K,2], pts1 [K,2], conf [K]) — pixel coords
            _cmask = _corres[2] >= 1.5
            _corres_pts0 = _corres[0][_cmask].cpu().numpy()  # [K, 2] x,y in view1
            _corres_pts1 = _corres[1][_cmask].cpu().numpy()  # [K, 2] x,y in view2

            pair_pointmaps.append(
                (_fi, _bi, _pts1, _pts2_cam2, _conf1, _conf2, _corres_pts0, _corres_pts1)
            )

    print(f"Processed {len(_pair_views)} pairs")
    print(f"Unique images with keypoints: {len(im_keypoints)}")
    print(f"Unique image pairs with matches: {len(all_colmap_matches)}")
    print(f"Pointmap pairs stored: {len(pair_pointmaps)}")
    return (
        N_fwd,
        all_colmap_matches,
        im_keypoints,
        image_to_colmap,
        pair_pointmaps,
    )


@app.cell
def _(np, pair_pointmaps, poses_bw, poses_fw):
    """Estimate SIM3 alignment T_bw2fw via scene-point correspondences.

    For each pair (fi, bi), MASt3R's pixel matches (u1,v1)↔(u2,v2) give:
      P_fw  = C2W_fw[fi] @ pts3d_1[v1, u1]      — fw world coords
      P_bw  = C2W_bw[bi] @ pts3d_2_cam2[v2, u2]  — bw world coords

    These are true 3D-3D scene correspondences across the two world frames,
    independent of any assumed camera co-location.

    We aggregate correspondences from all pairs, then run Umeyama SIM3 + RANSAC
    to get a robust T_bw2fw = (s, R, t).
    """
    import matplotlib.pyplot as _plt

    def _umeyama(P_src, P_dst):
        """SIM3: P_dst ≈ s*R*P_src + t  (Umeyama 1991)."""
        n = len(P_src)
        mu_s = P_src.mean(0)
        mu_d = P_dst.mean(0)
        Ps = P_src - mu_s
        Pd = P_dst - mu_d
        sigma2 = (Ps ** 2).sum() / n
        H = Ps.T @ Pd / n
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        s = (S * np.diag(D)).sum() / max(sigma2, 1e-12)
        t = mu_d - s * R @ mu_s
        return s, R, t

    # ── Step 1: build 3D-3D correspondences from MASt3R pixel matches ──────
    _all_P_fw = []
    _all_P_bw = []

    for fi, bi, pts3d_1, pts3d_2_cam2, conf1, conf2, corres_pts0, corres_pts1 in pair_pointmaps:
        if len(corres_pts0) < 4:
            continue

        C2W_fw = poses_fw[fi]   # [4, 4]  fw cam → fw world
        C2W_bw = poses_bw[bi]   # [4, 4]  bw cam → bw world
        H1, W1 = pts3d_1.shape[:2]
        H2, W2 = pts3d_2_cam2.shape[:2]

        # Integer pixel indices (clipped to grid bounds)
        u1 = corres_pts0[:, 0].round().astype(int).clip(0, W1 - 1)
        v1 = corres_pts0[:, 1].round().astype(int).clip(0, H1 - 1)
        u2 = corres_pts1[:, 0].round().astype(int).clip(0, W2 - 1)
        v2 = corres_pts1[:, 1].round().astype(int).clip(0, H2 - 1)

        p_cam1 = pts3d_1[v1, u1]           # [K, 3] in cam1 frame
        p_cam2 = pts3d_2_cam2[v2, u2]      # [K, 3] in cam2 frame

        # Keep only finite, high-confidence correspondences
        valid = (
            np.isfinite(p_cam1).all(axis=1)
            & np.isfinite(p_cam2).all(axis=1)
            & (conf1[v1, u1] > 1.5)
            & (conf2[v2, u2] > 1.5)
        )
        p_cam1 = p_cam1[valid]
        p_cam2 = p_cam2[valid]
        if len(p_cam1) < 4:
            continue

        # Project to respective world frames
        P_fw = (C2W_fw[:3, :3] @ p_cam1.T + C2W_fw[:3, 3:4]).T   # [K, 3]
        P_bw = (C2W_bw[:3, :3] @ p_cam2.T + C2W_bw[:3, 3:4]).T   # [K, 3]

        _all_P_fw.append(P_fw)
        _all_P_bw.append(P_bw)

    _all_P_fw = np.concatenate(_all_P_fw, axis=0)   # [N, 3]
    _all_P_bw = np.concatenate(_all_P_bw, axis=0)   # [N, 3]
    print(f"3D-3D correspondences: {len(_all_P_fw):,} across {len(pair_pointmaps)} pairs")

    # ── Step 2: RANSAC SIM3 ─────────────────────────────────────────────────
    N_RANSAC   = 2000
    MIN_SAMPLE = 4
    INLIER_THR = 0.5   # metres — adjust if reconstructions have a different scale

    _rng = np.random.default_rng(42)
    _best_mask = np.zeros(len(_all_P_fw), dtype=bool)

    for _ in range(N_RANSAC):
        _idx = _rng.choice(len(_all_P_fw), MIN_SAMPLE, replace=False)
        _s, _R, _t = _umeyama(_all_P_bw[_idx], _all_P_fw[_idx])
        _res = np.linalg.norm(
            (_s * _R @ _all_P_bw.T + _t[:, None]).T - _all_P_fw, axis=1
        )
        _mask = _res < INLIER_THR
        if _mask.sum() > _best_mask.sum():
            _best_mask = _mask

    # Final fit on all inliers
    _scale, _R, _t = _umeyama(_all_P_bw[_best_mask], _all_P_fw[_best_mask])

    T_bw2fw = np.eye(4, dtype=np.float64)
    T_bw2fw[:3, :3] = _scale * _R
    T_bw2fw[:3, 3] = _t

    # Residuals on inlier set
    _aligned_inliers = (_scale * _R @ _all_P_bw[_best_mask].T + _t[:, None]).T
    _res_in = np.linalg.norm(_aligned_inliers - _all_P_fw[_best_mask], axis=1)
    print(f"RANSAC: {_best_mask.sum()} / {len(_all_P_fw)} inliers  "
          f"(thr={INLIER_THR}m)")
    print(f"SIM3 scale={_scale:.4f}")
    print(f"  Inlier residuals: median={np.median(_res_in):.4f}m, "
          f"mean={np.mean(_res_in):.4f}m, max={np.max(_res_in):.4f}m")

    # ── Step 3: transform all bw poses into fw world frame ──────────────────
    poses_bw_aligned = np.array([T_bw2fw @ p for p in poses_bw])

    # ── Visualisation ────────────────────────────────────────────────────────
    _fig = _plt.figure(figsize=(14, 5))
    _ax1 = _fig.add_subplot(131, projection="3d")
    _ax1.plot(poses_fw[:, 0, 3], poses_fw[:, 1, 3], poses_fw[:, 2, 3],
              "b-", lw=1, label="fw")
    _ax1.plot(poses_bw[:, 0, 3], poses_bw[:, 1, 3], poses_bw[:, 2, 3],
              "r-", lw=1, label="bw (original)")
    _ax1.set_title("Before alignment")
    _ax1.legend(fontsize=7)

    _ax2 = _fig.add_subplot(132, projection="3d")
    _ax2.plot(poses_fw[:, 0, 3], poses_fw[:, 1, 3], poses_fw[:, 2, 3],
              "b-", lw=1, label="fw")
    _ax2.plot(poses_bw_aligned[:, 0, 3], poses_bw_aligned[:, 1, 3], poses_bw_aligned[:, 2, 3],
              "r-", lw=1, label="bw (aligned)")
    _inlier_P_fw = _all_P_fw[_best_mask][::max(1, _best_mask.sum() // 500)]
    _ax2.scatter(_inlier_P_fw[:, 0], _inlier_P_fw[:, 1], _inlier_P_fw[:, 2],
                 c="green", s=3, alpha=0.4, label="inlier anchors")
    _ax2.set_title("After SIM3 alignment")
    _ax2.legend(fontsize=7)

    _ax3 = _fig.add_subplot(133)
    _ax3.hist(_res_in, bins=50)
    _ax3.axvline(np.median(_res_in), color="r", linestyle="--", label="median")
    _ax3.set_xlabel("Inlier residual (m)")
    _ax3.set_title("Residual distribution")
    _ax3.legend(fontsize=7)

    _plt.tight_layout()
    _plt.show()
    return (poses_bw_aligned,)


@app.cell
def _(frames_bw, frames_fw, output_fw_dir, pair_pointmaps, poses_fw):
    def _():
        def _():
            """Estimate depth scale, normalize per-pair depth, build merged point cloud."""
            import open3d as _o3d
            import numpy as _np
            from PIL import Image as _PIL
            from scipy.optimize import minimize_scalar
            from scipy.spatial import cKDTree
            from tqdm import tqdm as _tqdm

            TARGET_SIZE = 512
            CONF_DIRECT = 2.0    # confidence threshold for direct view (pts3d_1)
            CONF_CROSS = 3.0     # stricter threshold for cross-view (pts3d_2)
            SUBSAMPLE = 8

            # ── Step 1: estimate global depth scale ──────────────────────────
            _sorted = sorted(enumerate(pair_pointmaps), key=lambda x: x[1][0])

            def _overlap_residual(log_s):
                s = 10.0 ** log_s
                residuals = []
                for idx in range(len(_sorted) - 1):
                    _, (fi_a, _, pts_a, _, conf_a, *_) = _sorted[idx]
                    _, (fi_b, _, pts_b, _, conf_b, *_) = _sorted[idx + 1]
                    if fi_b - fi_a > 10:
                        continue
                    mask_a = (conf_a > 2.0) & _np.isfinite(pts_a).all(axis=-1)
                    mask_b = (conf_b > 2.0) & _np.isfinite(pts_b).all(axis=-1)
                    pa = pts_a[mask_a][::SUBSAMPLE]
                    pb = pts_b[mask_b][::SUBSAMPLE]
                    if len(pa) < 20 or len(pb) < 20:
                        continue
                    C2W_a, C2W_b = poses_fw[fi_a], poses_fw[fi_b]
                    wa = (C2W_a[:3, :3] @ (s * pa).T + C2W_a[:3, 3:4]).T
                    wb = (C2W_b[:3, :3] @ (s * pb).T + C2W_b[:3, 3:4]).T
                    dists, _ = cKDTree(wb).query(wa, k=1)
                    residuals.append(_np.median(dists))
                return _np.median(residuals) if residuals else 1e9

            res = minimize_scalar(_overlap_residual, bounds=(-4.0, 1.0), method="bounded",
                                  options={"xatol": 0.01})
            depth_scale = 10.0 ** res.x
            print(f"Global depth scale: {depth_scale:.6f}  (residual: {res.fun:.6f})")

            # ── Step 2: compute per-pair median depth for normalization ──────
            # Each MASt3R pair predicts depth independently. The median z-depth
            # fluctuates between pairs, creating "waves" in the merged cloud.
            # Normalize each pair's depth so its median matches the global median.
            _median_depths = []
            for fi, bi, pts3d_1, _, conf1, *_ in pair_pointmaps:
                mask = (conf1 > CONF_DIRECT) & _np.isfinite(pts3d_1).all(axis=-1)
                if mask.sum() < 10:
                    _median_depths.append(_np.nan)
                    continue
                # Use z-depth (distance along optical axis) not Euclidean norm
                _median_depths.append(_np.median(pts3d_1[mask, 2]))
            _median_depths = _np.array(_median_depths)
            _D_global = _np.nanmedian(_median_depths)
            print(f"Per-pair median depth: global={_D_global:.3f}m, "
                  f"std={_np.nanstd(_median_depths):.3f}m, "
                  f"range=[{_np.nanmin(_median_depths):.3f}, {_np.nanmax(_median_depths):.3f}]")

            # ── Step 3: build scaled + normalized point cloud ────────────────
            VOXEL_SIZE = depth_scale * 0.02  # ~2 cm in scene scale

            all_pts = []
            all_rgb = []

            for k, (fi, bi, pts3d_1, pts3d_2_cam2, conf1, conf2, *_) in enumerate(
                _tqdm(pair_pointmaps, desc="Building point cloud")
            ):
                C2W = poses_fw[fi]

                # Per-pair depth normalization factor
                if _np.isnan(_median_depths[k]) or _median_depths[k] < 1e-3:
                    pair_norm = 1.0
                else:
                    pair_norm = _D_global / _median_depths[k]

                for pts3d, conf, frame_path, conf_thr in [
                    (pts3d_1, conf1, frames_fw[fi], CONF_DIRECT),
                    (pts3d_2_cam2, conf2, frames_bw[bi], CONF_CROSS),
                ]:
                    H, W = pts3d.shape[:2]
                    valid = (conf > conf_thr) & _np.isfinite(pts3d).all(axis=-1)
                    pts_cam = pts3d[valid]
                    if len(pts_cam) == 0:
                        continue

                    # Apply per-pair normalization then global scale
                    pts_scaled = depth_scale * pair_norm * pts_cam
                    pts_world = (C2W[:3, :3] @ pts_scaled.T + C2W[:3, 3:4]).T

                    pil = _PIL.open(frame_path).convert("RGB")
                    Wo, Ho = pil.size
                    sc = TARGET_SIZE / max(Ho, Wo)
                    new_H = round(Ho * sc / 16) * 16
                    new_W = round(Wo * sc / 16) * 16
                    pil = pil.resize((new_W, new_H), _PIL.LANCZOS)
                    rgb = _np.array(pil).reshape(-1, 3)[valid.ravel()].astype(_np.float64) / 255.0

                    all_pts.append(pts_world)
                    all_rgb.append(rgb)

            all_pts = _np.concatenate(all_pts, axis=0)
            all_rgb = _np.concatenate(all_rgb, axis=0)
            print(f"Raw points: {len(all_pts):,}")

            pcd = _o3d.geometry.PointCloud()
            pcd.points = _o3d.utility.Vector3dVector(all_pts.astype(_np.float64))
            pcd.colors = _o3d.utility.Vector3dVector(all_rgb)

            pcd = pcd.voxel_down_sample(VOXEL_SIZE)
            print(f"After voxel downsample ({VOXEL_SIZE:.5f}): {len(pcd.points):,}")

            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            print(f"After outlier removal: {len(pcd.points):,}")

            out_path = output_fw_dir.parent / "mast3r_merged.ply"
            _o3d.io.write_point_cloud(str(out_path), pcd)
            return print(f"Saved to {out_path}")
        return _()


    _()
    return


@app.cell(disabled=True)
def _(
    N_fwd,
    all_colmap_matches,
    frames_bw,
    frames_fw,
    im_keypoints,
    image_to_colmap,
    output_fw_dir,
):
    """Build COLMAP database with cameras, images, keypoints, and matches."""
    import sqlite3
    import numpy as _np
    from PIL import Image as _PIL

    out_dir = output_fw_dir.parent / "mast3r_colmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "colmap.db"

    if db_path.exists():
        db_path.unlink()

    def _pair_id(id1, id2):
        lo, hi = (id1, id2) if id1 < id2 else (id2, id1)
        return lo * 2147483647 + hi

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
            prior_tx REAL, prior_ty REAL, prior_tz REAL);
        CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB);
        CREATE TABLE IF NOT EXISTS matches (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB);
        CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            config INTEGER NOT NULL,
            F BLOB, E BLOB, H BLOB,
            qvec BLOB, tvec BLOB);
    """)
    con.commit()

    # Camera: PINHOLE (model id=1) using mean forward intrinsics fx,fy,cx,cy
    _intr_mean = _np.load(output_fw_dir / "intrinsics.npy").mean(axis=0)
    _fx, _fy, _cx, _cy = float(_intr_mean[0]), float(_intr_mean[1]), float(_intr_mean[2]), float(_intr_mean[3])
    _pil_sample = _PIL.open(frames_fw[0])
    _W_orig, _H_orig = _pil_sample.size
    _cam_params = _np.array([_fx, _fy, _cx, _cy], dtype=_np.float64)
    cur.execute(
        "INSERT INTO cameras (model, width, height, params, prior_focal_length) VALUES (?,?,?,?,?)",
        (1, _W_orig, _H_orig, _cam_params.tobytes(), 1),
    )
    con.commit()
    camera_id = cur.lastrowid

    # Images
    colmap_image_ids = {}   # global_idx → colmap image_id (1-based)
    for _gidx in sorted(image_to_colmap.keys()):
        _is_bw = _gidx >= N_fwd
        _li = _gidx - N_fwd if _is_bw else _gidx
        _frame_path = frames_bw[_li] if _is_bw else frames_fw[_li]
        _name = ("bwd_" if _is_bw else "fwd_") + _frame_path.name
        cur.execute("INSERT INTO images (name, camera_id) VALUES (?,?)", (_name, camera_id))
        con.commit()
        colmap_image_ids[_gidx] = cur.lastrowid

    # Keypoints — must be ordered by colmap_kp_index (values in _kp_dict),
    # NOT sorted by ravel_id (keys), because match indices reference colmap_kp_index.
    # Also: unravel using the exact MASt3R grid (round-to-16), then scale to original res.
    _TARGET_SIZE = 512
    for _gidx, _kp_dict in im_keypoints.items():
        if not _kp_dict:
            continue
        _is_bw = _gidx >= N_fwd
        _li = _gidx - N_fwd if _is_bw else _gidx
        _frame_path = frames_bw[_li] if _is_bw else frames_fw[_li]
        _pil_kp = _PIL.open(_frame_path)
        _Wkp, _Hkp = _pil_kp.size

        # MASt3R grid dimensions (must match _load_view: round to multiple of 16)
        _scale = _TARGET_SIZE / max(_Hkp, _Wkp)
        _new_H = round(_Hkp * _scale / 16) * 16
        _new_W = round(_Wkp * _scale / 16) * 16

        # Scale factors: MASt3R coords → original image coords
        _sx = _Wkp / _new_W
        _sy = _Hkp / _new_H

        # Build keypoint array ordered by colmap_kp_index (the dict values)
        _n_kp = max(_kp_dict.values()) + 1
        _kp_xy = _np.zeros((_n_kp, 2), dtype=_np.float32)
        for _ravel_id, _colmap_idx in _kp_dict.items():
            _y, _x = divmod(int(_ravel_id), _new_W)
            _kp_xy[_colmap_idx] = [_x * _sx, _y * _sy]

        _cimid = colmap_image_ids[_gidx]
        cur.execute(
            "INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?,?,?,?)",
            (_cimid, _kp_xy.shape[0], 2, _kp_xy.tobytes()),
        )
    con.commit()

    # Matches
    for (_gidx0, _gidx1), _matches_arr in all_colmap_matches.items():
        _cimid0 = colmap_image_ids[_gidx0]
        _cimid1 = colmap_image_ids[_gidx1]
        _pid = _pair_id(_cimid0, _cimid1)
        _m = _matches_arr.astype(_np.uint32)
        cur.execute(
            "INSERT OR REPLACE INTO matches (pair_id, rows, cols, data) VALUES (?,?,?,?)",
            (_pid, _m.shape[0], 2, _m.tobytes()),
        )
    con.commit()
    con.close()

    # Pairs text file for geometric verification
    pairs_path = out_dir / "pairs.txt"
    with open(pairs_path, "w") as _f:
        for (_gidx0, _gidx1) in all_colmap_matches:
            _is_bw0 = _gidx0 >= N_fwd
            _is_bw1 = _gidx1 >= N_fwd
            _li0 = _gidx0 - N_fwd if _is_bw0 else _gidx0
            _li1 = _gidx1 - N_fwd if _is_bw1 else _gidx1
            _n0 = ("bwd_" if _is_bw0 else "fwd_") + (frames_bw[_li0] if _is_bw0 else frames_fw[_li0]).name
            _n1 = ("bwd_" if _is_bw1 else "fwd_") + (frames_bw[_li1] if _is_bw1 else frames_fw[_li1]).name
            _f.write(f"{_n0} {_n1}\n")

    print(f"COLMAP database: {db_path}")
    print(f"Pairs file:      {pairs_path}")
    print(f"  Images:        {len(colmap_image_ids)}")
    print(f"  Match pairs:   {len(all_colmap_matches)}")
    return camera_id, colmap_image_ids, db_path, out_dir, pairs_path


@app.cell(disabled=True)
def _(
    N_fwd,
    camera_id,
    colmap_image_ids,
    db_path,
    frames_bw,
    frames_fw,
    image_to_colmap,
    out_dir,
    output_fw_dir,
    pairs_path,
    poses_bw_aligned,
    poses_fw,
):
    """COLMAP geometric verification and triangulation with aligned poses."""
    import os
    import numpy as _np
    import pycolmap
    from PIL import Image as _PIL
    from scipy.spatial.transform import Rotation as _Rot

    pycolmap.verify_matches(
        str(db_path),
        str(pairs_path),
        options=pycolmap.TwoViewGeometryOptions(),
    )
    print("Geometric verification complete.")

    reconstruction = pycolmap.Reconstruction()

    # Camera
    _intr_fw = _np.load(output_fw_dir / "intrinsics.npy").mean(axis=0)
    _fx, _fy, _cx, _cy = float(_intr_fw[0]), float(_intr_fw[1]), float(_intr_fw[2]), float(_intr_fw[3])
    _W, _H = _PIL.open(frames_fw[0]).size
    _cam = pycolmap.Camera(
        model="PINHOLE",
        width=_W,
        height=_H,
        params=[_fx, _fy, _cx, _cy],
        camera_id=camera_id,
    )
    reconstruction.add_camera(_cam)

    # Images with prior poses (C2W → W2C) — bw poses are now in fw world frame
    for _gidx in sorted(image_to_colmap.keys()):
        _is_bw = _gidx >= N_fwd
        _li = _gidx - N_fwd if _is_bw else _gidx
        _poses = poses_bw_aligned if _is_bw else poses_fw
        _c2w = _poses[_li]
        _w2c = _np.linalg.inv(_c2w)
        _R_mat = _w2c[:3, :3]
        _t_vec = _w2c[:3, 3]
        _qxyz = _Rot.from_matrix(_R_mat).as_quat()   # [x, y, z, w]
        _qvec = _np.array([_qxyz[3], _qxyz[0], _qxyz[1], _qxyz[2]])  # COLMAP [w,x,y,z]

        _frame_path = frames_bw[_li] if _is_bw else frames_fw[_li]
        _name = ("bwd_" if _is_bw else "fwd_") + _frame_path.name
        _cimid = colmap_image_ids[_gidx]

        _img = pycolmap.Image(
            id=_cimid,
            name=_name,
            camera_id=camera_id,
            cam_from_world=pycolmap.Rigid3d(
                rotation=pycolmap.Rotation3d(_qvec),
                translation=_t_vec,
            ),
        )
        reconstruction.add_image(_img)

    # Symlink all frames into a unified image directory
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for _gidx in image_to_colmap:
        _is_bw = _gidx >= N_fwd
        _li = _gidx - N_fwd if _is_bw else _gidx
        _src = frames_bw[_li] if _is_bw else frames_fw[_li]
        _prefix = "bwd_" if _is_bw else "fwd_"
        _dst = img_dir / (_prefix + _src.name)
        if not _dst.exists():
            os.symlink(_src, _dst)

    sparse_dir = out_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    pycolmap.triangulate_points(
        reconstruction=reconstruction,
        database_path=str(db_path),
        image_path=str(img_dir),
        output_path=str(sparse_dir),
    )

    print(f"Triangulation complete.")
    print(f"  3D points:          {reconstruction.num_points3D()}")
    print(f"  Registered images:  {reconstruction.num_reg_images()}")
    reconstruction.write(str(sparse_dir))
    print(f"Sparse model written to: {sparse_dir}")
    return (reconstruction,)


@app.cell(disabled=True)
def _(N_fwd, poses_bw_aligned, poses_fw, reconstruction):
    """Visualise aligned trajectories and triangulated 3D points."""
    import matplotlib.pyplot as _plt
    import numpy as _np
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    fig = _plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(poses_fw[:, 0, 3], poses_fw[:, 1, 3], poses_fw[:, 2, 3],
            "b-o", markersize=2, linewidth=1, label="Forward trajectory")
    ax.plot(poses_bw_aligned[:, 0, 3], poses_bw_aligned[:, 1, 3], poses_bw_aligned[:, 2, 3],
            "r-o", markersize=2, linewidth=1, label="Backward trajectory (aligned)")

    _pts_xyz = []
    _colors = []
    for _pt3d in reconstruction.points3D.values():
        _pts_xyz.append(_pt3d.xyz)
        _n_fwd_obs = sum(1 for obs in _pt3d.track.elements if obs.image_id - 1 < N_fwd)
        _colors.append("blue" if _n_fwd_obs >= len(_pt3d.track.elements) - _n_fwd_obs else "red")

    if _pts_xyz:
        _pts_arr = _np.array(_pts_xyz)
        ax.scatter(_pts_arr[:, 0], _pts_arr[:, 1], _pts_arr[:, 2],
                   c=_colors, s=0.5, alpha=0.3)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("MASt3R COLMAP reconstruction: fwd (blue) + bwd (red)")
    ax.legend()
    _plt.tight_layout()
    _plt.show()
    return


if __name__ == "__main__":
    app.run()
