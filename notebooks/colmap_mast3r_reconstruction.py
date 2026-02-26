import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


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
    """Configure input/output paths.

    Change these to point to your forward and backward reconstruction outputs.
    """
    output_fw_dir = Path(
        "/home/hugues/Documents/mee-deepreefmap/output/israel-eilat/2024_Sunrise_fw_leg"
    )
    output_bw_dir = Path(
        "/home/hugues/Documents/mee-deepreefmap/output/israel-eilat/2024_Sunrise_bw_leg"
    )
    colmap_workdir = output_fw_dir.parent / "colmap_mast3r"
    colmap_workdir.mkdir(parents=True, exist_ok=True)
    print(f"COLMAP workdir: {colmap_workdir}")
    return colmap_workdir, output_bw_dir, output_fw_dir


@app.cell
def _(np, output_bw_dir, output_fw_dir):
    """Load reconstruction outputs.

    poses.npy         → [N, 4, 4] C2W poses
    intrinsics.npy    → [N, 6] EUCM intrinsics (fx, fy, cx, cy, alpha, beta)
    frames/*.jpg      → N frame images
    """
    poses_fw = np.load(output_fw_dir / "poses.npy")
    intrinsics_fw = np.load(output_fw_dir / "intrinsics.npy")
    frames_fw = sorted(output_fw_dir.glob("frames/*.jpg"), key=lambda p: int(p.stem))

    poses_bw = np.load(output_bw_dir / "poses.npy")
    intrinsics_bw = np.load(output_bw_dir / "intrinsics.npy")
    frames_bw = sorted(output_bw_dir.glob("frames/*.jpg"), key=lambda p: int(p.stem))

    N_fw = len(frames_fw)
    N_bw = len(frames_bw)
    print(f"Forward:  {N_fw} frames, poses {poses_fw.shape}")
    print(f"Backward: {N_bw} frames, poses {poses_bw.shape}")
    return N_bw, N_fw, frames_bw, frames_fw, intrinsics_fw, poses_bw, poses_fw


@app.cell
def _(colmap_workdir, frames_bw, frames_fw, intrinsics_fw, np):
    """Undistort all frames using EUCM → pinhole rectification.

    GoPro Hero 10 uses a fisheye lens modeled by EUCM (Enhanced Unified Camera
    Model) with params [fx, fy, cx, cy, alpha, beta]. COLMAP needs undistorted
    images for PINHOLE camera model to work correctly.

    The rectified images use [fx, fy, cx, cy] as pinhole intrinsics.
    """
    import torch as _torch
    from PIL import Image as _PIL
    from sfm.inverse_warp import rectify_eucm
    from tqdm import tqdm as _tqdm

    rect_dir = colmap_workdir / "images_rect"
    rect_dir.mkdir(exist_ok=True)

    # Use mean intrinsics for rectification (they're nearly identical per frame)
    _intr_mean = np.mean(intrinsics_fw, axis=0).astype(np.float32)  # [fx,fy,cx,cy,alpha,beta]
    _intr_t = _torch.from_numpy(_intr_mean)

    def _rectify_and_save(src_path, dst_path):
        if dst_path.exists():
            return
        pil = _PIL.open(src_path).convert("RGB")
        arr = np.array(pil).astype(np.float32) / 255.0
        H, W = arr.shape[:2]
        img_t = _torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        mask_t = _torch.ones(1, 1, H, W)
        depth_t = _torch.ones(1, 1, H, W)
        rect_img, _, _ = rectify_eucm(img_t, mask_t, depth_t, _intr_t)
        rect_uint8 = (rect_img.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        _PIL.fromarray(rect_uint8).save(dst_path, quality=95)

    # Rectify all frames (indexed by position so rect_frames_fw[i] == rectified frames_fw[i])
    rect_frames_fw = []
    for p in _tqdm(frames_fw, desc="Rectifying fw"):
        dst = rect_dir / f"fwd_{int(p.stem):06d}.jpg"
        _rectify_and_save(p, dst)
        rect_frames_fw.append(dst)

    rect_frames_bw = []
    for p in _tqdm(frames_bw, desc="Rectifying bw"):
        dst = rect_dir / f"bwd_{int(p.stem):06d}.jpg"
        _rectify_and_save(p, dst)
        rect_frames_bw.append(dst)

    # Pinhole intrinsics after rectification (just fx, fy, cx, cy — distortion removed)
    pinhole_intr = _intr_mean[:4]
    print(f"Rectified {len(rect_frames_fw)} fw + {len(rect_frames_bw)} bw frames → {rect_dir}")
    print(f"Pinhole intrinsics: fx={pinhole_intr[0]:.1f} fy={pinhole_intr[1]:.1f} "
          f"cx={pinhole_intr[2]:.1f} cy={pinhole_intr[3]:.1f}")
    return pinhole_intr, rect_dir, rect_frames_bw, rect_frames_fw


@app.cell
def _():
    """Load MASt3R model."""
    import torch
    from mast3r.model import AsymmetricMASt3R

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    model.eval()
    print(f"MASt3R loaded on {device}")
    return device, model, torch


@app.cell
def _(rect_frames_bw, rect_frames_fw, torch):
    """Extract DINOv2 global descriptors for cross-view pair selection."""
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

    descs_fw = _extract(rect_frames_fw)
    descs_bw = _extract(rect_frames_bw)
    print(f"DINOv2 descriptors: fw {descs_fw.shape}, bw {descs_bw.shape}")
    return descs_bw, descs_fw


@app.cell
def _(N_bw, N_fw, descs_bw, descs_fw, np):
    """Select fw↔bw pairs using DINOv2 similarity + temporal prior.

    Also select within-pass sequential pairs for COLMAP's sequential
    connectivity. The combined pair list gives COLMAP both:
      - Sequential edges (odometry-like) within each pass
      - Cross-pass edges (loop closures) between fw and bw
    """
    import matplotlib.pyplot as _plt

    # ── Cross-pass pairs (fw ↔ bw) ──────────────────────────────────────
    _norms_fw = np.linalg.norm(descs_fw, axis=1, keepdims=True)
    _norms_bw = np.linalg.norm(descs_bw, axis=1, keepdims=True)
    sim = (descs_fw / _norms_fw) @ (descs_bw / _norms_bw).T

    FW_STEP = 3          # one cross pair every 3 fw frames (denser sampling)
    WINDOW = 150         # search ±150 bw frames around time-prior center
    SIM_THRESH = 0.85    # lower threshold to get more cross-pass pairs

    cross_pairs_local = []   # (fw_local_idx, bw_local_idx)
    for i in range(0, N_fw, FW_STEP):
        j_center = int(round((1 - i / max(N_fw - 1, 1)) * (N_bw - 1)))
        j_lo = max(0, j_center - WINDOW)
        j_hi = min(N_bw, j_center + WINDOW + 1)
        j_best = j_lo + int(np.argmax(sim[i, j_lo:j_hi]))
        if sim[i, j_best] >= SIM_THRESH:
            cross_pairs_local.append((i, j_best))

    # ── Sequential pairs (logwin: connect at distances 5, 10, 20, 40) ──
    # Close pairs give robust matches; distant pairs give parallax for triangulation.
    # COLMAP needs wide-baseline pairs to bootstrap initialization.
    LOGWIN_OFFSETS = [5, 10, 20, 40]

    seq_fw_local = set()
    for i in range(0, N_fw, 5):
        for off in LOGWIN_OFFSETS:
            j = min(i + off, N_fw - 1)
            if i != j:
                seq_fw_local.add((min(i, j), max(i, j)))
    seq_fw_local = sorted(seq_fw_local)

    seq_bw_local = set()
    for i in range(0, N_bw, 5):
        for off in LOGWIN_OFFSETS:
            j = min(i + off, N_bw - 1)
            if i != j:
                seq_bw_local.add((min(i, j), max(i, j)))
    seq_bw_local = sorted(seq_bw_local)

    print(f"Cross pairs (fw↔bw): {len(cross_pairs_local)}")
    print(f"Sequential fw pairs: {len(seq_fw_local)} (logwin offsets {LOGWIN_OFFSETS})")
    print(f"Sequential bw pairs: {len(seq_bw_local)} (logwin offsets {LOGWIN_OFFSETS})")

    # ── Visualisation ───────────────────────────────────────────────────
    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(14, 5))
    _ax1.imshow(sim, aspect="auto", origin="lower", cmap="viridis")
    if cross_pairs_local:
        _fi_list, _bi_list = zip(*cross_pairs_local)
        _ax1.scatter(_bi_list, _fi_list, s=2, c="red", alpha=0.7)
    _ax1.set_xlabel("Backward frame index")
    _ax1.set_ylabel("Forward frame index")
    _ax1.set_title(f"DINOv2 cosine similarity (n={len(cross_pairs_local)} cross pairs)")
    _plt.colorbar(_ax1.images[0], ax=_ax1)

    _sims = [float(sim[i, j]) for i, j in cross_pairs_local]
    _ax2.hist(_sims, bins=50)
    _ax2.set_xlabel("Cosine similarity")
    _ax2.set_title("Cross-pair similarity distribution")
    _plt.tight_layout()
    _plt.show()
    return cross_pairs_local, seq_bw_local, seq_fw_local


@app.cell
def _(
    N_fw,
    colmap_workdir,
    cross_pairs_local,
    np,
    pinhole_intr,
    poses_bw,
    poses_fw,
    rect_dir,
    rect_frames_fw,
    seq_bw_local,
    seq_fw_local,
):
    """Create COLMAP database with cameras + images.

    Global index scheme:
      fwd frame i  → global index i        → name "fwd_{i:06d}.jpg"
      bwd frame j  → global index N_fw + j  → name "bwd_{j:06d}.jpg"

    Images are read from rect_dir (undistorted EUCM → pinhole).
    """
    import sqlite3
    from PIL import Image as _PIL
    from scipy.spatial.transform import Rotation as _Rot

    # rect_dir already contains all rectified images; use it directly as img_dir
    img_dir = rect_dir

    # Collect all unique frame indices that participate in any pair
    used_fw_indices = set()
    used_bw_indices = set()

    for fi, bi in cross_pairs_local:
        used_fw_indices.add(fi)
        used_bw_indices.add(bi)
    for fi, fj in seq_fw_local:
        used_fw_indices.add(fi)
        used_fw_indices.add(fj)
    for bi, bj in seq_bw_local:
        used_bw_indices.add(bi)
        used_bw_indices.add(bj)

    used_fw_indices = sorted(used_fw_indices)
    used_bw_indices = sorted(used_bw_indices)
    print(f"Unique frames: {len(used_fw_indices)} fw + {len(used_bw_indices)} bw")

    # ── Build COLMAP database ────────────────────────────────────────────
    db_path = colmap_workdir / "colmap.db"
    if db_path.exists():
        db_path.unlink()

    _con = sqlite3.connect(str(db_path))
    cur = _con.cursor()
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
    _con.commit()

    # Camera: PINHOLE using rectified intrinsics (fx, fy, cx, cy)
    _pil_sample = _PIL.open(rect_frames_fw[0])
    W_orig, H_orig = _pil_sample.size
    _cam_params = np.array([pinhole_intr[0], pinhole_intr[1], pinhole_intr[2], pinhole_intr[3]],
                           dtype=np.float64)
    cur.execute(
        "INSERT INTO cameras (model, width, height, params, prior_focal_length) VALUES (?,?,?,?,?)",
        (1, W_orig, H_orig, _cam_params.tobytes(), 1),
    )
    _con.commit()
    camera_id = cur.lastrowid
    print(f"Camera {camera_id}: PINHOLE {W_orig}x{H_orig} "
          f"fx={pinhole_intr[0]:.1f} fy={pinhole_intr[1]:.1f}")

    # Images with prior poses (C2W → W2C for COLMAP convention)
    # For now we register poses relative to fw world frame.
    # bw poses are in their own frame — we let COLMAP figure out the alignment.
    global_to_colmap_imid = {}

    def _add_image(name, global_idx, c2w):
        w2c = np.linalg.inv(c2w)
        R_mat = w2c[:3, :3]
        t_vec = w2c[:3, 3]
        qxyz = _Rot.from_matrix(R_mat).as_quat()   # [x, y, z, w]
        qvec = [float(qxyz[3]), float(qxyz[0]), float(qxyz[1]), float(qxyz[2])]  # COLMAP [w,x,y,z]
        cur.execute(
            "INSERT INTO images (name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, "
            "prior_tx, prior_ty, prior_tz) VALUES (?,?,?,?,?,?,?,?,?)",
            (name, camera_id, *qvec, *t_vec.tolist()),
        )
        _con.commit()
        global_to_colmap_imid[global_idx] = cur.lastrowid

    for fi in used_fw_indices:
        _add_image(f"fwd_{fi:06d}.jpg", fi, poses_fw[fi])
    for bi in used_bw_indices:
        _add_image(f"bwd_{bi:06d}.jpg", N_fw + bi, poses_bw[bi])

    print(f"Registered {len(global_to_colmap_imid)} images in COLMAP DB")
    _con.close()

    # ── Build pair list (global indices) for MASt3R matching ─────────────
    # Each pair: (global_idx_0, global_idx_1)
    all_pairs_global = []

    # Cross-pass pairs
    for fi, bi in cross_pairs_local:
        all_pairs_global.append((fi, N_fw + bi))
    # Sequential fw pairs
    for fi, fj in seq_fw_local:
        if fi != fj:
            all_pairs_global.append((fi, fj))
    # Sequential bw pairs
    for bi, bj in seq_bw_local:
        if bi != bj:
            all_pairs_global.append((N_fw + bi, N_fw + bj))

    # Deduplicate
    all_pairs_global = list(set(
        (min(a, b), max(a, b)) for a, b in all_pairs_global
    ))
    print(f"Total unique pairs for MASt3R matching: {len(all_pairs_global)}")
    return (
        H_orig,
        W_orig,
        all_pairs_global,
        db_path,
        global_to_colmap_imid,
        img_dir,
        sqlite3,
    )


@app.cell
def _(
    H_orig,
    N_fw,
    W_orig,
    all_pairs_global,
    db_path,
    device,
    global_to_colmap_imid,
    model,
    np,
    rect_frames_bw,
    rect_frames_fw,
    sqlite3,
):
    """Run MASt3R inference on all pairs and export matches to COLMAP DB.

    Uses the official MASt3R colmap integration (get_im_matches + export_matches)
    but adapted to work without kapture, directly from our frame lists.
    Images are rectified (undistorted) so PINHOLE model is correct.
    """
    import torchvision.transforms as _T
    from PIL import Image as _PIL
    from tqdm import tqdm as _tqdm

    from dust3r.inference import inference as dust3r_inference
    from mast3r.fast_nn import extract_correspondences_nonsym

    TARGET_SIZE = 512
    CONF_THR = 1.5
    BATCH_SIZE = 4

    _ImgNorm = _T.Compose([_T.ToTensor(), _T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def _load_view(global_idx):
        is_bw = global_idx >= N_fw
        local_idx = global_idx - N_fw if is_bw else global_idx
        path = rect_frames_bw[local_idx] if is_bw else rect_frames_fw[local_idx]
        pil = _PIL.open(path).convert("RGB")
        W, H = pil.size
        scale = TARGET_SIZE / max(H, W)
        new_H = round(H * scale / 16) * 16
        new_W = round(W * scale / 16) * 16
        pil = pil.resize((new_W, new_H), _PIL.Resampling.LANCZOS)
        return dict(
            img=_ImgNorm(pil)[None],
            true_shape=np.int32([[new_H, new_W]]),
            idx=global_idx,
            instance=str(global_idx),
        ), new_H, new_W

    # ── Run MASt3R on all pairs ──────────────────────────────────────────
    # We collect matches in the same format as mast3r.colmap.database:
    #   im_keypoints: {global_idx: {ravel_id: count}}
    #   im_matches:   {(imidx_lo, imidx_hi): colmap_matches_array}
    im_keypoints = {gidx: {} for gidx in global_to_colmap_imid}
    im_matches = {}
    view_cache = {}
    view_shape_cache = {}

    def _get_view(global_idx):
        if global_idx not in view_cache:
            view, h, w = _load_view(global_idx)
            view_cache[global_idx] = view
            view_shape_cache[global_idx] = (h, w)
        return view_cache[global_idx]

    for batch_start in _tqdm(range(0, len(all_pairs_global), BATCH_SIZE), desc="MASt3R inference"):
        batch_pairs = all_pairs_global[batch_start : batch_start + BATCH_SIZE]
        batch_views = [(_get_view(g0), _get_view(g1)) for g0, g1 in batch_pairs]

        result = dust3r_inference(batch_views, model, device, batch_size=BATCH_SIZE, verbose=False)
        pred1, pred2 = result["pred1"], result["pred2"]

        for k in range(pred1["pts3d"].shape[0]):
            pair_idx = batch_start + k
            if pair_idx >= len(all_pairs_global):
                break
            gidx0, gidx1 = all_pairs_global[pair_idx]

            # Extract correspondences using MASt3R descriptors
            descs = [pred1["desc"][k], pred2["desc"][k]]
            confs = [pred1["desc_conf"][k], pred2["desc_conf"][k]]
            corres = extract_correspondences_nonsym(
                descs[0], descs[1], confs[0], confs[1],
                device=device, subsample=8,
            )
            conf_mask = corres[2] >= CONF_THR
            matches_im0 = corres[0][conf_mask].cpu().numpy()  # [K, 2] x,y
            matches_im1 = corres[1][conf_mask].cpu().numpy()  # [K, 2] x,y

            if len(matches_im0) == 0:
                continue

            # Convert to ravel format and store
            H0, W0 = batch_views[k][0]["true_shape"][0]
            H1, W1 = batch_views[k][1]["true_shape"][0]

            qx0 = matches_im0[:, 0].round().astype(np.int32).clip(0, W0 - 1)
            qy0 = matches_im0[:, 1].round().astype(np.int32).clip(0, H0 - 1)
            ravel0 = qx0 + W0 * qy0

            qx1 = matches_im1[:, 0].round().astype(np.int32).clip(0, W1 - 1)
            qy1 = matches_im1[:, 1].round().astype(np.int32).clip(0, H1 - 1)
            ravel1 = qx1 + W1 * qy1

            for r in ravel0:
                im_keypoints[gidx0][int(r)] = im_keypoints[gidx0].get(int(r), 0) + 1
            for r in ravel1:
                im_keypoints[gidx1][int(r)] = im_keypoints[gidx1].get(int(r), 0) + 1

            # Store matches ordered by (lo, hi) global index
            imidx_lo, imidx_hi = (gidx0, gidx1) if gidx0 < gidx1 else (gidx1, gidx0)
            if gidx0 < gidx1:
                colmap_matches = np.stack([ravel0, ravel1], axis=-1)
            else:
                colmap_matches = np.stack([ravel1, ravel0], axis=-1)
            colmap_matches = np.unique(colmap_matches, axis=0)

            key = (imidx_lo, imidx_hi)
            if key in im_matches:
                im_matches[key] = np.concatenate([im_matches[key], colmap_matches], axis=0)
                im_matches[key] = np.unique(im_matches[key], axis=0)
            else:
                im_matches[key] = colmap_matches

    # ── Export keypoints and matches to COLMAP DB ────────────────────────
    _con = sqlite3.connect(str(db_path))
    _cur = _con.cursor()

    # Keypoints: convert ravel indices to (x, y) in original image resolution
    for gidx, kp_dict in im_keypoints.items():
        if not kp_dict:
            continue
        if gidx not in view_shape_cache:
            continue
        mast3r_H, mast3r_W = view_shape_cache[gidx]
        sx = W_orig / mast3r_W
        sy = H_orig / mast3r_H

        # Build ordered keypoint list: assign sequential colmap indices
        kp_to_colmap_idx = {}
        kp_xy = []
        for ravel_id in sorted(kp_dict.keys()):
            y, x = divmod(int(ravel_id), mast3r_W)
            kp_to_colmap_idx[ravel_id] = len(kp_xy)
            kp_xy.append([x * sx, y * sy])

        kp_arr = np.array(kp_xy, dtype=np.float32)
        kp_arr[:, 0] = kp_arr[:, 0].clip(0, W_orig - 0.01)
        kp_arr[:, 1] = kp_arr[:, 1].clip(0, H_orig - 0.01)

        cimid = global_to_colmap_imid[gidx]
        _cur.execute(
            "INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?,?,?,?)",
            (cimid, kp_arr.shape[0], 2, kp_arr.tobytes()),
        )
        im_keypoints[gidx]["__idx_map__"] = kp_to_colmap_idx

    _con.commit()

    # Matches: convert ravel IDs to colmap keypoint indices
    def _pair_id(id1, id2):
        lo, hi = (id1, id2) if id1 < id2 else (id2, id1)
        return lo * 2147483647 + hi

    n_exported_pairs = 0
    for (gidx0, gidx1), ravel_matches in im_matches.items():
        idx_map0 = im_keypoints[gidx0].get("__idx_map__")
        idx_map1 = im_keypoints[gidx1].get("__idx_map__")
        if idx_map0 is None or idx_map1 is None:
            continue

        final_matches = []
        for r0, r1 in ravel_matches:
            if int(r0) in idx_map0 and int(r1) in idx_map1:
                final_matches.append([idx_map0[int(r0)], idx_map1[int(r1)]])
        if not final_matches:
            continue

        final_matches = np.array(final_matches, dtype=np.uint32)
        cimid0 = global_to_colmap_imid[gidx0]
        cimid1 = global_to_colmap_imid[gidx1]
        pid = _pair_id(cimid0, cimid1)

        _cur.execute(
            "INSERT OR REPLACE INTO matches (pair_id, rows, cols, data) VALUES (?,?,?,?)",
            (pid, final_matches.shape[0], 2, final_matches.tobytes()),
        )
        n_exported_pairs += 1

    _con.commit()
    _con.close()

    n_total_matches = sum(len(v) for k, v in im_matches.items() if not isinstance(k, str))
    print(f"\nExported to COLMAP DB: {db_path}")
    print(f"  Images with keypoints: {sum(1 for v in im_keypoints.values() if '__idx_map__' in v)}")
    print(f"  Match pairs: {n_exported_pairs}")
    print(f"  Total matches: {n_total_matches}")
    return


@app.cell
def _(colmap_workdir, db_path, img_dir):
    def _():
        """Run COLMAP geometric verification then incremental mapper.

        Step 1: Geometric verification — COLMAP computes essential matrices
        and identifies geometric inliers from our raw MASt3R matches.
        Step 2: Incremental mapping — joint bundle adjustment over all images.
        """
        import pycolmap
        import sqlite3 as _sqlite3

        # ── Build pairs.txt from the matches table ───────────────────────────
        # COLMAP verify_matches needs a text file listing image name pairs
        _con = _sqlite3.connect(str(db_path))
        _cur = _con.cursor()

        # Get image_id → name mapping
        _id_to_name = {}
        for row in _cur.execute("SELECT image_id, name FROM images"):
            _id_to_name[row[0]] = row[1]

        # Get all matched pairs from pair_id
        pairs_path = colmap_workdir / "pairs.txt"
        with open(pairs_path, "w") as _f:
            for row in _cur.execute("SELECT pair_id FROM matches"):
                pid = row[0]
                # Reverse COLMAP pair_id encoding: pair_id = id1 * 2147483647 + id2
                id2 = pid % 2147483647
                id1 = pid // 2147483647
                if id1 in _id_to_name and id2 in _id_to_name:
                    _f.write(f"{_id_to_name[id1]} {_id_to_name[id2]}\n")
        _con.close()
        print(f"Pairs file: {pairs_path}")

        # ── Step 1: Geometric verification ───────────────────────────────────
        print("Running geometric verification...")
        pycolmap.verify_matches(
            str(db_path),
            str(pairs_path),
        )
        print("Geometric verification complete.")

        # ── Step 2: Incremental mapping ──────────────────────────────────────
        import shutil
        sparse_dir = colmap_workdir / "sparse"
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)  # clear stale models from previous runs
        sparse_dir.mkdir()

        print("Running COLMAP incremental mapping...")
        print("  (This may take a few minutes depending on the number of images)")

        opts = pycolmap.IncrementalPipelineOptions({
            "multiple_models": False,
            "extract_colors": True,
            "ba_refine_focal_length": True,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": False,
        })
   
        pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(img_dir),
            output_path=str(sparse_dir),
            options=opts,
        )

        # Read from disk — the returned in-memory dict only reflects intermediate
        # snapshots (e.g. initial 2-image pair), not the final merged model.
        # The largest model on disk is the real result.
        model_dirs = sorted(sparse_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1)
        model_dirs = [d for d in model_dirs if d.is_dir() and d.name.isdigit()]
        if not model_dirs:
            print("WARNING: COLMAP produced no model on disk.")
            return None

        # Pick the model with the most registered images
        best_rec = None
        for md in model_dirs:
            try:
                r = pycolmap.Reconstruction(str(md))
                if best_rec is None or r.num_reg_images() > best_rec.num_reg_images():
                    best_rec = r
            except Exception:
                pass

        if best_rec is None:
            print("WARNING: Could not load any reconstruction from disk.")
            return None

        rec = best_rec
        print("\nReconstruction complete:")
        print(f"  Registered images: {rec.num_reg_images()} / {rec.num_images()}")
        print(f"  3D points:         {rec.num_points3D()}")
        print(f"  Mean reproj error: {rec.compute_mean_reprojection_error():.3f} px")
        return rec


    rec = _()
    return (rec,)


@app.cell
def _(N_fw, np, rec):
    """Extract optimized C2W poses from COLMAP reconstruction.

    Maps COLMAP image names back to fw/bw local indices and builds
    aligned pose arrays.
    """
    import matplotlib.pyplot as _plt
    from scipy.spatial.transform import Rotation as _Rot

    if rec is None:
        print("No reconstruction available.")
        opt_poses_fw = None
        opt_poses_bw = None
    else:
        # Parse optimized poses from COLMAP reconstruction
        # COLMAP stores W2C (world-to-camera); we want C2W (camera-to-world)
        opt_poses = {}  # global_idx → 4x4 C2W
        for img in rec.images.values():
            name = img.name  # e.g., "fwd_000050.jpg" or "bwd_000123.jpg"
            prefix, idx_str = name.split("_", 1)
            local_idx = int(idx_str.replace(".jpg", ""))
            global_idx = local_idx if prefix == "fwd" else N_fw + local_idx

            # COLMAP cam_from_world = W2C (method in pycolmap 3.13+)
            rigid = img.cam_from_world()
            mat34 = np.array(rigid.matrix())  # 3x4
            T_w2c = np.eye(4)
            T_w2c[:3, :] = mat34
            T_c2w = np.linalg.inv(T_w2c)
            opt_poses[global_idx] = T_c2w

        # Separate into fw and bw
        opt_fw_indices = sorted(k for k in opt_poses if k < N_fw)
        opt_bw_indices = sorted(k for k in opt_poses if k >= N_fw)

        opt_poses_fw = {k: opt_poses[k] for k in opt_fw_indices}
        opt_poses_bw = {k - N_fw: opt_poses[k] for k in opt_bw_indices}

        print(f"Optimized poses: {len(opt_poses_fw)} fw, {len(opt_poses_bw)} bw")

        # ── Visualise trajectories ───────────────────────────────────────
        _fig = _plt.figure(figsize=(14, 5))

        _ax1 = _fig.add_subplot(121, projection="3d")
        _fw_t = np.array([opt_poses_fw[k][:3, 3] for k in sorted(opt_poses_fw)])
        _bw_t = np.array([opt_poses_bw[k][:3, 3] for k in sorted(opt_poses_bw)])
        _ax1.plot(*_fw_t.T, "b-", lw=1, label="fw (optimized)")
        _ax1.plot(*_bw_t.T, "r-", lw=1, label="bw (optimized)")
        _ax1.set_title("COLMAP optimized trajectories")
        _ax1.legend(fontsize=7)

        # Plot triangulated 3D points (subsampled)
        _ax2 = _fig.add_subplot(122, projection="3d")
        _pts_xyz = np.array([pt.xyz for pt in rec.points3D.values()])
        _pts_rgb = np.array([pt.color for pt in rec.points3D.values()]) / 255.0
        if len(_pts_xyz) > 5000:
            _subsample = np.random.choice(len(_pts_xyz), 5000, replace=False)
            _pts_xyz = _pts_xyz[_subsample]
            _pts_rgb = _pts_rgb[_subsample]
        _ax2.scatter(*_pts_xyz.T, c=_pts_rgb, s=0.5, alpha=0.5)
        _ax2.plot(*_fw_t.T, "b-", lw=1)
        _ax2.plot(*_bw_t.T, "r-", lw=1)
        _ax2.set_title(f"Triangulated points ({rec.num_points3D():,})")

        _plt.tight_layout()
        _plt.show()
    return opt_poses_bw, opt_poses_fw


@app.cell
def _(
    colmap_workdir,
    device,
    model,
    np,
    opt_poses_bw,
    opt_poses_fw,
    rect_frames_bw,
    rect_frames_fw,
):
    def _():
        def _():
            """Re-integrate MASt3R depth maps using COLMAP-optimized poses.

            Instead of using the original per-pass poses (which have independent drift),
            we use the globally consistent poses from COLMAP's bundle adjustment.
            This should eliminate the boundary duplication issue.
            """
            import open3d as _o3d
            import torchvision.transforms as _T
            from PIL import Image as _PIL
            from tqdm import tqdm as _tqdm

            from dust3r.inference import inference as _dust3r_inference
            from mast3r.fast_nn import extract_correspondences_nonsym as _extract_corres

            if opt_poses_fw is None or opt_poses_bw is None:
                print("No optimized poses available. Skipping depth re-integration.")
            else:
                TARGET_SIZE = 512
                CONF_THR = 2.0
                _ImgNorm = _T.Compose([_T.ToTensor(), _T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

                def _load_view_simple(path):
                    pil = _PIL.open(path).convert("RGB")
                    W, H = pil.size
                    scale = TARGET_SIZE / max(H, W)
                    new_H = round(H * scale / 16) * 16
                    new_W = round(W * scale / 16) * 16
                    pil = pil.resize((new_W, new_H), _PIL.Resampling.LANCZOS)
                    return dict(
                        img=_ImgNorm(pil)[None],
                        true_shape=np.int32([[new_H, new_W]]),
                        idx=0,
                        instance="0",
                    ), pil

                # Use sequential frame pairs to get depth maps, then project to world
                # using the optimized poses
                all_pts = []
                all_rgb = []

                # Process forward pass: sequential pairs
                fw_indices = sorted(opt_poses_fw.keys())
                print(f"Re-integrating {len(fw_indices)} fw frames...")
                for i in _tqdm(range(0, len(fw_indices) - 1, 2), desc="FW depth"):
                    fi = fw_indices[i]
                    fj = fw_indices[min(i + 1, len(fw_indices) - 1)]

                    view0, pil0 = _load_view_simple(rect_frames_fw[fi])
                    view1, _ = _load_view_simple(rect_frames_fw[fj])
                    view0["idx"] = 0
                    view0["instance"] = "0"
                    view1["idx"] = 1
                    view1["instance"] = "1"

                    result = _dust3r_inference([(view0, view1)], model, device, batch_size=1, verbose=False)
                    pts3d = result["pred1"]["pts3d"][0].cpu().numpy()  # [H, W, 3] in cam frame
                    conf = result["pred1"]["conf"][0].cpu().numpy()

                    valid = (conf > CONF_THR) & np.isfinite(pts3d).all(axis=-1)
                    pts_cam = pts3d[valid]
                    if len(pts_cam) == 0:
                        continue

                    C2W = opt_poses_fw[fi]
                    pts_world = (C2W[:3, :3] @ pts_cam.T + C2W[:3, 3:4]).T
                    rgb = np.array(pil0).reshape(-1, 3)[valid.ravel()].astype(np.float64) / 255.0

                    all_pts.append(pts_world)
                    all_rgb.append(rgb)

                # Process backward pass: sequential pairs
                bw_indices = sorted(opt_poses_bw.keys())
                print(f"Re-integrating {len(bw_indices)} bw frames...")
                for i in _tqdm(range(0, len(bw_indices) - 1, 2), desc="BW depth"):
                    bi = bw_indices[i]
                    bj = bw_indices[min(i + 1, len(bw_indices) - 1)]

                    view0, pil0 = _load_view_simple(rect_frames_bw[bi])
                    view1, _ = _load_view_simple(rect_frames_bw[bj])
                    view0["idx"] = 0
                    view0["instance"] = "0"
                    view1["idx"] = 1
                    view1["instance"] = "1"

                    result = _dust3r_inference([(view0, view1)], model, device, batch_size=1, verbose=False)
                    pts3d = result["pred1"]["pts3d"][0].cpu().numpy()
                    conf = result["pred1"]["conf"][0].cpu().numpy()

                    valid = (conf > CONF_THR) & np.isfinite(pts3d).all(axis=-1)
                    pts_cam = pts3d[valid]
                    if len(pts_cam) == 0:
                        continue

                    C2W = opt_poses_bw[bi]
                    pts_world = (C2W[:3, :3] @ pts_cam.T + C2W[:3, 3:4]).T
                    rgb = np.array(pil0).reshape(-1, 3)[valid.ravel()].astype(np.float64) / 255.0

                    all_pts.append(pts_world)
                    all_rgb.append(rgb)

                # ── Merge and deduplicate ────────────────────────────────────────
                all_pts_arr = np.concatenate(all_pts, axis=0)
                all_rgb_arr = np.concatenate(all_rgb, axis=0)
                print(f"Raw points: {len(all_pts_arr):,}")

                pcd = _o3d.geometry.PointCloud()
                pcd.points = _o3d.utility.Vector3dVector(all_pts_arr.astype(np.float64))
                pcd.colors = _o3d.utility.Vector3dVector(all_rgb_arr)

                # Auto-scale voxel size from scene extent
                extent = all_pts_arr.max(0) - all_pts_arr.min(0)
                scene_scale = float(extent.max())
                voxel_size = scene_scale / 1600
                print(f"Scene scale: {scene_scale:.3f}, voxel size: {voxel_size:.5f}")

                pcd = pcd.voxel_down_sample(voxel_size)
                print(f"After voxel downsample: {len(pcd.points):,}")

                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                print(f"After outlier removal: {len(pcd.points):,}")

                out_path = colmap_workdir / "merged_optimized.ply"
                _o3d.io.write_point_cloud(str(out_path), pcd)
            return print(f"Saved merged point cloud: {out_path}")
        return _()


    _()
    return


if __name__ == "__main__":
    app.run()
