import marimo

__generated_with = "0.19.11"
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
    # (mast3r uses `from mast3r.xyz import ...` internally)
    MAST3R_REPO_PATH = Path(
        "/home/hugues/Documents/phd-workspace/projects/reef_change_detection/src/reef_change_detection/mast3r"
    )
    if str(MAST3R_REPO_PATH) not in sys.path:
        sys.path.insert(0, str(MAST3R_REPO_PATH))
    return Path, np


@app.cell
def _(Path, np):
    frames_2022_dir = Path(
        "/home/hugues/Documents/phd-workspace/data/change-detection-data/arta-ras-korali/2022/dive3/cam1/frames"
    )
    frames_2025_dir = Path(
        "/home/hugues/Documents/phd-workspace/data/change-detection-data/arta-ras-korali/2025/cam1/frames"
    )

    frames_2025_bw_dir = Path(
        "/home/hugues/Documents/phd-workspace/data/change-detection-data/arta-ras-korali/2025/cam2/frames"
    )

    frames_makassar_dir = Path(
        "/home/hugues/Documents/phd-workspace/data/change-detection-data/makassar/2025/day1/cam6_johnny/frames"
    )

    frames_2022 = list(frames_2022_dir.glob("*.jpg"))
    frames_2025 = list(frames_2025_dir.glob("*.jpg"))
    frames_2025_bw = list(frames_2025_bw_dir.glob("*.jpg"))
    frames_makassar = list(frames_makassar_dir.glob("*.jpg"))

    print("Found", len(frames_2022), "frames in 2022")
    print("Found", len(frames_2025), "frames in 2025")
    print("Found", len(frames_2025_bw), "frames in 2025 (Backward transect)")
    print("Found", len(frames_makassar), "frames in makassar")


    def interpolate_frame_number(frame_vid_1, n_frames_1, n_frames_2):
        return int(np.floor(frame_vid_1 * (n_frames_2 / n_frames_1)))

    return frames_2022, frames_2022_dir, frames_2025, frames_2025_bw


@app.cell
def _():
    # Now we can import directly since mast3r repo is on sys.path
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    import torch

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use from_pretrained to load the model (not the constructor)
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    return device, model, torch


@app.cell
def _(torch):
    import torchvision.transforms as T

    dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dinov2_model = dinov2_model.eval().to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dinov2_transform = T.Compose([
        T.Resize((518, 518), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return dinov2_model, dinov2_transform


@app.cell
def _(dinov2_model, dinov2_transform, torch):
    from PIL import Image
    import numpy as _np

    def extract_dinov2_descriptors(frame_paths, batch_size=32):
        """Extract DINOv2 CLS token descriptors.
        Returns: np.ndarray (N, 768), L2-normalized, float32.
        """
        device = next(dinov2_model.parameters()).device
        all_desc = []
        for i in range(0, len(frame_paths), batch_size):
            batch = [dinov2_transform(Image.open(str(p)).convert("RGB"))
                     for p in frame_paths[i:i+batch_size]]
            batch_tensor = torch.stack(batch).to(device)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
                cls_tokens = dinov2_model(batch_tensor)  # (B, 768)
            cls_tokens = torch.nn.functional.normalize(cls_tokens, p=2, dim=-1)
            all_desc.append(cls_tokens.cpu())
        desc = torch.cat(all_desc, dim=0).numpy().astype(_np.float32)
        return desc

    return (extract_dinov2_descriptors,)


@app.cell
def _(extract_dinov2_descriptors):
    import faiss
    import numpy as _np_faiss

    def build_retrieval_index(frame_paths, batch_size=32):
        """Build FAISS index over DINOv2 CLS descriptors."""
        descriptors = extract_dinov2_descriptors(frame_paths, batch_size=batch_size)
        index = faiss.IndexFlatIP(descriptors.shape[1])  # 768
        index.add(descriptors)
        return index, descriptors

    def retrieve_candidates(
        query_paths, faiss_index, candidate_paths,
        top_k=20, batch_size=32, temporal_prior=None,
    ):
        """Stage 1: retrieve top-k candidates per query via DINOv2 cosine similarity.

        temporal_prior: optional dict with "previous_match_idx" and "window_size".
            Restricts search to a window around the previous best match.
            Set to None for the first query or when not using temporal constraint.
        """
        query_descs = extract_dinov2_descriptors(query_paths, batch_size=batch_size)
        results = []
        for qi in range(len(query_paths)):
            q = query_descs[qi:qi+1]
            if temporal_prior and temporal_prior.get("previous_match_idx") is not None:
                prev = temporal_prior["previous_match_idx"]
                w = temporal_prior.get("window_size", 50)
                lo = max(0, prev - w)
                hi = min(faiss_index.ntotal, prev + w + 1)
                window_descs = _np_faiss.vstack([
                    faiss_index.reconstruct(j) for j in range(lo, hi)
                ])
                sub = faiss.IndexFlatIP(q.shape[1])
                sub.add(window_descs)
                k = min(top_k, hi - lo)
                sims, idxs = sub.search(q, k)
                results.append([(lo + int(idxs[0, j]), float(sims[0, j])) for j in range(k)])
            else:
                k = min(top_k, faiss_index.ntotal)
                sims, idxs = faiss_index.search(q, k)
                results.append([(int(idxs[0, j]), float(sims[0, j])) for j in range(k)])
        return results

    return build_retrieval_index, retrieve_candidates


@app.cell
def _(device, frames_2022, frames_2025, model, np, torch):
    from dust3r.utils.image import load_images
    from dust3r.inference import inference
    from mast3r.fast_nn import extract_correspondences_nonsym
    images = load_images([str(frames_2022[0]), str(frames_2025[10])], size=512)
    _output = inference([tuple(images)], model, device, batch_size=1)
    (_view1, _pred1) = (_output['view1'], _output['pred1'])
    (_view2, _pred2) = (_output['view2'], _output['pred2'])
    (_desc1, _desc2) = (_pred1['desc'].squeeze(0).detach(), _pred2['desc'].squeeze(0).detach())
    confidences = [_pred1['desc_conf'].squeeze(0).detach(), _pred2['desc_conf'].squeeze(0).detach()]
    _corres = extract_correspondences_nonsym(_desc1, _desc2, confidences[0], confidences[1], device=device, subsample=8, pixel_tol=0)
    _conf = _corres[2]
    _mask = _conf >= 0.5
    _matches_im0 = _corres[0][_mask].cpu().numpy()
    _matches_im1 = _corres[1][_mask].cpu().numpy()
    (_H0, _W0) = _view1['true_shape'][0]
    valid_matches_im0 = (_matches_im0[:, 0] >= 3) & (_matches_im0[:, 0] < int(_W0) - 3) & (_matches_im0[:, 1] >= 3) & (_matches_im0[:, 1] < int(_H0) - 3)
    (_H1, _W1) = _view2['true_shape'][0]
    valid_matches_im1 = (_matches_im1[:, 0] >= 3) & (_matches_im1[:, 0] < int(_W1) - 3) & (_matches_im1[:, 1] >= 3) & (_matches_im1[:, 1] < int(_H1) - 3)
    valid_matches = valid_matches_im0 & valid_matches_im1
    (_matches_im0, _matches_im1) = (_matches_im0[valid_matches], _matches_im1[valid_matches])
    import torchvision.transforms.functional
    from matplotlib import pyplot as pl
    _n_viz = 10
    _num_matches = _matches_im0.shape[0]
    _match_idx_to_viz = np.round(np.linspace(0, _num_matches - 1, _n_viz)).astype(int)
    (_viz_matches_im0, _viz_matches_im1) = (_matches_im0[_match_idx_to_viz], _matches_im1[_match_idx_to_viz])
    _image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    _image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    _viz_imgs = []
    for (_i, _view) in enumerate([_view1, _view2]):
        _rgb_tensor = _view['img'] * _image_std + _image_mean
        _viz_imgs.append(_rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    (_H0, _W0, _H1, _W1) = (*_viz_imgs[0].shape[:2], *_viz_imgs[1].shape[:2])
    _img0 = np.pad(_viz_imgs[0], ((0, max(_H1 - _H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    _img1 = np.pad(_viz_imgs[1], ((0, max(_H0 - _H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    _img = np.concatenate((_img0, _img1), axis=1)
    pl.figure()
    pl.imshow(_img)
    _cmap = pl.get_cmap('jet')
    for _i in range(_n_viz):
        ((_x0, _y0), (_x1, _y1)) = (_viz_matches_im0[_i].T, _viz_matches_im1[_i].T)
        pl.plot([_x0, _x1 + _W0], [_y0, _y1], '-+', color=_cmap(_i / (_n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)
    return extract_correspondences_nonsym, inference, load_images


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MASt3R → COLMAP Pipeline

    **Workflow:**
    1. **Load images** - Use `load_images()` to prepare images for MASt3R
    2. **Generate pairs** - Use `make_pairs()` with a scene graph strategy:
       - `'swin-5'`: Sliding window (good for video sequences)
       - `'logwin-5'`: Log-spaced window (better for longer sequences)
       - `'complete'`: All pairs (slow, only for <50 images)
    3. **Run inference** - Extract dense features and 3D predictions
    4. **Extract matches** - Get 2D-2D correspondences from descriptors
    5. **Export to COLMAP** - Create database with keypoints and matches
    6. **Run reconstruction** - COLMAP incremental mapper or glomap
    """)
    return


@app.cell
def _(frames_2022, frames_2022_dir, model):
    # =============================================================================
    # MASt3R → COLMAP Pipeline for 100 images
    import os
    # This shows how to extract MASt3R features and export them to COLMAP
    from tqdm import tqdm
    from mast3r.image_pairs import make_pairs
    from mast3r.colmap.mapping import scene_prepare_images
    image_root = str(frames_2022_dir)
    image_paths = [str(frames_2022[0]), str(frames_2022[0])]
    print(f'Selected {len(image_paths)} images from {image_root}')
    maxdim = max(model.patch_embed.img_size)
    patch_size = model.patch_embed.patch_size
    images_1 = scene_prepare_images(image_root, maxdim, patch_size, image_paths)
    # 1. Select 100 images (sorted for sequential pairing)
    # Use relative paths from a common root directory
    # image_paths = sorted([p.name for p in frames_2022[:100]])  # relative paths
    # 2. Load images for MASt3R using scene_prepare_images (includes metadata for COLMAP export)
    # Get model parameters for proper image sizing
    print(f'Loaded {len(images_1)} images with shape metadata')  # typically 512  # typically 16
    return image_paths, image_root, images_1, make_pairs, os, tqdm


@app.cell
def _(images_1, make_pairs):
    # 3. Generate image pairs using a scene graph strategy
    # For sequential video frames, use sliding window ('swin-N') or log window ('logwin-N')
    # - 'complete': all pairs (slow for 100+ images: 4950 pairs!)
    # - 'swin-5': sliding window of 5 neighbors (efficient for video)
    # - 'logwin-5': log-spaced window (good for longer sequences)
    # - 'oneref-0': all images matched to image 0
    pairs = make_pairs(images_1, scene_graph='swin-5', symmetrize=True)
    print(f'Generated {len(pairs)} pairs (with symmetrization)')
    return (pairs,)


@app.cell
def _(inference, model, pairs, torch, tqdm):
    device_1 = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    all_preds = []
    for _i in tqdm(range(0, len(pairs), batch_size), desc='Running MASt3R inference'):
        batch_pairs = pairs[_i:_i + batch_size]
        with torch.no_grad():
            _output = inference(batch_pairs, model, device_1, batch_size=batch_size, verbose=False)
        all_preds.append(_output)
    print(f'Processed {len(pairs)} pairs')
    return all_preds, device_1


@app.cell
def _(all_preds, device_1, extract_correspondences_nonsym):
    def extract_matches_from_pred(pred1, pred2, conf_thr=1.5):
        """Extract 2D-2D matches from MASt3R predictions."""
        matches_list = []
        for _i in range(len(_pred1['pts3d'])):
            descs = [_pred1['desc'][_i], _pred2['desc'][_i]]
            confidences = [_pred1['desc_conf'][_i], _pred2['desc_conf'][_i]]
            _corres = extract_correspondences_nonsym(descs[0], descs[1], confidences[0], confidences[1], device=device_1, subsample=8, pixel_tol=0)
            _conf = _corres[2]
            _mask = _conf >= conf_thr
            _matches_im0 = _corres[0][_mask].cpu().numpy()
            _matches_im1 = _corres[1][_mask].cpu().numpy()
            matches_list.append({'pts0': _matches_im0, 'pts1': _matches_im1, 'conf': _conf[_mask].cpu().numpy()})
        return matches_list
    if all_preds:
        example_matches = extract_matches_from_pred(all_preds[0]['pred1'], all_preds[0]['pred2'])
        print(f"Example: found {len(example_matches[0]['pts0'])} matches in first pair")
    return


@app.cell
def _(all_preds, device_1, image_paths, image_root, images_1, os, pairs):
    import pycolmap
    from mast3r.colmap.database import export_matches, export_images, get_im_matches

    def mast3r_to_colmap(image_paths, images, pairs, all_preds, output_dir, image_root, conf_thr=1.5):
        """
        Export MASt3R features to COLMAP database.

        Args:
            image_paths: List of relative image file paths
            images: Loaded images from scene_prepare_images()
            pairs: Image pairs from make_pairs()
            all_preds: MASt3R predictions from inference()
            output_dir: Output directory for COLMAP files
            image_root: Root directory containing images
            conf_thr: Confidence threshold for matches
        """
        from kapture.converter.colmap.database import COLMAPDatabase
        os.makedirs(output_dir, exist_ok=True)
        db_path = os.path.join(output_dir, 'colmap.db')
        if os.path.exists(db_path):
            os.remove(db_path)
        db = COLMAPDatabase.connect(db_path)
        db.create_tables()
        (image_to_colmap, im_keypoints) = export_images(db, images, [_img['instance'] for _img in images], focals=None, ga_world_to_cam=None, camera_model='SIMPLE_RADIAL')
        im_matches = {}
        pair_idx = 0
        for pred_batch in all_preds:
            batch_matches = get_im_matches(pred1=pred_batch['pred1'], pred2=pred_batch['pred2'], pairs=pairs[pair_idx:pair_idx + len(pred_batch['pred1']['pts3d'])], image_to_colmap=image_to_colmap, im_keypoints=im_keypoints, conf_thr=conf_thr, is_sparse=True, pixel_tol=0, device=device_1)
            im_matches.update(batch_matches)
            pair_idx = pair_idx + len(pred_batch['pred1']['pts3d'])
        colmap_image_pairs = export_matches(db, images, image_to_colmap, im_keypoints, im_matches, min_len_track=3, skip_geometric_verification=False)
        db.commit()
        db.close()
        print(f'Created COLMAP database at {db_path}')
        print(f'Exported {len(colmap_image_pairs)} image pairs with matches')
        return (db_path, colmap_image_pairs)
    output_dir = '/home/hugues/Documents/phd-workspace/data/mast3r_colmap_output'
    (db_path, colmap_pairs) = mast3r_to_colmap(image_paths, images_1, pairs, all_preds, output_dir, image_root)
    return colmap_pairs, db_path, output_dir, pycolmap


@app.cell
def _(colmap_pairs, db_path, image_root, os, output_dir, pycolmap):
    _sparse_dir = os.path.join(output_dir, 'sparse')
    os.makedirs(_sparse_dir, exist_ok=True)
    pairs_path = os.path.join(output_dir, 'pairs.txt')
    with open(pairs_path, 'w') as f:
        for (_img1, img2) in colmap_pairs:
            f.write(f'{_img1} {img2}\n')
    print(f'Created {pairs_path} with {len(colmap_pairs)} pairs')
    print('Verifying matches...')
    pycolmap.verify_matches(database_path=db_path, pairs_path=pairs_path)
    print('Running reconstruction...')
    _reconstructions = pycolmap.incremental_mapping(database_path=db_path, image_path=image_root, output_path=_sparse_dir)
    print(f'\nReconstruction complete!')
    print(f'Output saved to: {_sparse_dir}')
    print(f'\nOpen in COLMAP GUI with:')
    print(f'  colmap gui --import_path {_sparse_dir}/0')
    return


@app.cell
def _(db_path, image_root, os, pycolmap):
    output_dir_1 = '/home/hugues/Documents/phd-workspace/data/mast3r_colmap_output'
    _sparse_dir = os.path.join(output_dir_1, 'sparse')
    os.makedirs(_sparse_dir, exist_ok=True)
    pycolmap.verify_matches(database_path=db_path, pairs_path=os.path.join(output_dir_1, 'pairs.txt'))
    _reconstructions = pycolmap.incremental_mapping(database_path=db_path, image_path=image_root, output_path=_sparse_dir)
    print(f'Reconstruction saved to {_sparse_dir}')
    print(f'Open with: colmap gui --import_path {_sparse_dir}/0')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frame Retrieval: Find Best Matching Frames

    Given a query frame from one time period, find the best matching frames from another time period using MASt3R correspondences.
    """)
    return


@app.cell
def _(torch):
    from dust3r.utils.image import load_images as _load_images
    from dust3r.inference import inference as _inference
    from mast3r.fast_nn import extract_correspondences_nonsym as _extract_corr

    def preload_images(frame_paths, size=512):
        """Pre-load images once so they can be reused across queries."""
        return _load_images([str(p) for p in frame_paths], size=size)

    def _mast3r_score_pairs(
        query_frame_path, candidate_frame_paths, mast3r_model,
        device='cuda', conf_threshold=1.0, batch_size=8,
        preloaded_query=None, preloaded_candidates=None,
    ):
        """Run MASt3R on (query, candidate) pairs and return scored results.

        Returns list of (candidate_path, num_matches, avg_confidence) sorted
        by (num_matches, avg_confidence) descending.
        """
        from tqdm import tqdm

        query_img = (preloaded_query if preloaded_query is not None
                     else _load_images([str(query_frame_path)], size=512)[0])

        results = []
        for _i in tqdm(range(0, len(candidate_frame_paths), batch_size), desc='MASt3R scoring'):
            batch_paths = candidate_frame_paths[_i:_i + batch_size]
            if preloaded_candidates is not None:
                batch_imgs = preloaded_candidates[_i:_i + batch_size]
            else:
                batch_imgs = _load_images([str(p) for p in batch_paths], size=512)

            pairs = [(query_img, cand_img) for cand_img in batch_imgs]
            with torch.no_grad():
                _output = _inference(pairs, mast3r_model, device, batch_size=len(pairs), verbose=False)
            del pairs

            for (j, cand_path) in enumerate(batch_paths):
                desc_query = _output['pred1']['desc'][j].squeeze(0)
                desc_cand = _output['pred2']['desc'][j].squeeze(0)
                conf_query = _output['pred1']['desc_conf'][j].squeeze(0)
                conf_cand = _output['pred2']['desc_conf'][j].squeeze(0)
                _corres = _extract_corr(desc_query, desc_cand, conf_query, conf_cand, device=device, subsample=8, pixel_tol=0)
                _conf = _corres[2]
                valid_mask = _conf >= conf_threshold
                _num_matches = valid_mask.sum().item()
                _avg_conf = _conf[valid_mask].mean().item() if _num_matches > 0 else 0.0
                results.append((cand_path, _num_matches, _avg_conf))

            del _output
            torch.cuda.empty_cache()

        results.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return results

    def find_best_matching_frames(
        query_frame_path, candidate_frame_paths, model,
        device='cuda', top_k=5, conf_threshold=1.0, batch_size=8,
        preloaded_query=None, preloaded_candidates=None,
    ):
        """Brute-force MASt3R matching (original, kept for backward compat)."""
        results = _mast3r_score_pairs(
            query_frame_path, candidate_frame_paths, model,
            device=device, conf_threshold=conf_threshold, batch_size=batch_size,
            preloaded_query=preloaded_query, preloaded_candidates=preloaded_candidates,
        )
        return results[:top_k]

    def find_best_matching_frames_twostage(
        query_frame_path, candidate_frame_paths, mast3r_model,
        faiss_index, retrieve_fn, device='cuda',
        retrieval_top_k=20, final_top_k=5, conf_threshold=1.0,
        batch_size=8, preloaded_query=None, preloaded_candidates=None,
        temporal_prior=None,
    ):
        """Two-stage: DINOv2 shortlist -> MASt3R reranking.

        Returns:
            top_results: [(frame_path, num_matches, avg_confidence), ...] (length <= final_top_k)
            shortlist: [(candidate_index, dinov2_similarity), ...] from Stage 1
        """
        # Stage 1: DINOv2 retrieval
        shortlist = retrieve_fn(
            query_paths=[query_frame_path], faiss_index=faiss_index,
            candidate_paths=candidate_frame_paths, top_k=retrieval_top_k,
            temporal_prior=temporal_prior,
        )[0]

        # Stage 2: MASt3R dense matching on shortlist only
        shortlist_indices = [idx for idx, _ in shortlist]
        shortlist_paths = [candidate_frame_paths[idx] for idx in shortlist_indices]

        # Build preloaded subset if full candidates are preloaded
        shortlist_preloaded = None
        if preloaded_candidates is not None:
            shortlist_preloaded = [preloaded_candidates[idx] for idx in shortlist_indices]

        mast3r_results = _mast3r_score_pairs(
            query_frame_path, shortlist_paths, mast3r_model,
            device=device, conf_threshold=conf_threshold, batch_size=batch_size,
            preloaded_query=preloaded_query, preloaded_candidates=shortlist_preloaded,
        )

        top_results = mast3r_results[:final_top_k]
        return top_results, shortlist

    return find_best_matching_frames_twostage, preload_images


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Retrieval Evaluation: How Well Does Frame Matching Work?

    For every Nth frame in 2025, retrieve the top-k matching frames in 2022 and measure:
    1. **Hit rate**: fraction of queries where at least one top-k match has >= C correspondences
    2. **Temporal coherence**: do retrieved frame indices correlate with query frame indices?
       (Assuming same transect direction, early frames in 2025 should match early frames in 2022)

    Evaluated on two video pairs:
    - **Pair 1**: 2025 cam1 (forward) vs 2022
    - **Pair 2**: 2025 cam2 (backward) vs 2025 cam1 (forward)
    """)
    return


@app.cell
def _():
    import re
    from scipy.stats import spearmanr

    def extract_frame_number(path):
        """Extract numeric frame index from filename."""
        nums = re.findall(r'\d+', path.stem)
        return int(nums[-1]) if nums else 0

    return extract_frame_number, spearmanr


@app.cell
def _(
    build_retrieval_index,
    extract_frame_number,
    find_best_matching_frames_twostage,
    np,
    preload_images,
    retrieve_candidates,
    spearmanr,
):
    def evaluate_retrieval(
        query_frames,
        candidate_frames,
        model,
        device,
        top_k=5,
        conf_threshold=1.0,
        min_correspondences=100,
        batch_size=16,
        retrieval_top_k=20,
        use_temporal_prior=False,
        temporal_window=50,
    ):
        """
        Run two-stage retrieval evaluation on a query/candidate pair.

        Stage 1: DINOv2 CLS token retrieval (FAISS cosine search)
        Stage 2: MASt3R dense matching on top-k shortlist

        Returns dict with arrays and scalar metrics (superset of original):
        - query_indices, match_indices, n_correspondences_best: per-query arrays
        - n_hits_per_query: how many of top-k exceed min_correspondences per query
        - hit_rate: fraction of queries with at least one good hit in top-k
        - spearman_corr / spearman_p: temporal coherence of best-match indices
        - retrieval_similarities: per-query best DINOv2 similarity (Stage 1)
        - index_build_time_s: one-time FAISS index build cost
        """
        import time
        from tqdm import tqdm

        # Build FAISS index over candidates (one-time)
        print(f"  Building DINOv2 FAISS index over {len(candidate_frames)} candidates...")
        t0 = time.time()
        faiss_index, _descriptors = build_retrieval_index(candidate_frames, batch_size=batch_size)
        index_build_time = time.time() - t0
        print(f"  Index built in {index_build_time:.1f}s")

        # Pre-load MASt3R candidate images once
        print(f"  Pre-loading {len(candidate_frames)} candidate images for MASt3R...")
        cached_candidates = preload_images(candidate_frames, size=512)

        query_indices = []
        match_indices = []
        n_correspondences_best = []
        n_hits_per_query = []
        retrieval_similarities = []
        prev_match_idx = None

        for query in tqdm(query_frames, desc="Evaluating retrieval (two-stage)"):
            temporal_prior = None
            if use_temporal_prior and prev_match_idx is not None:
                temporal_prior = {
                    "previous_match_idx": prev_match_idx,
                    "window_size": temporal_window,
                }

            query_img = preload_images([query], size=512)[0]
            top_matches, shortlist = find_best_matching_frames_twostage(
                query, candidate_frames, model, faiss_index=faiss_index,
                retrieve_fn=retrieve_candidates, device=device,
                retrieval_top_k=retrieval_top_k, final_top_k=top_k,
                conf_threshold=conf_threshold, batch_size=batch_size,
                preloaded_query=query_img, preloaded_candidates=cached_candidates,
                temporal_prior=temporal_prior,
            )

            # Best DINOv2 similarity from Stage 1
            best_sim = shortlist[0][1] if shortlist else 0.0
            retrieval_similarities.append(best_sim)

            # Count how many of the top-k pass the correspondences threshold
            hits = sum(1 for m in top_matches if m[1] >= min_correspondences)
            n_hits_per_query.append(hits)

            # Track frame indices for temporal coherence
            q_idx = extract_frame_number(query)
            best_idx = extract_frame_number(top_matches[0][0]) if top_matches else -1
            query_indices.append(q_idx)
            match_indices.append(best_idx)
            n_correspondences_best.append(top_matches[0][1] if top_matches else 0)

            # Update temporal prior: find index of best match in candidate list
            if top_matches and use_temporal_prior:
                best_path = top_matches[0][0]
                try:
                    prev_match_idx = candidate_frames.index(best_path)
                except ValueError:
                    # Path object vs string mismatch — linear scan
                    for ci, cp in enumerate(candidate_frames):
                        if str(cp) == str(best_path):
                            prev_match_idx = ci
                            break

        query_indices = np.array(query_indices)
        match_indices = np.array(match_indices)
        n_correspondences_best = np.array(n_correspondences_best)
        n_hits_per_query = np.array(n_hits_per_query)
        retrieval_similarities = np.array(retrieval_similarities)

        hit_rate = float(np.mean(n_hits_per_query > 0))

        if len(query_indices) > 2:
            corr, p_value = spearmanr(query_indices, match_indices)
        else:
            corr, p_value = 0.0, 1.0

        return {
            "query_indices": query_indices,
            "match_indices": match_indices,
            "n_correspondences_best": n_correspondences_best,
            "n_hits_per_query": n_hits_per_query,
            "hit_rate": hit_rate,
            "spearman_corr": corr,
            "spearman_p": p_value,
            "retrieval_similarities": retrieval_similarities,
            "index_build_time_s": index_build_time,
        }

    return (evaluate_retrieval,)


@app.cell
def _(device, evaluate_retrieval, frames_2022, frames_2025, model):
    # Pair 1: 2025 cam1 (forward) vs 2022
    import random
    _queries = sorted(frames_2025)[:100:10]
    _candidates = sorted(frames_2022)[:3000:10]

    print("Pair 1: 2025 cam1 (forward) -> 2022")
    print(f"  Queries: {len(_queries)}, Candidates: {len(_candidates)}")

    eval_pair1 = evaluate_retrieval(
        _queries, _candidates, model, device=device,
        top_k=5, conf_threshold=1.0, min_correspondences=100, batch_size=32,
        retrieval_top_k=40, use_temporal_prior=True,
    )

    print(f"\n  Hit rate (>=100 corr in top-5): {eval_pair1['hit_rate']:.1%}")
    print(f"  Spearman rho: {eval_pair1['spearman_corr']:.3f} (p={eval_pair1['spearman_p']:.2e})")
    return (eval_pair1,)


@app.cell
def _(device, evaluate_retrieval, frames_2025, frames_2025_bw, model):
    # Pair 2: 2025 cam2 (backward) vs 2025 cam1 (forward)
    # Queries sorted when using temporal prior (sequential assumption)
    #_queries = sorted(random.sample(sorted(frames_2025_bw), min(10, len(frames_2025_bw))))
    _queries = sorted(frames_2025, reverse=True)[:200:10]
    _candidates = sorted(frames_2025_bw)[::10]

    print("Pair 2: 2025 cam2 (backward) -> 2025 cam1 (forward)")
    print(f"  Queries: {len(_queries)}, Candidates: {len(_candidates)}")

    eval_pair2 = evaluate_retrieval(
        _queries, _candidates, model, device=device,
        top_k=5, conf_threshold=1.0, min_correspondences=100, batch_size=16,
        retrieval_top_k=40, use_temporal_prior=False, temporal_window=50,
    )

    print(f"\n  Hit rate (>=100 corr in top-5): {eval_pair2['hit_rate']:.1%}")
    print(f"  Spearman rho: {eval_pair2['spearman_corr']:.3f} (p={eval_pair2['spearman_p']:.2e})")
    return (eval_pair2,)


@app.cell
def _(eval_pair1, eval_pair2):
    import matplotlib.pyplot as plt_eval

    fig, axes = plt_eval.subplots(2, 3, figsize=(18, 10))

    for row, (data, name) in enumerate([
        (eval_pair1, "2025 cam1 (forward) vs 2022"),
        (eval_pair2, "2025 cam2 (backward) vs 2025 cam1 (forward)"),
    ]):
        # 1. Temporal coherence scatter
        ax = axes[row, 0]
        sc = ax.scatter(
            data["query_indices"], data["match_indices"],
            c=data["n_correspondences_best"], cmap="viridis", s=12, alpha=0.7,
        )
        ax.set_xlabel("Query frame # (2025)")
        ax.set_ylabel("Best match frame # (2022)")
        ax.set_title(f"{name}\nSpearman rho = {data['spearman_corr']:.3f}")
        fig.colorbar(sc, ax=ax, label="# correspondences")

        # 2. Correspondences histogram
        ax = axes[row, 1]
        ax.hist(data["n_correspondences_best"], bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(100, color="red", linestyle="--", label="threshold=100")
        ax.set_xlabel("# correspondences (best match)")
        ax.set_ylabel("Count")
        ax.set_title(f"Hit rate: {data['hit_rate']:.1%}")
        ax.legend()

        # 3. Correspondences along transect
        ax = axes[row, 2]
        ax.plot(
            data["query_indices"], data["n_correspondences_best"],
            ".-", alpha=0.7, markersize=3,
        )
        ax.axhline(100, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Query frame # (2025)")
        ax.set_ylabel("# correspondences (best match)")
        ax.set_title("Match quality along transect")

    fig.suptitle("Retrieval Evaluation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("retrieval_evaluation.png", dpi=150, bbox_inches="tight")
    print("Saved retrieval_evaluation.png")

    plt_eval.show()
    return


@app.cell
def _(eval_pair1, eval_pair2, mo, np):
    _summary = f"""
    ## Retrieval Evaluation Summary

    | Metric | Pair 1 (forward) | Pair 2 (bw vs fw) |
    |--------|------------------|--------------------|
    | Hit rate (>=100 corr) | {eval_pair1['hit_rate']:.1%} | {eval_pair2['hit_rate']:.1%} |
    | Spearman rho | {eval_pair1['spearman_corr']:.3f} | {eval_pair2['spearman_corr']:.3f} |
    | Median correspondences | {np.median(eval_pair1['n_correspondences_best']):.0f} | {np.median(eval_pair2['n_correspondences_best']):.0f} |
    | Mean hits in top-5 | {np.mean(eval_pair1['n_hits_per_query']):.1f} | {np.mean(eval_pair2['n_hits_per_query']):.1f} |

    **Interpretation:**
    - **Hit rate** = fraction of 2025 query frames where at least 1 of top-5 retrieved frames has >= 100 correspondences
    - **Spearman rho** close to +1 = strong temporal coherence (early queries match early candidates)
    - Spearman rho close to 0 = no temporal structure (possibly different transect paths or poor matching)
    """
    mo.md(_summary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ad-hoc Exploration

    Single-query retrieval for interactive exploration.
    """)
    return


@app.cell
def _(frames_2022, frames_2025):
    query_frame = frames_2025[50]
    candidate_frames = frames_2022[:1600:10]
    return candidate_frames, query_frame


@app.cell
def _(
    build_retrieval_index,
    candidate_frames,
    device,
    find_best_matching_frames_twostage,
    model,
    query_frame,
    retrieve_candidates,
):
    print(f'Query frame: {query_frame.name}')
    print(f'Searching in {len(candidate_frames)} candidate frames (two-stage)...\n')

    # Build FAISS index for ad-hoc candidates
    _adhoc_index, _ = build_retrieval_index(candidate_frames, batch_size=32)

    top_matches, _shortlist = find_best_matching_frames_twostage(
        query_frame, candidate_frames, model,
        faiss_index=_adhoc_index, retrieve_fn=retrieve_candidates,
        device=device, retrieval_top_k=20, final_top_k=5,
        conf_threshold=1.0, batch_size=16,
    )
    print('\nTop 5 matching frames:')
    for (_rank, (_frame_path, _num_matches, _avg_conf)) in enumerate(top_matches, 1):
        print(f'{_rank}. {_frame_path.name}')
        print(f'   Matches: {_num_matches}, Avg confidence: {_avg_conf:.2f}')
    return (top_matches,)


@app.cell
def _(
    device,
    extract_correspondences_nonsym,
    inference,
    load_images,
    model,
    np,
    query_frame,
    top_matches,
    torch,
):
    import matplotlib.pyplot as plt

    _best_match_path = top_matches[0][0]
    images_2 = load_images([str(query_frame), str(_best_match_path)], size=512)
    _output = inference([tuple(images_2)], model, device, batch_size=1, verbose=False)
    (_view1, _pred1) = (_output['view1'], _output['pred1'])
    (_view2, _pred2) = (_output['view2'], _output['pred2'])
    (_desc1, _desc2) = (_pred1['desc'].squeeze(0).detach(), _pred2['desc'].squeeze(0).detach())
    (conf1, conf2) = (_pred1['desc_conf'].squeeze(0).detach(), _pred2['desc_conf'].squeeze(0).detach())
    _corres = extract_correspondences_nonsym(_desc1, _desc2, conf1, conf2, device=device, subsample=8, pixel_tol=0)
    _conf = _corres[2]
    _mask = _conf >= 1.5
    _matches_im0 = _corres[0][_mask].cpu().numpy()
    _matches_im1 = _corres[1][_mask].cpu().numpy()
    _image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    _image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    _viz_imgs = []
    for _view in [_view1, _view2]:
        _rgb_tensor = _view['img'] * _image_std + _image_mean
        _viz_imgs.append(_rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    (_H0, _W0, _H1, _W1) = (*_viz_imgs[0].shape[:2], *_viz_imgs[1].shape[:2])
    _img0 = np.pad(_viz_imgs[0], ((0, max(_H1 - _H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    _img1 = np.pad(_viz_imgs[1], ((0, max(_H0 - _H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    _img = np.concatenate((_img0, _img1), axis=1)
    _n_viz = 5
    if len(_matches_im0) > 100:
        _match_idx_to_viz = np.round(np.linspace(0, len(_matches_im0) - 1, _n_viz)).astype(int)
        (_viz_matches_im0, _viz_matches_im1) = (_matches_im0[_match_idx_to_viz], _matches_im1[_match_idx_to_viz])
        plt.figure(figsize=(20, 10))
        plt.imshow(_img)
        plt.title(f'Query: {query_frame.name} → Best Match: {_best_match_path.name}\n{len(_matches_im0)} correspondences (showing {_n_viz})', fontsize=12)
        _cmap = plt.get_cmap('jet')
        for _i in range(_n_viz):
            ((_x0, _y0), (_x1, _y1)) = (_viz_matches_im0[_i].T, _viz_matches_im1[_i].T)
            plt.plot([_x0, _x1 + _W0], [_y0, _y1], '-+', color=_cmap(_i / (_n_viz - 1)), scalex=False, scaley=False, linewidth=1.5, markersize=6)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('best_match_correspondences.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f'Saved to best_match_correspondences.png')
    else:
        print('No matches found!')
    return


if __name__ == "__main__":
    app.run()
