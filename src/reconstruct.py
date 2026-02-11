import argparse
import json
import os
from time import time

import h5py
import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

import segmentation
from segmentation.model import SegmentationModel
from sfm.estimator import DepthPoseEstimator
from sfm.inverse_warp import EUCMCamera, Pose, rectify_eucm
from reconstruction_utils import (
    get_closest_to_centroid_with_attributes_of_closest_to_cam,
    map_3d,
    get_matching_indices,
    get_edgeness,
    aggregate_2d_grid,
)
from video_utils import extract_frames_and_gopro_gravity_vector, render_video

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Start by parsing args
parser = argparse.ArgumentParser(description='Reconstruct a 3D model from a video')
parser.add_argument('--input_video', type=str, help='Path to video file - can be multiple files, in which case the paths should be comma separated')
parser.add_argument('--out_dir', type=str, default='out', help='Path to output directory - will be created if does not exist')
parser.add_argument('--tmp_dir', type=str, default='tmp', help='Path to temporary directory - will be created if does not exist')
parser.add_argument('--timestamp', type=str, help='Begin and End timestamp of the transect. In case multiple videos are supplied, the format should be comma separated e.g. of the form "0:23-end,begin-1:44"')
parser.add_argument('--sfm_checkpoint', type=str, default='../sfm_net.pth', help='Path to the sfm_net checkpoint')
parser.add_argument('--segmentation_checkpoint', type=str, default='../segmentation_net.pth', help='Path to the segmentation_net checkpoint')
parser.add_argument('--height', type=int, default=384, help='Height in pixels to which input video is scaled')
parser.add_argument('--width', type=int, default=640, help='Width in pixels to which input video is scaled')
parser.add_argument('--seg_height', type=int, default=384*2, help='Height in pixels to which input video is scaled')
parser.add_argument('--seg_width', type=int, default=640*2, help='Width in pixels to which input video is scaled')
parser.add_argument('--fps', type=int, default=8, help='FPS of the input video')
parser.add_argument('--reverse', action='store_true', help='Whether the transect video is filmed backwards (with a back-facing camera)')
parser.add_argument('--number_of_points_per_image', type=int, default=2000, help='Number of points to sample from each image')
parser.add_argument('--frames_per_volume', type=int, default=500, help='Number of frames per TSDF Volume')
parser.add_argument('--tsdf_overlap', type=int, default=100, help='Overlap in frames over TSDF Volumes')
parser.add_argument('--tsdf_voxel_size', type=float, default=None, help='TSDF voxel size in metres (auto-computed from depth range if not set). Use the same value for both models for fair comparison.')
parser.add_argument('--distance_thresh', type=float, default=0.2, help='Distance threshold for points added to cloud')
parser.add_argument('--ignore_classes_in_point_cloud', type=str, default="background,fish,human", help='Classes to ignore when adding points to cloud')
parser.add_argument('--ignore_classes_in_benthic_cover', type=str, default="background,fish,human,transect tools,transect line,dark", help='Classes to ignore when calculating benthic cover percentages')
parser.add_argument('--intrinsics_file', type=str, default="../example_inputs/intrinsics_eucm.json", help='Path to intrinsics file')
parser.add_argument('--class_to_label_file', type=str, default="../example_inputs/class_to_label.json", help='Path to label_to_class_file')
parser.add_argument('--class_to_color_file', type=str, default="../example_inputs/class_to_color.json", help='Path to class_to_color_file')
parser.add_argument('--output_2d_grid_size', type=int, default=2000, help='Size of the 2D grid used for benthic cover analysis - a higher grid size will produce higher resolution outputs but takes longer to compute and may have empty grid cells')
parser.add_argument('--buffer_size', type=int, default=2, help='Number of frames to use for temporal smoothing')
parser.add_argument('--render_video', action='store_true', help='Whether to render output 4-panel video')
# Depth/pose estimator selection
parser.add_argument('--depth_model', type=str, default='legacy', choices=['legacy', 'da3'],
                    help='Depth/pose estimation backend: "legacy" for original SfMModel, "da3" for Depth Anything 3')
parser.add_argument('--da3_model', type=str, default='depth-anything/DA3-LARGE-1.1',
                    help='HuggingFace model (DA3-SMALL/BASE/LARGE/GIANT)')
parser.add_argument('--da3_chunk_size', type=int, default=120,
                    help='Frames per chunk for DA3-Streaming')
parser.add_argument('--da3_overlap', type=int, default=60,
                    help='Overlap between chunks for alignment')
parser.add_argument('--da3_loop_enable', action='store_true', default=True,
                    help='Enable loop closure detection')
parser.add_argument('--da3_streaming_root', type=str, default=None,
                    help='Path to DA3-Streaming root (default: src/da3_streaming)')
parser.add_argument('--conf_threshold', type=float, default=0.4,
                    help='Minimum depth confidence (0.4 recommended for DA3)')
args = parser.parse_args()

def main(args):

    t = time()

    with open(args.class_to_color_file) as f:
        class_to_color = {k: (np.array(v)).astype(np.uint8) for k,v in json.load(f).items()}
    with open(args.class_to_label_file) as f:
        class_to_label = json.load(f)
        label_to_class = {v:k for k,v in class_to_label.items()}
    label_to_color = {k: class_to_color[v] for k,v in label_to_class.items()}
 

    grav = extract_frames_and_gopro_gravity_vector(
        args.input_video.split(","), 
        args.timestamp.split(","), 
        args.seg_width, 
        args.seg_height, 
        args.fps, 
        args.tmp_dir,
        args.reverse,
    )
    print("Extracted Frames And Gravity Vector in", time() - t, "seconds")
    

    h5f = h5py.File(args.tmp_dir + '/tmp.hdf5', 'w')
    
    img_list = [args.tmp_dir + "/rgb/" +file for file in sorted(os.listdir(args.tmp_dir + "/rgb")) if "jpg" in file]
    print("Running Neural Networks ...")
    
    depths, depth_uncertainties, depth_confidences, poses, semantic_segmentation, intrinsics, camera_type = get_nn_predictions(
        img_list,
        grav,
        len(class_to_label) + 1,
        h5f,
        args,
    )

    print("Ran NN Predictions in ", time() - t, "seconds")
    print("Building Point Cloud ...")
    os.makedirs(args.out_dir + "/videos", exist_ok=True)
    xyz_index_arr, distance2cam_arr, seg_arr, frame_index_arr, depth_unc_arr, keep_masks, dist_cutoffs = get_point_cloud(
        img_list,
        depths,
        poses,
        depth_uncertainties,
        semantic_segmentation,
        intrinsics,
        label_to_color,
        class_to_label,
        h5f,
        args,
        camera_type=camera_type,
        depth_confidences=depth_confidences,
    )

    print("Integrating TSDF!")
    tsdf_xyz, tsdf_rgb = tsdf_point_cloud(
        img_list,
        depths,
        keep_masks,
        poses,
        intrinsics,
        np.mean(depths),
        args.frames_per_volume,
        args.tsdf_overlap,
        dist_cutoffs,
        camera_type=camera_type,
        tsdf_voxel_size=args.tsdf_voxel_size,
    )
    print("Integrated TSDF Point Cloud in ", time() - t, "seconds")

    idx = get_matching_indices(tsdf_xyz, xyz_index_arr)
    print("Matched TSDF to Point Cloud in ", time() - t, "seconds")
    rgb_seg_arr = np.vectorize(lambda k: label_to_color[k], signature='()->(n)')(seg_arr[idx])
    tsdf_pc = pd.DataFrame({
        'x':tsdf_xyz[:,0],
        'y':tsdf_xyz[:,1],
        'z':tsdf_xyz[:,2],
        'r':tsdf_rgb[:,0],
        'g':tsdf_rgb[:,1],
        'b':tsdf_rgb[:,2],
        'distance_to_cam': distance2cam_arr[idx],
        'class': seg_arr[idx],
        'class_r': rgb_seg_arr[:,0],
        'class_g': rgb_seg_arr[:,1],
        'class_b': rgb_seg_arr[:,2], 
        'frame_index': frame_index_arr[idx],
        'depth_uncertainty': depth_unc_arr[idx],
    }) 
    tsdf_pc.to_csv(args.out_dir + "/point_cloud_tsdf.csv", index=False)

    # Save PLY point clouds (RGB-colored and semantic-colored)
    pc_rgb = o3d.geometry.PointCloud()
    pc_rgb.points = o3d.utility.Vector3dVector(tsdf_xyz)
    pc_rgb.colors = o3d.utility.Vector3dVector(tsdf_rgb.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(args.out_dir + "/point_cloud_rgb.ply", pc_rgb)

    pc_seg = o3d.geometry.PointCloud()
    pc_seg.points = o3d.utility.Vector3dVector(tsdf_xyz)
    pc_seg.colors = o3d.utility.Vector3dVector(rgb_seg_arr.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(args.out_dir + "/point_cloud_semantic.ply", pc_seg)

    print("Saved TSDF Point Cloud in ", time() - t, "seconds")


    print("Starting Benthic Cover Analsysis after ", time() - t, "seconds")
    results, percentage_covers = benthic_cover_analysis(tsdf_pc, label_to_class, args.ignore_classes_in_benthic_cover.split(","), bins=args.output_2d_grid_size)
    np.save(args.out_dir + "/results.npy", results)
    json.dump(percentage_covers, open(args.out_dir + "/percentage_covers.json", "w"))
    print("Finished Benthic Cover Analysis in ", time() - t, "seconds")

    os.system("cp "+args.class_to_color_file+" "+ args.out_dir)
    if args.render_video:
        os.system("cp "+args.tmp_dir+"/*_.mp4 "+ args.out_dir + "/videos")
        render_video(img_list, depths, semantic_segmentation, results, args.fps, class_to_label, label_to_color, args.tmp_dir, args.reverse, camera_type=camera_type, intrinsics_file=args.intrinsics_file)
        os.system("mv " + args.tmp_dir + "/out.mp4 " + args.out_dir + "/videos")
        print("Rendered Video in ", time() - t, "seconds")
    return 


def expand_zeros(mask):
    # Add an extra batch dimension and channel dimension to the mask for convolution
    mask = mask.unsqueeze(0).unsqueeze(0).float()  # Shape: 1x1xHxW

    # Define a 3x3 kernel filled with ones
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device)

    # Perform 2D convolution with padding=1 to keep the same output size
    conv_result = torch.nn.functional.conv2d(mask, kernel, padding=1)

    # Any place where the convolution result is less than 9 means it had a zero in the neighborhood
    result_mask = (conv_result == 9).squeeze().bool()

    return result_mask


def create_estimator(args) -> DepthPoseEstimator:
    """Instantiate the selected depth/pose estimation backend."""
    if args.depth_model == "legacy":
        from sfm.legacy_estimator import LegacySfMEstimator
        estimator = LegacySfMEstimator()
    elif args.depth_model == "da3":
        from sfm.da3_estimator import DA3Estimator
        estimator = DA3Estimator()
    else:
        raise ValueError(f"Unknown depth model: {args.depth_model}")
    estimator.setup(str(device), args)
    return estimator


def get_nn_predictions(img_list, grav, num_classes, h5f, args):
    """Run depth/pose estimation and semantic segmentation on extracted frames.

    Depth/pose prediction is delegated to the selected estimator backend.
    Segmentation still runs inline (it is independent of the depth model).
    """
    totensor = torchvision.transforms.ToTensor()
    normalize = torchvision.transforms.Normalize(
        mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]
    )

    # --- Depth / pose estimation via the selected backend ---
    estimator = create_estimator(args)
    est_output = estimator.predict(img_list, grav, args.height, args.width, args)

    depths = est_output.depths                  # [N, H, W] float32
    depth_uncertainties = est_output.depth_uncertainties  # [N, H, W] or None
    depth_confidences = est_output.depth_confidences      # [N, H, W] or None
    poses = est_output.poses                    # [N, 4, 4] float32
    intrinsics_predicted = est_output.intrinsics  # [N, K] float32
    camera_type = est_output.camera_type

    # Log depth statistics so users can calibrate --distance_thresh
    depth_median = float(np.median(depths[depths > 0]))
    depth_q95 = float(np.quantile(depths[depths > 0], 0.95))
    print(f"Depth stats: median={depth_median:.3f}m, 95th-pct={depth_q95:.3f}m")

    # Auto-adjust distance_thresh to the 90th percentile of the actual depth
    # distribution when the user hasn't explicitly set it.  This ensures both
    # legacy (relative depth) and DA3 (metric depth) keep the same fraction
    # of their depth range -- important for fair comparison.
    if args.distance_thresh == 0.2:
        old_thresh = args.distance_thresh
        args.distance_thresh = float(np.quantile(depths[depths > 0], 0.90))
        print(
            f"Auto-adjusted --distance_thresh from {old_thresh} to "
            f"{args.distance_thresh:.3f} (90th-pct of depth) to match depth scale"
        )

    # Store depths/uncertainties in HDF5 for downstream consumers
    h5f.create_dataset("depths", data=depths)
    if depth_uncertainties is None:
        depth_uncertainties = np.zeros_like(depths)
    h5f.create_dataset("depth_uncertainties", data=depth_uncertainties)
    h5f.create_dataset("intrinsics", data=intrinsics_predicted)

    # --- Semantic segmentation (unchanged -- runs independently) ---
    segmentation_model = SegmentationModel(num_classes).to(device)
    segmentation_model.load_state_dict(
        torch.load(args.segmentation_checkpoint, map_location=device)
    )
    segmentation_model.eval()

    semantic_segmentation = h5f.create_dataset(
        "semantic_segmentation", (len(img_list), args.height, args.width), dtype="u1"
    )

    buffer_size = args.buffer_size
    semseg_buffer = torch.zeros(
        (3, num_classes, args.height, args.width), requires_grad=False
    ).to(device)
    wtens = (
        torch.tensor([1.0, 2.0, 1.0], requires_grad=False)
        .to(device)
        .unsqueeze(1)
        .unsqueeze(1)
        .unsqueeze(1)
    )

    # Seed segmentation buffer
    images = [
        normalize(totensor(Image.open(img_list[i]))).to(device).unsqueeze(0)
        for i in range(buffer_size - 1)
    ]
    with torch.no_grad():
        semseg_logits = []
        for i in range(buffer_size - 1):
            semseg_logits.append(
                segmentation.model.predict(
                    segmentation_model, images[i], num_classes, args.height, args.width
                )
            )
        for i in range(buffer_size - 2):
            semantic_segmentation[i] = (
                torch.stack(semseg_logits[max(0, i - 1): i + 1])
                .mean(dim=0)
                .argmax(dim=0)
                .cpu()
                .numpy()
            )
        if len(semseg_logits) == 1:
            semseg_logits.append(semseg_logits[-1])
        semseg_buffer[0] = semseg_logits[-2]
        semseg_buffer[1] = semseg_logits[-1]
        del semseg_logits

    # Sliding window for segmentation
    with torch.no_grad():
        for end_index in tqdm(range(buffer_size - 1, len(img_list)), desc="Seg"):
            new_im = normalize(totensor(Image.open(img_list[end_index]))).to(device).unsqueeze(0)
            semseg_buffer[2] = segmentation.model.predict(
                segmentation_model, new_im, num_classes, args.height, args.width
            )
            semantic_segmentation[end_index - 1] = (
                (semseg_buffer * wtens).mean(dim=0).argmax(dim=0).cpu().numpy()
            )
            semseg_buffer[:2] = semseg_buffer[1:].clone()

    semantic_segmentation[-1] = (
        (semseg_buffer * wtens).mean(dim=0).argmax(dim=0).cpu().numpy()
    )

    return depths, depth_uncertainties, depth_confidences, poses, semantic_segmentation, intrinsics_predicted, camera_type


def _unproject_depth_pinhole(depth_map, fx, fy, cx, cy):
    """Unproject a depth map to 3D points using a pinhole camera model.

    Parameters
    ----------
    depth_map : np.ndarray [H, W]
    fx, fy, cx, cy : float

    Returns
    -------
    coords : torch.Tensor [3, H*W] on CPU
    """
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return torch.from_numpy(np.stack([x, y, z], axis=0).reshape(3, -1))


def get_point_cloud(image_list, depths, poses, depth_uncertainties, semantic_segmentation, intrinsics, label_to_color, class_to_label, h5f, args, camera_type="eucm", depth_confidences=None):
    ignore_classes = args.ignore_classes_in_point_cloud.split(",")

    def class_to_color(class_arr):
        color_arr = np.zeros((class_arr.shape[0], 3), dtype=np.uint8)
        for val in np.unique(class_arr):
            color_arr[class_arr==val] = label_to_color[val]
        return color_arr

    # Scale the edge detection threshold relative to the depth range so it
    # has the same *relative* effect for both legacy (small relative depths)
    # and DA3 (metric depths).  Reference: 0.04 was tuned for legacy with
    # median depth ~0.15.
    _reference_depth = 0.15
    _depth_median = float(np.median(depths[depths > 0]))
    edgeness_thresh = 0.04 * (_depth_median / _reference_depth)
    print(f"Edgeness threshold: {edgeness_thresh:.4f} (depth_median={_depth_median:.3f})")

    dist_cutoffs = []
    with torch.no_grad():

        xyz_arr = h5f.create_dataset("xyz_arr", (len(image_list)*args.number_of_points_per_image, 3), dtype='f4')
        distance2cam_arr = h5f.create_dataset("distance2cam_arr", (len(image_list)*args.number_of_points_per_image), dtype='f4')
        seg_arr = h5f.create_dataset("seg_arr", (len(image_list)*args.number_of_points_per_image), dtype='u1')
        keep_masks = h5f.create_dataset("keep_masks", (len(image_list),  args.height,  args.width), dtype='u1')
        depth_unc_arr = h5f.create_dataset("depth_unc_arr", (len(image_list)*args.number_of_points_per_image), dtype='f4')
        frame_index_arr = h5f.create_dataset("frame_index_arr", (len(image_list)*args.number_of_points_per_image), dtype='u2')

        cursor = 0

        for i in tqdm(range(len(poses))):

            pose = torch.tensor((poses[i])[:3]).float().to(device)
            depth_i_tensor = torch.tensor(depths[i]).to(device)

            # --- Unproject depth to 3D using the appropriate camera model ---
            if camera_type == "eucm":
                cam = EUCMCamera(torch.tensor(intrinsics[i]).unsqueeze(0).to(device), Tcw=Pose(T=1))
                coords = cam.reconstruct_depth_map(depth_i_tensor.unsqueeze(0).unsqueeze(0).to(device)).squeeze()
                coords = coords.reshape(3, -1)
            else:  # pinhole
                fx, fy, cx, cy = intrinsics[i][:4]
                coords = _unproject_depth_pinhole(depths[i], fx, fy, cx, cy).to(device)

            # Transform to world coordinates
            coords = (pose @ torch.cat([coords, torch.ones_like(coords[:1])], dim=0).reshape(4, -1)).T.cpu()

            dist_cutoffs.append(args.distance_thresh)
            keep_mask = depth_i_tensor.squeeze() < args.distance_thresh
            seg = torch.tensor(semantic_segmentation[i]).to(device)

            # Filter by depth quality: confidence (DA3) or uncertainty (legacy).
            # Both models get equivalent quality filtering so comparison is fair.
            if depth_confidences is not None and args.conf_threshold > 0:
                conf_i = torch.tensor(depth_confidences[i]).to(device)
                keep_mask = torch.logical_and(keep_mask, conf_i > args.conf_threshold)
            elif depth_uncertainties is not None:
                unc_i = torch.tensor(depth_uncertainties[i]).to(device)
                # Reject pixels with uncertainty > 20% of the local depth value
                depth_i_safe = depth_i_tensor.squeeze().clamp(min=1e-6)
                relative_unc = unc_i / depth_i_safe
                keep_mask = torch.logical_and(keep_mask, relative_unc < 0.2)

            # Exclude points on the 'edge' of objects (high depth gradient)
            keep_mask = torch.logical_and(get_edgeness(depth_i_tensor) < edgeness_thresh, keep_mask)

            # Mask out fisheye border region (only for legacy EUCM, not DA3 pinhole)
            if camera_type == "eucm":
                keep_mask[30:170, 30:-30] = 0

            for class_name in ignore_classes:
                keep_mask = torch.logical_and(seg != class_to_label[class_name], keep_mask)
            keep_mask = expand_zeros(keep_mask)
            keep_mask = keep_mask.cpu().numpy()
            keep_masks[i] = keep_mask.astype(np.uint8)
            keep_mask = keep_mask.reshape(-1)
            valid_points = keep_mask.sum().item()
            random_selection = np.random.permutation(valid_points)[:args.number_of_points_per_image]
            offset = min(valid_points, args.number_of_points_per_image)

            xyz_arr[cursor:cursor+offset]=coords[keep_mask][random_selection]
            distance2cam_arr[cursor:cursor+offset]=  depths[i].reshape(-1)[keep_mask][random_selection]

            seg_arr[cursor:cursor+offset] = semantic_segmentation[i].reshape(-1).astype(np.uint8)[keep_mask][random_selection]
            dunc = depth_uncertainties[i].reshape(-1)[keep_mask][random_selection]
            depth_unc_arr[cursor:cursor+offset]=dunc
            frame_index_arr[cursor:cursor+offset]= np.zeros_like(dunc, dtype=np.uint16)+i

            cursor += offset


        if cursor == 0:
            raise RuntimeError(
                f"No valid points survived filtering (distance_thresh={args.distance_thresh:.3f}m). "
                f"This usually means --distance_thresh is too small for the depth model's output scale. "
                f"Try increasing it, e.g. --distance_thresh=3.0"
            )

        print(f"Filtering redundant points ({cursor} points before dedup)")
        xyz_index_arr = map_3d(np.concatenate([
            xyz_arr[:cursor],
            distance2cam_arr[:cursor].reshape(-1,1),
            np.arange(len(xyz_arr)).reshape(-1, 1)[:cursor]], axis=1), get_closest_to_centroid_with_attributes_of_closest_to_cam, 0.003)
        filtered_indices = xyz_index_arr[:, -1].astype(np.uint32)

        return xyz_index_arr[:,:3], distance2cam_arr[:cursor][filtered_indices], seg_arr[:cursor][filtered_indices], frame_index_arr[:cursor][filtered_indices], depth_unc_arr[:cursor][filtered_indices], keep_masks, dist_cutoffs
        
    
def tsdf_point_cloud(img_list, depths, masks, poses, intrinsics, cutoff, frames_per_volume, tsdf_overlap, dist_cutoffs, camera_type="eucm", tsdf_voxel_size=None):
    """Integrate frames into a TSDF volume and extract a point cloud.

    When *camera_type* is ``"eucm"``, each frame is rectified from fisheye to
    pinhole before integration (original behaviour).  When ``"pinhole"``, the
    frames are already rectified so we skip that step and use them directly.
    """
    xyz = []
    rgb = []

    # Scale voxel size based on depth range to avoid OOM with metric depth
    # Legacy model: ~0.6mm voxels for ~0.2m depth range
    # DA3 model: scale proportionally (e.g., ~3mm voxels for ~1m depth)
    if tsdf_voxel_size is not None:
        voxel_length = tsdf_voxel_size
    else:
        base_voxel = 0.3 / 512.0  # ~0.0006m
        depth_scale = max(1.0, cutoff / 0.2)  # Scale relative to legacy 0.2m
        voxel_length = base_voxel * min(depth_scale, 10.0)  # Cap at 10x
    # SDF truncation distance: controls the "blur radius" around each surface.
    # Legacy (EUCM): use original absolute truncation (hand-tuned, ~58x voxel).
    # DA3 (pinhole): use standard ratio (~6x voxel) per Open3D/TSDF best practice.
    if camera_type == "eucm":
        sdf_trunc = 0.035 * max(1.0, cutoff / 0.2)
    else:
        sdf_trunc = voxel_length * 6.0
    print(f"TSDF voxel_length={voxel_length:.4f}m, sdf_trunc={sdf_trunc:.4f}m")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    totensor = torchvision.transforms.ToTensor()

    mask_out_background = np.ones_like(masks[0].astype(np.float32))
    intrinsics_t = torch.tensor(intrinsics).float()

    # Mask out fisheye border region (only for legacy EUCM, not DA3 pinhole)
    if camera_type == "eucm":
        mask_out_background[:170, 80:-80] *= 0
    for i in tqdm(range(len(poses))):

        if i > len(poses)-10:
            mask_out_background = np.ones_like(masks[0])

        mask_i = masks[i].astype(np.float32) * mask_out_background

        if camera_type == "eucm":
            # Rectify fisheye frame to pinhole for TSDF integration
            projected_img, projected_mask, projected_depth = rectify_eucm(
                totensor(Image.open(img_list[i])).unsqueeze(0),
                torch.tensor(mask_i).unsqueeze(0).unsqueeze(0).float(),
                torch.tensor(depths[i]).unsqueeze(0).unsqueeze(0),
                intrinsics_t[i]
            )
            frame_fx = float(intrinsics_t[i][0])
            frame_fy = float(intrinsics_t[i][1])
            frame_cx = float(intrinsics_t[i][2])
            frame_cy = float(intrinsics_t[i][3])
        else:
            # Pinhole: frames are already rectified, use directly.
            # Resize image to match depth map dimensions (height x width).
            img_pil = Image.open(img_list[i]).resize(
                (args.width, args.height), Image.BILINEAR
            )
            img_np = np.array(img_pil, dtype=np.uint8)  # [H, W, 3]
            projected_img_hwc = img_np
            projected_mask = mask_i.astype(np.float32)
            projected_depth = np.ascontiguousarray(
                depths[i].astype(np.float32) * projected_mask
            )
            frame_fx = float(intrinsics[i][0])
            frame_fy = float(intrinsics[i][1])
            frame_cx = float(intrinsics[i][2])
            frame_cy = float(intrinsics[i][3])

            depth_img = o3d.geometry.Image(projected_depth)
            color_img = o3d.geometry.Image(np.ascontiguousarray(projected_img_hwc))

        if camera_type == "eucm":
            # EUCM path outputs [3,H,W] float arrays, convert here
            masked_depth = np.ascontiguousarray(
                (projected_depth * projected_mask).astype(np.float32)
            )
            depth_img = o3d.geometry.Image(masked_depth)
            color_img = o3d.geometry.Image(
                np.ascontiguousarray(projected_img.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img, depth_trunc=dist_cutoffs[i], convert_rgb_to_intensity=False, depth_scale=1)

        cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=args.width,
            height=args.height,
            fx=frame_fx,
            fy=frame_fy,
            cx=frame_cx,
            cy=frame_cy,
        )

        # Skip frames with singular pose matrices
        try:
            pose_inv = np.linalg.inv(poses[i])
        except np.linalg.LinAlgError:
            print(f"Warning: Skipping frame {i} due to singular pose matrix")
            continue

        volume.integrate(rgbd, cam_intrinsic, pose_inv)

        if (i % frames_per_volume) == (frames_per_volume - tsdf_overlap):
            volume2 = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel_length,
                sdf_trunc=sdf_trunc * 0.5,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

        if i % frames_per_volume >= (frames_per_volume - tsdf_overlap):
            volume2.integrate(rgbd, cam_intrinsic, pose_inv)

        if (i % frames_per_volume) == frames_per_volume - 1:
            pc = volume.extract_point_cloud()
            pc = volume.extract_point_cloud()
            xyz.append(np.array(pc.points))
            rgb.append((np.array(pc.colors)*255).astype(np.uint8))
            volume = volume2

    pc = volume.extract_point_cloud()
    pc = volume.extract_point_cloud()
    xyz.append(np.array(pc.points))
    rgb.append((np.array(pc.colors)*255).astype(np.uint8))
    return np.concatenate(xyz), np.concatenate(rgb)
 

def benthic_cover_analysis(pc, label_to_class, ignore_classes_in_benthic_cover, bins=1000):
    #step 1: fit PCA
    pca = PCA(n_components=2)
    pca.fit(pc[['x','y', 'z']].values)  
    x_axis = pca.components_[0]  # Estimated x-axis
    y_axis = pca.components_[1]  # Estimated y-axis

    # Step 2: Calculate the normal vector to the x-y plane as the z-axis
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # Step 3: Create the transformation matrix
    transformation_matrix = np.vstack((x_axis, y_axis, z_axis)).T

    # Now, you can apply this transformation matrix to your point cloud
    transformed = np.dot(pc[['x','y', 'z']].values, transformation_matrix)  
    transformed -= np.min(transformed, axis=0)
    xmax, ymax, zmax = np.max(transformed, axis=0)
    
    discretization = xmax / bins
    pcarr = np.concatenate([transformed, pc.drop(columns=["x", "y", "z"]).values], axis=1)
    out = aggregate_2d_grid(pcarr, size=discretization)

    xcoords = out[:,0].astype(np.int32)
    ycoords = out[:,1].astype(np.int32)

    img = np.zeros((xcoords.max()+1, ycoords.max()+1, 12))
    
    img[xcoords, ycoords] = out[:,2:]
    
    percentage_covers = {}
    benthic_class = out[:,7].astype(np.uint8)
    all_classes = (benthic_class!=0)
    for class_label, class_name in label_to_class.items():
        if class_name not in ignore_classes_in_benthic_cover:
            percentage_covers[class_name] = (benthic_class==class_label).sum() 
        else:
            all_classes = np.logical_and(all_classes, benthic_class!=class_label)
    all_classes = all_classes.sum()
    percentage_covers = {k: v / all_classes for k,v in percentage_covers.items()}
    return img, percentage_covers

if __name__ == "__main__":
    main(args)