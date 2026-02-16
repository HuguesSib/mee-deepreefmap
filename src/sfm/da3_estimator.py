"""Depth Anything 3 Streaming estimator.

Uses DA3_Streaming for globally-aligned depth and pose estimation
with chunked processing, loop closure detection, and SIM3 alignment.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import subprocess
from typing import Literal

import numpy as np
from PIL import Image
from tqdm import tqdm

from sfm.estimator import DepthPoseEstimator, EstimatorOutput, accumulate_poses_with_gravity


class DA3Estimator(DepthPoseEstimator):
    """Depth/pose backend using DA3-Streaming with global alignment."""

    @property
    def camera_type(self) -> Literal["pinhole"]:
        return "pinhole"

    def setup(self, device: str, args) -> None:
        self.device = device
        self.chunk_size = getattr(args, "da3_chunk_size", 120)
        self.overlap = getattr(args, "da3_overlap", 60)
        self.loop_enable = getattr(args, "da3_loop_enable", True)

        # Path to da3_streaming configs (git submodule in external/)
        streaming_root = getattr(args, "da3_streaming_root", None)
        if streaming_root is None:
            streaming_root = Path(__file__).parent.parent.parent / "Depth-Anything-3" / "da3_streaming"
        self.da3_streaming_root = Path(streaming_root)

    def predict(
        self,
        img_list: list[str],
        gravity: np.ndarray | None,
        height: int,
        width: int,
        args,
    ) -> EstimatorOutput:
        """Run DA3-Streaming and return globally-aligned depth/poses."""

        # Create temporary directory for frames and output.
        # Clean any stale data from previous runs to avoid mixing results.
        tmp_dir = Path(args.tmp_dir) / "da3_streaming"
        frames_dir = tmp_dir / "frames"
        output_dir = tmp_dir / "output"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resize and copy frames to frames_dir (DA3_Streaming expects a directory)
        print(f"Preparing {len(img_list)} frames for DA3-Streaming...")
        for i, img_path in enumerate(tqdm(img_list, desc="Linking frames")):
            dst = frames_dir / f"{i:07d}.jpg"
            img = Image.open(img_path).resize((width, height), Image.BILINEAR)
            img.save(dst, quality=95)

        # Run DA3-Streaming
        self._run_streaming(frames_dir, output_dir)

        # Load results
        depths, poses, intrinsics, confs = self._load_streaming_output(
            output_dir, len(img_list), height, width
        )

        # Apply gravity-based rotation correction to prevent drift ("banana"
        # artifact on long sequences).  Decompose the absolute C2W poses from
        # DA3-Streaming into relative transforms, then re-accumulate with
        # per-frame gravity alignment from the GoPro IMU.
        if gravity is not None:
            poses = self._apply_gravity_correction(poses, gravity)

        return EstimatorOutput(
            depths=depths,
            poses=poses,
            intrinsics=intrinsics,
            camera_type="pinhole",
            depth_confidences=confs,
        )

    @staticmethod
    def _apply_gravity_correction(
        poses: np.ndarray, gravity: np.ndarray
    ) -> np.ndarray:
        """Re-accumulate DA3 absolute poses with gravity alignment.

        DA3-Streaming outputs absolute C2W poses from visual odometry.  For
        long linear transects the small rotation errors accumulate into a
        curved ("banana") trajectory.  This method decomposes the absolute
        poses into relative frame-to-frame transforms, then re-accumulates
        them with per-frame gravity correction from the GoPro IMU -- the same
        approach used by the legacy estimator.
        """
        n = len(poses)
        # Decompose absolute C2W poses into relative transforms
        relative_poses = np.zeros((n - 1, 4, 4), dtype=np.float64)
        for i in range(n - 1):
            relative_poses[i] = np.linalg.inv(poses[i]) @ poses[i + 1]

        # Re-accumulate with gravity correction
        corrected = accumulate_poses_with_gravity(relative_poses, gravity)
        return corrected.astype(np.float32)

    def _run_streaming(self, frames_dir: Path, output_dir: Path) -> None:
        """Run DA3_Streaming on the frames directory."""
        import sys

        # DA3-Streaming has no __init__.py and uses bare imports (loop_utils,
        # fastloop, etc.) so it expects its own directory on sys.path.  Add it
        # directly and use bare imports to match the upstream design.
        streaming_dir = str(self.da3_streaming_root)
        if streaming_dir not in sys.path:
            sys.path.insert(0, streaming_dir)

        from da3_streaming import DA3_Streaming  # noqa: E402
        from loop_utils.config_utils import load_config  # noqa: E402

        # Load base config (in src/configs/base_config.yaml)
        config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
        config = load_config(str(config_path))

        # Check if the weights exist
        if not Path(config["Weights"]["DA3"]).exists():
            # Run the download_weights.sh script
            download_weights_script = Path(__file__).parent.parent.parent.parent / "scripts" / "download_weights.sh"
            subprocess.run(["bash", download_weights_script])

        # Override chunk settings
        config["Model"]["chunk_size"] = self.chunk_size
        config["Model"]["overlap"] = self.overlap
        config["Model"]["loop_enable"] = self.loop_enable
        config["Model"]["save_depth_conf_result"] = True
        config["Model"]["delete_temp_files"] = True

        # Use torch for alignment (more compatible)
        config["Model"]["align_lib"] = "torch"

        print(f"  chunk_size={self.chunk_size}, overlap={self.overlap}, loop_enable={self.loop_enable}")

        streaming = DA3_Streaming(str(frames_dir), str(output_dir), config)
        streaming.run()
        streaming.close()

    def _load_streaming_output(
        self,
        output_dir: Path,
        n_frames: int,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load DA3-Streaming output and convert to EstimatorOutput format."""

        # Load camera poses (C2W format)
        poses_path = output_dir / "camera_poses.txt"
        c2w_matrices = self._load_camera_poses(poses_path)

        # Load intrinsics
        intrinsics_path = output_dir / "intrinsic.txt"
        intrinsics_4 = self._load_intrinsics(intrinsics_path)

        # Load per-frame depth and confidence
        results_dir = output_dir / "results_output"
        depths = []
        confs = []
        scale_x, scale_y = 1.0, 1.0

        for i in range(n_frames):
            frame_path = results_dir / f"frame_{i}.npz"
            if not frame_path.exists():
                raise FileNotFoundError(f"Missing frame: {frame_path}")

            data = np.load(frame_path)
            depth = data["depth"]
            conf = data["conf"]

            # Compute scale factors on first frame
            if i == 0 and (depth.shape[0] != height or depth.shape[1] != width):
                scale_y = height / depth.shape[0]
                scale_x = width / depth.shape[1]

            # Resize if needed
            if depth.shape[0] != height or depth.shape[1] != width:
                depth = self._resize_array(depth, height, width)
                conf = self._resize_array(conf, height, width)

            depths.append(depth)
            confs.append(conf)

        depths = np.stack(depths).astype(np.float32)
        confs = np.stack(confs).astype(np.float32)
        poses = c2w_matrices.astype(np.float32)

        # Scale intrinsics if depth maps were resized: [fx, fy, cx, cy]
        intrinsics = intrinsics_4.astype(np.float32)
        if scale_x != 1.0 or scale_y != 1.0:
            intrinsics[:, 0] *= scale_x  # fx
            intrinsics[:, 1] *= scale_y  # fy
            intrinsics[:, 2] *= scale_x  # cx
            intrinsics[:, 3] *= scale_y  # cy

        return depths, poses, intrinsics, confs

    @staticmethod
    def _load_camera_poses(poses_path: Path) -> np.ndarray:
        """Load C2W camera poses from text file."""
        poses = []
        with open(poses_path) as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                if len(values) != 16:
                    raise ValueError(f"Expected 16 values per line, got {len(values)}")
                pose = np.array(values).reshape(4, 4)
                poses.append(pose)
        return np.array(poses, dtype=np.float32)

    @staticmethod
    def _load_intrinsics(intrinsics_path: Path) -> np.ndarray:
        """Load intrinsics [fx, fy, cx, cy] from text file."""
        intrinsics = []
        with open(intrinsics_path) as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                if len(values) != 4:
                    raise ValueError(f"Expected 4 values (fx fy cx cy), got {len(values)}")
                intrinsics.append(values)
        return np.array(intrinsics, dtype=np.float32)

    @staticmethod
    def _resize_array(arr: np.ndarray, height: int, width: int) -> np.ndarray:
        """Resize 2D array using bilinear interpolation."""
        import torch
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        t = torch.nn.functional.interpolate(
            t, size=(height, width), mode="bilinear", align_corners=False
        )
        return t.squeeze().numpy()
