"""Legacy SfM estimator wrapping the original DeepLabV3Plus-based SfMModel.

This backend preserves the exact inference logic from the original
``get_nn_predictions()`` in ``reconstruct.py``, including the sliding-window
buffer, temporal depth averaging, and pose accumulation with gravity correction.
"""

from __future__ import annotations

import json
from typing import Literal

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm

from reconstruction_utils import get_rotation_matrix_to_align_pose_with_gravity
from sfm.estimator import DepthPoseEstimator, EstimatorOutput
from sfm.inverse_warp import pose_vec2mat
from sfm.model import SfMModel


# ---------------------------------------------------------------------------
# Helpers (moved from reconstruct.py to keep them close to the only user)
# ---------------------------------------------------------------------------

def _reset_batchnorm_layers(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eps = 1e-4


def _change_bn_momentum(model: nn.Module, value: float) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = value


# ---------------------------------------------------------------------------
# Gravity-based pose correction
# ---------------------------------------------------------------------------

def _accumulate_poses_with_gravity(
    relative_poses: np.ndarray,
    gravity: np.ndarray | None,
    grav_buffer: int = 100,
) -> np.ndarray:
    """Convert relative poses to cumulative absolute poses with gravity alignment.

    Parameters
    ----------
    relative_poses : np.ndarray
        ``[N-1, 4, 4]`` relative frame-to-frame transforms.
    gravity : np.ndarray | None
        ``[N, 3]`` per-frame gravity vectors, or ``None``.
    grav_buffer : int
        Smoothing window for gravity vectors.

    Returns
    -------
    np.ndarray
        ``[N, 4, 4]`` cumulative camera-to-world poses.
    """
    n_relative = len(relative_poses)

    if gravity is not None:
        pose0 = np.eye(4)
        cum_poses = np.zeros((n_relative + 1, 4, 4))

        grav0 = np.mean(gravity[:grav_buffer], axis=0)
        correction = get_rotation_matrix_to_align_pose_with_gravity(pose0, grav0)
        pose0[:3, :3] = correction @ pose0[:3, :3]
        cum_poses[0] = pose0.copy()

        for i, (rel, _g) in enumerate(zip(relative_poses, gravity[1:])):
            g = np.mean(
                gravity[max(0, 1 + i - grav_buffer): min(i + grav_buffer, len(gravity) - 1)],
                axis=0,
            )
            pose0 = pose0 @ rel
            correction = get_rotation_matrix_to_align_pose_with_gravity(pose0, g)
            pose0[:3, :3] = correction @ pose0[:3, :3]
            cum_poses[i + 1] = pose0.copy()
    else:
        cum_poses = np.zeros((n_relative + 1, 4, 4))
        cum_poses[0] = np.eye(4)
        for i in range(n_relative):
            cum_poses[i + 1] = cum_poses[i] @ relative_poses[i]

    return cum_poses


# ---------------------------------------------------------------------------
# LegacySfMEstimator
# ---------------------------------------------------------------------------

class LegacySfMEstimator(DepthPoseEstimator):
    """Depth/pose backend using the original DeepLabV3Plus + pose-decoder SfMModel.

    Faithfully reproduces the inference pipeline that was previously inlined
    inside ``get_nn_predictions()`` in ``reconstruct.py``.
    """

    @property
    def camera_type(self) -> Literal["eucm"]:
        return "eucm"

    def setup(self, device: str, args) -> None:
        self.device = torch.device(device)
        self.model = SfMModel().to(self.device)
        self.model.load_state_dict(
            torch.load(args.sfm_checkpoint, map_location=self.device)
        )
        _change_bn_momentum(self.model, 0.01)
        _reset_batchnorm_layers(self.model)
        self.model.eval()

        self.intrinsics = (
            torch.tensor(list(json.load(open(args.intrinsics_file)).values()))
            .float()
            .to(self.device)
            .unsqueeze(0)
        )

        self._totensor = torchvision.transforms.ToTensor()
        self._normalize = torchvision.transforms.Normalize(
            mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        img_list: list[str],
        gravity: np.ndarray | None,
        height: int,
        width: int,
        args,
    ) -> EstimatorOutput:
        buffer_size = args.buffer_size

        depths, depth_uncertainties, intrinsics_out, raw_poses = self._run_sliding_window(
            img_list, height, width, buffer_size,
        )

        # --- Convert raw pairwise poses to cumulative absolute poses ---
        relative_poses = self._relative_poses_from_raw(raw_poses)
        poses = _accumulate_poses_with_gravity(relative_poses, gravity)

        return EstimatorOutput(
            depths=depths,
            poses=poses.astype(np.float32),
            intrinsics=intrinsics_out,
            camera_type="eucm",
            depth_uncertainties=depth_uncertainties,
        )

    # ------------------------------------------------------------------
    # Internal: sliding-window depth/pose inference
    # ------------------------------------------------------------------

    def _run_sliding_window(
        self,
        img_list: list[str],
        height: int,
        width: int,
        buffer_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run the buffered sliding-window inference loop.

        Returns
        -------
        depths : np.ndarray [N, H, W]
        depth_uncertainties : np.ndarray [N, H, W]
        intrinsics_out : np.ndarray [N, 6]
        raw_poses : dict
            Nested dict of pairwise pose predictions for later aggregation.
        """
        device = self.device
        n_frames = len(img_list)

        # --- Temp HDF5 buffers for memory efficiency ---
        h5path = f"/tmp/legacy_sfm_{id(self)}.hdf5"
        h5f = h5py.File(h5path, "w")
        depths_buf = h5f.create_dataset(
            "depths_buf", (buffer_size, n_frames, height, width), dtype="f4"
        )
        intrinsics_buf = h5f.create_dataset(
            "intrinsics_buf", (buffer_size, n_frames, 6), dtype="f4"
        )

        counts = np.zeros(n_frames, dtype=np.uint8)
        raw_poses: dict = {}

        # Seed the buffer with the first (buffer_size - 1) frames
        images = [
            self._load_image(img_list[i], device) for i in range(buffer_size - 1)
        ]
        images = [F.resize(x, (height, width)) for x in images]
        depth_features = [
            [f.detach() for f in self.model.extract_features(x)] for x in images
        ]

        # Sliding window
        with torch.no_grad():
            for end_index in tqdm(range(buffer_size - 1, n_frames), desc="SfM"):
                new_im = self._load_image(img_list[end_index], device)
                new_im = F.resize(new_im, (height, width))
                images.append(new_im)
                depth_features.append(
                    [f.detach() for f in self.model.extract_features(images[-1])]
                )

                depth, pose, intrinsics_upd = (
                    self.model.get_depth_and_poses_from_features(
                        images, depth_features, self.intrinsics
                    )
                )

                for i in range(buffer_size):
                    idx = end_index - buffer_size + i + 1
                    c = counts[idx]
                    depths_buf[c, idx] = depth[i].squeeze().detach().cpu().numpy()
                    intrinsics_buf[c, idx] = intrinsics_upd[i].detach().cpu().numpy()
                    counts[idx] += 1

                    for j in range(buffer_size):
                        jdx = end_index - buffer_size + j + 1
                        if pose[i][j] != []:
                            raw_poses.setdefault(idx, {}).setdefault(jdx, []).append(
                                pose[i][j].detach().unsqueeze(0).cpu().numpy()
                            )

                images.pop(0)
                depth_features.pop(0)

        # --- Aggregate buffered predictions ---
        depths = np.empty((n_frames, height, width), dtype=np.float32)
        depth_unc = np.empty((n_frames, height, width), dtype=np.float32)
        intrinsics_out = np.empty((n_frames, 6), dtype=np.float32)

        for i in range(n_frames):
            c = counts[i]
            depths[i] = np.mean(depths_buf[:c, i], axis=0)
            depth_unc[i] = np.std(depths_buf[:c, i], axis=0)
            intrinsics_out[i] = np.median(intrinsics_buf[:c, i], axis=0)

        h5f.close()
        return depths, depth_unc, intrinsics_out, raw_poses

    # ------------------------------------------------------------------
    # Internal: pose helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _relative_poses_from_raw(raw_poses: dict) -> np.ndarray:
        """Convert raw pairwise pose predictions to relative 4x4 transforms.

        Mirrors the original logic: bidirectional median, drift removal.
        """
        n = len(raw_poses) - 1
        rel = []
        for i in range(n):
            fwd = np.median(raw_poses[i + 1][i], axis=0)
            bwd = np.median(raw_poses[i][i + 1], axis=0)
            p_vec = torch.tensor((fwd - bwd) / 2)
            mat = pose_vec2mat(p_vec).squeeze().cpu().numpy()
            rel.append(np.vstack([mat, [0, 0, 0, 1]]))

        rel = np.array(rel)
        # Remove median rotation drift
        med_rot = np.median(rel[:, :3, :3] - np.eye(3), axis=0)
        rel[:, :3, :3] -= med_rot
        return rel

    # ------------------------------------------------------------------
    # Internal: image loading
    # ------------------------------------------------------------------

    def _load_image(self, path: str, device: torch.device) -> torch.Tensor:
        """Load, normalize, and move a single frame to device. Returns ``[1, 3, H, W]``."""
        return self._normalize(self._totensor(Image.open(path))).to(device).unsqueeze(0)
