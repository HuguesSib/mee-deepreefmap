"""Abstract interface for depth and pose estimation backends.

This module defines the contract that all depth/pose estimators must follow,
allowing the reconstruction pipeline to swap between different backends
(e.g., the legacy SfMModel, Depth Anything 3) without changing downstream code.

It also contains shared helpers (e.g. gravity correction) used by multiple
estimator backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from reconstruction_utils import get_rotation_matrix_to_align_pose_with_gravity


@dataclass
class EstimatorOutput:
    """Standardized output from any depth/pose estimation backend.

    All backends must produce data in this format so that the reconstruction
    pipeline can consume it uniformly.
    """

    depths: np.ndarray
    """Per-frame depth maps in meters. Shape: ``[N, H, W]``, dtype float32."""

    poses: np.ndarray
    """Cumulative camera-to-world transforms. Shape: ``[N, 4, 4]``, dtype float32."""

    intrinsics: np.ndarray
    """Per-frame camera intrinsics.
    Shape: ``[N, 6]`` for EUCM (fx, fy, cx, cy, alpha, beta)
    or ``[N, 4]`` for pinhole (fx, fy, cx, cy). dtype float32.
    """

    camera_type: Literal["eucm", "pinhole"]
    """Which camera model the intrinsics correspond to."""

    depth_uncertainties: np.ndarray | None = None
    """Optional per-pixel depth uncertainty. Shape: ``[N, H, W]``, dtype float32."""

    depth_confidences: np.ndarray | None = None
    """Optional per-pixel depth confidence. Shape: ``[N, H, W]``, dtype float32."""


class DepthPoseEstimator(ABC):
    """Base class for all depth/pose estimation backends.

    Subclasses implement :meth:`setup` to load model weights and
    :meth:`predict` to run inference on a sequence of frames.
    """

    @property
    @abstractmethod
    def camera_type(self) -> Literal["eucm", "pinhole"]:
        """Camera model produced by this estimator."""
        ...

    @abstractmethod
    def setup(self, device: str, args) -> None:
        """Load model weights and prepare for inference.

        Parameters
        ----------
        device : str
            Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
        args : argparse.Namespace
            CLI arguments (estimator may read backend-specific flags).
        """
        ...

    @abstractmethod
    def predict(
        self,
        img_list: list[str],
        gravity: np.ndarray | None,
        height: int,
        width: int,
        args,
    ) -> EstimatorOutput:
        """Run depth and pose estimation on a frame sequence.

        Parameters
        ----------
        img_list : list[str]
            Ordered paths to extracted video frames.
        gravity : np.ndarray | None
            Per-frame gravity vectors ``[N, 3]`` from GoPro GPMF metadata,
            or ``None`` if unavailable.
        height : int
            Target output height in pixels.
        width : int
            Target output width in pixels.
        args : argparse.Namespace
            CLI arguments (for buffer_size, checkpoint paths, etc.).

        Returns
        -------
        EstimatorOutput
            Depths, poses, intrinsics, and optional confidence/uncertainty.
        """
        ...


def accumulate_poses_with_gravity(
    relative_poses: np.ndarray,
    gravity: np.ndarray | None,
    grav_buffer: int = 100,
) -> np.ndarray:
    """Convert relative poses to cumulative absolute poses with gravity alignment.

    At each frame, the rotation component of the accumulated pose is corrected
    so that the camera's "up" direction matches the gravity vector measured by
    the GoPro's IMU.  This prevents systematic rotation drift from compounding
    over long sequences (the "banana" artifact).

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
