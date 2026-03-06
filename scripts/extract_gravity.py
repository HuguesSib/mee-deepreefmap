#!/usr/bin/env python3
"""Extract GoPro gravity vectors and save as gravity.npy.

Standalone utility so you don't need to re-run the full reconstruction
pipeline just to get gravity data for an existing output directory.

Usage:
    python extract_gravity.py --video /path/to/gopro.mp4 --timestamp begin-end \
        --out_dir /path/to/output_dir

    # The output directory should contain frames/ and poses.npy so we can
    # match the number of gravity vectors to the number of frames.
"""

import argparse
import os
import sys

import numpy as np

# Add src/ to path for video_utils imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from video_utils import get_gravity_vectors, get_video_length


def main():
    parser = argparse.ArgumentParser(description="Extract GoPro gravity vectors")
    parser.add_argument("--video", required=True, help="Path to GoPro MP4 file")
    parser.add_argument(
        "--timestamp",
        default="begin-end",
        help="Timestamp range, e.g. '30-180' or 'begin-end' (default: begin-end)",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory (should contain frames/ and poses.npy)",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=None,
        help="Number of frames. If not provided, inferred from poses.npy or frames/ dir.",
    )
    args = parser.parse_args()

    # Determine number of frames
    n_frames = args.n_frames
    if n_frames is None:
        poses_path = os.path.join(args.out_dir, "poses.npy")
        frames_dir = os.path.join(args.out_dir, "frames")
        if os.path.exists(poses_path):
            n_frames = len(np.load(poses_path))
            print(f"Inferred {n_frames} frames from poses.npy")
        elif os.path.exists(frames_dir):
            n_frames = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
            print(f"Inferred {n_frames} frames from frames/ directory")
        else:
            print("ERROR: Cannot determine frame count. Provide --n_frames or ensure "
                  "out_dir contains poses.npy or frames/")
            sys.exit(1)

    gravity = get_gravity_vectors(args.video, args.timestamp, n_frames)
    if gravity is None:
        print("ERROR: Could not extract gravity vectors. Is this an unedited GoPro video?")
        sys.exit(1)

    out_path = os.path.join(args.out_dir, "gravity.npy")
    np.save(out_path, gravity)
    print(f"Saved gravity vectors {gravity.shape} to {out_path}")


if __name__ == "__main__":
    main()
