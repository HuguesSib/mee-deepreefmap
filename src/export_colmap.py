"""
Export DeepReefMap reconstruction to COLMAP sparse format for visualization in COLMAP GUI.

Usage:
    python3 export_colmap.py --out_dir /path/to/reconstruction_output

Output structure (inside out_dir/colmap/):
    sparse/0/cameras.txt
    sparse/0/images.txt
    sparse/0/points3D.txt
"""

import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation


def export_colmap(out_dir: str):
    poses = np.load(os.path.join(out_dir, "poses.npy"))        # (N, 4, 4) cam-to-world
    intrinsics = np.load(os.path.join(out_dir, "intrinsics.npy"))  # (N, 6) [fx,fy,cx,cy,alpha,beta]

    colmap_dir = os.path.join(out_dir, "colmap", "sparse", "0")
    os.makedirs(colmap_dir, exist_ok=True)

    N = len(poses)
    width, height = 1280, 768

    # Use median intrinsics (EUCM alpha/beta are dropped; COLMAP PINHOLE uses fx,fy,cx,cy)
    fx, fy, cx, cy = np.median(intrinsics[:, :4], axis=0)

    # --- cameras.txt (single shared camera) ---
    cameras_path = os.path.join(colmap_dir, "cameras.txt")
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    # --- images.txt (one entry per frame) ---
    images_path = os.path.join(colmap_dir, "images.txt")
    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {N}\n")

        for i in range(N):
            T_c2w = poses[i]           # 4x4 camera-to-world
            T_w2c = np.linalg.inv(T_c2w)
            R_w2c = T_w2c[:3, :3]
            t_w2c = T_w2c[:3, 3]

            # scipy Rotation: scalar-last (x,y,z,w) → COLMAP wants scalar-first (w,x,y,z)
            rot = Rotation.from_matrix(R_w2c)
            qx, qy, qz, qw = rot.as_quat()

            f.write(f"{i+1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{t_w2c[0]:.9f} {t_w2c[1]:.9f} {t_w2c[2]:.9f} "
                    f"1 {i}.jpg\n")
            f.write("\n")  # empty POINTS2D line

    # --- points3D.txt (empty — load PLY separately in GUI) ---
    points3d_path = os.path.join(colmap_dir, "points3D.txt")
    with open(points3d_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")

    print(f"Exported COLMAP sparse model to: {colmap_dir}")
    print(f"  cameras.txt : 1 PINHOLE camera  (fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f})")
    print(f"  images.txt  : {N} images")
    print(f"  points3D.txt: empty (load PLY separately in COLMAP GUI)")
    print()
    print("To open in COLMAP GUI:")
    print(f"  1. File → New project → set image path to: {os.path.join(out_dir, 'frames')}")
    print(f"  2. File → Import model → select: {colmap_dir}")
    print(f"  3. To load point cloud: File → Import PLY point cloud")
    print(f"     → {os.path.join(out_dir, 'point_cloud_rgb.ply')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, help="Path to reconstruction output directory")
    args = parser.parse_args()
    export_colmap(args.out_dir)
