# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MEE-DeepReefMap is a system for scalable 3D semantic mapping of coral reefs using deep learning. It processes GoPro Hero 10 videos to create 3D reconstructions with semantic segmentation. Based on the paper "Scalable 3D Semantic Mapping of Coral Reefs using Deep Learning" (arXiv:2309.12804).

## Build & Run Commands

```bash
# Install dependencies (requires uv package manager)
uv sync

# Note: gpmfstream must be installed separately first
git clone https://github.com/hovren/gpmfstream.git
cd gpmfstream && git submodule update --init && python3 setup.py install

# For DA3 depth model: install DA3-Streaming (external dependency)
git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3/da3_streaming
pip install -r requirements.txt
bash scripts/download_weights.sh
cd ../..
# Alternatively, symlink into the project:
#   ln -s /path/to/Depth-Anything-3/da3_streaming src/da3_streaming

# Run reconstruction with legacy depth model (original)
cd src
python3 reconstruct.py --input_video=/path/to/video.mp4 --timestamp=0-100 --out_dir=./output

# Run reconstruction with DA3-Streaming (recommended for better depth)
# Point --da3_streaming_root to your DA3-Streaming clone
python3 reconstruct.py --input_video=/path/to/video.mp4 --timestamp=0-100 --out_dir=./output \
    --depth_model=da3 --da3_model=depth-anything/DA3-LARGE-1.1 \
    --da3_streaming_root=/path/to/Depth-Anything-3/da3_streaming
# Or if you symlinked into src/da3_streaming, the default path works:
python3 reconstruct.py --input_video=/path/to/video.mp4 --timestamp=0-100 --out_dir=./output \
    --depth_model=da3 --da3_model=depth-anything/DA3-LARGE-1.1

# Train SfM network
cd src/sfm
python3 train_sfm.py --data /path/to/kitti_data --name "experiment-name"

# Train segmentation network
cd src/segmentation
python3 train_segmentation_model.py --data /path/to/seg_data --test_splits scene1,scene2

# Docker build and run
docker build -t deepreefmap .
docker run -v ./input:/input -v ./output:/output deepreefmap \
    --input_video=/input/video.mp4 --timestamp=0-100 --out_dir=/output
```

## Architecture

### Main Entry Point
`src/reconstruct.py` - Orchestrates the full 3D reconstruction pipeline:
1. Frame extraction from GoPro video with gravity vector extraction (GPMF metadata)
2. Neural network predictions (depth/pose/segmentation)
3. Point cloud generation from depth maps
4. TSDF volume integration for 3D mesh creation
5. Benthic cover analysis (2D grid statistics)

### Core Modules

**SfM Module** (`src/sfm/`):
- `model.py`: SfMModel - DeepLabV3Plus encoder with ResNext50 backbone for depth and pose estimation
- `legacy_estimator.py`: Wrapper for original SfMModel (--depth_model=legacy)
- `da3_estimator.py`: DA3-Streaming integration (--depth_model=da3)
  - Chunked processing with overlap for memory efficiency
  - Loop closure detection using SALAD
  - Global SIM3 alignment for consistent poses

**DA3 Streaming** (external dependency: [DA3-Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3)):
- Depth Anything 3 streaming pipeline (not included in repo — install separately)
- Chunked video processing with loop closure via SALAD image retrieval
- Pass location via `--da3_streaming_root` or symlink to `src/da3_streaming/`

**Segmentation Module** (`src/segmentation/`):
- `model.py`: SegmentationModel - DeepLabV3Plus with ResNext50_32x4d for ~40 coral reef classes
- `train_segmentation_model.py`: Training with pixel/IoU/polygon metrics, wandb integration

**Utilities**:
- `reconstruction_utils.py`: 3D processing (voxel hashing, point cloud deduplication, gravity alignment)
- `video_utils.py`: Video I/O via FFmpeg, GPMF metadata parsing, 4-panel rendering

### Key Technical Details
- Camera model: Enhanced Unified Camera Model with fisheye distortion parameters
- TSDF voxel size: 0.3/512 = ~0.0006m per voxel
- Overlapping TSDF volumes for large-scale reconstruction
- 2-frame temporal buffer for depth/segmentation smoothing
- Gravity vectors from accelerometer used for pose orientation correction

### Configuration Files
- `example_inputs/intrinsics_eucm.json`: Camera intrinsics (fx, fy, cx, cy, alpha, beta)
- `example_inputs/class_to_label.json`: 40+ semantic class definitions
- `example_inputs/class_to_color.json`: RGB colors for visualization

### Model Checkpoints (Git LFS)
- `sfm_net.pth`: Pre-trained depth/pose network (~105MB)
- `segmentation_net.pth`: Pre-trained segmentation network (~128MB)

## Key Dependencies
- Python 3.10-3.11
- PyTorch 2.0.1 with CUDA 11.8
- segmentation-models-pytorch, open3d
- wandb for experiment tracking
- gpmfstream (external) for GoPro metadata
- DA3-Streaming (external, optional) for Depth Anything 3 depth/pose estimation
