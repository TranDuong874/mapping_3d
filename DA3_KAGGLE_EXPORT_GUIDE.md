# DA3 Export & Inference Guide

This guide explains how to use the 2-step pipeline to process fisheye datasets using ORB-SLAM3 locally and then running Depth Anything 3 (DA3) pose-conditioned inference on Kaggle.

---

## Step 1: Local Data Export
**Script:** `run_orbslam3_da3_export.py`

This script runs ORB-SLAM3 on your local machine. It uses the native fisheye model for stable tracking but exports undistorted PINHOLE images and camera parameters required for the DA3 pipeline.

### Usage
```bash
python3 run_orbslam3_da3_export.py \
    --dataset-root ../dataset/dataset-corridor4_512_16 \
    --output-dir outputs/my_experiment \
    --viewer
```

### Key Parameters
| Parameter | Description |
| :--- | :--- |
| `--dataset-root` | Path to the dataset containing `mav0/cam0` and `mav0/imu0`. |
| `--output-dir` | Root directory for the export. The bundle will be saved in a `da3/` subfolder here. |
| `--viewer` | Enable the live Pangolin SLAM viewer. |
| `--native-tracking` | (Default: True) Runs SLAM on fisheye images for maximum stability. Set `--no-native-tracking` to run on pinhole instead. |
| `--undistort-balance` | (Default: -1.0) Controls the FoV of undistorted images. `-1.0` preserves the central focal length (best for detail). `0.0` to `1.0` follows OpenCV logic. |
| `--max-frames` | Limit the run to the first N frames. |

### Output Bundle
The script creates a `da3/` folder inside your output directory containing:
- `data/`: Undistorted PNG frames.
- `keyframes.csv`: Mapping of timestamps to image paths.
- `extrinsics.npy`: World-to-Camera poses for every tracked frame.
- `intrinsics.npy`: Pinhole intrinsics for every frame.
- `manifest.json`: Metadata for the DA3 pipeline.

---

## Step 2: Kaggle Inference
**Script:** `da3_conditioned.py`

Upload the `da3/` folder (or its parent) to Kaggle as a Dataset. This script will perform dense 3D reconstruction using the camera poses exported in Step 1.

### Usage
On Kaggle, if you have attached your dataset, you can simply run:
```bash
python3 da3_conditioned.py
```

### Key Parameters
| Parameter | Description |
| :--- | :--- |
| `--orb-output-dir` | Path to the bundle from Step 1. Auto-detected on Kaggle if not provided. |
| `--stride` | (Default: 1) Temporal sampling. Set to `2` to process every 2nd frame (2x faster). |
| `--max-frames` | Process only the first N frames from the bundle. |
| `--chunk-size` | (Default: 8) Number of frames sent to the GPU at once for pose-conditioning. |
| `--model-name` | HF model path (e.g., `depth-anything/DA3-SMALL` or `DA3-LARGE`). |
| `--process-res` | (Default: 512) Resolution for the depth model. |
| `--pixel-stride` | (Default: 2) Density of the output point cloud. `1` is densest, higher is lighter. |
| `--export-format` | (Default: `mini_npz-glb`) Formats to save. Options: `npz`, `mini_npz`, `glb`. |

### Running on Kaggle Tips
1.  **Auto-detection**: The script automatically looks in `/kaggle/input` for the `manifest.json` file.
2.  **Output**: Results are saved to `/kaggle/working/da3_results` by default.
3.  **Speed**: Use `--stride 2` or `--stride 4` if you have a very long sequence and only need a representative 3D map.
4.  **GPU**: Ensure the Kaggle "Accelerator" is set to **GPU P100** or **T4 x2**.

---

## Workflow Summary
1.  Run `run_orbslam3_da3_export.py` locally.
2.  Zip the `da3/` folder and upload to Kaggle as a private dataset.
3.  Run `da3_conditioned.py` in a Kaggle Notebook with GPU enabled.
4.  Download the `dense_point_cloud.ply` or `results.glb` from the Kaggle output.
