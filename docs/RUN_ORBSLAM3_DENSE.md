# `run_orbslam3_da3_realtime.py`

Dense monocular-inertial runner for TUM-VI style data.

Despite the file name, the script now supports two depth backends:

- `unidepth_v2` (default)
- `da3`

The pipeline is:

1. ORB-SLAM3 tracks `mav0/cam0` with `mav0/imu0`.
2. Tracked frames are queued to a separate depth backend process.
3. Depth is scaled with ORB sparse 2D-3D correspondences.
4. Depth is fused into either:
   - `tsdf_voxel` occupancy output (default)
   - `raycast` occupancy output
5. GUI windows can show the ORB input, depth preview, and live occupancy viewer.

## Dataset Layout

The script expects a TUM-VI style root:

```text
<dataset-root>/
  mav0/
    cam0/
      data.csv
      data/*.png
    imu0/
      data.csv
```

It uses only `cam0` plus IMU. This is an egocentric monocular-inertial run, not stereo.

## Quick Start

Default run: UniDepthV2 + TSDF voxelization

```bash
python3 run_orbslam3_da3_realtime.py \
  --dataset-root dataset/dataset-corridor1_512_16
```

DA3 fallback:

```bash
python3 run_orbslam3_da3_realtime.py \
  --dataset-root dataset/dataset-corridor1_512_16 \
  --depth-backend da3 \
  --depth-model depth-anything/DA3-SMALL
```

Raycast occupancy instead of TSDF:

```bash
python3 run_orbslam3_da3_realtime.py \
  --dataset-root dataset/dataset-corridor1_512_16 \
  --fusion-mode raycast
```

Faster UniDepth run:

```bash
python3 run_orbslam3_da3_realtime.py \
  --dataset-root dataset/dataset-corridor1_512_16 \
  --depth-backend unidepth_v2 \
  --unidepth-resolution-level 0 \
  --conf-percentile 85 \
  --queue-size 2
```

## Main Defaults

- Depth backend: `unidepth_v2`
- Depth model:
  - `unidepth_v2`: `lpiccinelli/unidepth-v2-vits14`
  - `da3`: `depth-anything/DA3-SMALL`
- Fusion mode: `tsdf_voxel`
- Confidence percentile: `85`
- Voxel size: `0.05 m`
- TSDF truncation band: `0.08 m`
- Camera clearance radius: `0.30 m`
- Camera clearance below camera: `0.18 m`
- Queue size: `2`

## Outputs

In the main output directory:

- `tracking_log.csv`: per-frame ORB tracking state
- `trajectory_tum.txt`: camera trajectory in TUM format
- `camera_path.xyz`: camera positions
- `sparse_points.xyz`: ORB sparse map points
- `live_status.txt`: status used by the live viewer
- `summary.json`: final run summary

In `<output-dir>/depth/`:

- `dense_frames.csv`: per-frame depth/fusion log
- `backend_status.txt`: depth worker status
- `occupancy_live.xyz`: live occupancy export for the viewer
- `occupancy_map.ply`: final occupancy point cloud
- `occupancy_map.npz`: final occupancy metadata
- `tsdf_mesh.ply`: only written when `--fusion-mode tsdf_voxel`

## Important Options

### Input and Runtime

- `--dataset-root`: dataset root containing `mav0/cam0` and `mav0/imu0`
- `--vocabulary-path`: ORB vocabulary file
- `--settings-path`: ORB camera/IMU settings file
- `--output-dir`: output directory
- `--max-frames`: stop early for smoke tests
- `--progress-every`: progress print interval
- `--no-realtime`: run as fast as possible instead of dataset pacing
- `--no-clahe`: disable grayscale CLAHE before ORB tracking
- `--device`: usually `cuda` or `cpu`

### GUI Toggles

- `--no-orb-viewer`: disable Pangolin ORB viewer
- `--no-input-window`: disable the input preview window
- `--no-depth-window`: disable the depth preview window
- `--no-map-window`: disable the live occupancy viewer

### Depth Backend

- `--depth-backend {unidepth_v2,da3}`: select the depth model family
- `--depth-model`: override the backend model name/path

UniDepth-specific:

- `--unidepth-src`: path to the vendored UniDepth source tree
- `--unidepth-resolution-level`: lower is faster; valid range is `0..9`

DA3-specific:

- `--da3-pose-conditioned`: enable rolling-window pose conditioning
- `--no-da3-pose-conditioned`: explicitly disable it
- `--da3-context-keyframes`: DA3 conditioning window size

Notes:

- DA3 pose conditioning is only used when `--depth-backend da3`.
- Some DA3 checkpoints do not support pose conditioning. The runner falls back to single-frame inference automatically.

### Fusion

- `--fusion-mode {tsdf_voxel,raycast}`

`tsdf_voxel`:

- integrates filtered depth into TSDF
- extracts a surface
- voxelizes the extracted surface for occupancy-style output

`raycast`:

- integrates occupancy directly with ray marching
- supports behind-wall rejection through occupied-voxel penetration tolerance

Relevant knobs:

- `--voxel-size`: occupancy voxel size in meters
- `--tsdf-sdf-trunc-m`: TSDF truncation band, only for `tsdf_voxel`
- `--penetration-tolerance-voxels`: only for `raycast`
- `--camera-clearance-radius-m`: carve occupied voxels around the camera path to remove floating points
- `--camera-clearance-below-m`: how far below the camera the carve reaches; keep small to preserve the floor
- `--max-map-voxels`: cap on retained occupancy voxels

## Depth Filtering

The script filters depth before fusion.

- `--min-depth-m`, `--max-depth-m`: accepted depth range
- `--conf-percentile`: keep only pixels at or above this model confidence percentile
- `--sky-threshold`: reject sky pixels when the backend provides a sky map
- `--min-scale-correspondences`: minimum ORB sparse correspondences required to estimate scale

Higher `--conf-percentile` is stricter. `85` to `90` is a reasonable range when you want cleaner occupancy.

## Throughput Knobs

These usually matter most for realtime behavior:

- `--queue-size`: keep small; `1` or `2` is usually right
- `--process-res`: DA3 preprocessing resolution
- `--reprojection-stride`: larger values reduce fusion load
- `--live-export-interval-sec`: reduce live viewer update frequency
- `--live-point-budget`: cap live viewer density
- `--final-point-budget`: cap final export density

## Recommended Setups

Fast default:

```bash
python3 run_orbslam3_da3_realtime.py \
  --dataset-root dataset/dataset-corridor1_512_16 \
  --depth-backend unidepth_v2 \
  --fusion-mode tsdf_voxel \
  --conf-percentile 85 \
  --queue-size 2
```

Cleaner TSDF surface:

```bash
python3 run_orbslam3_da3_realtime.py \
  --dataset-root dataset/dataset-corridor1_512_16 \
  --fusion-mode tsdf_voxel \
  --voxel-size 0.05 \
  --tsdf-sdf-trunc-m 0.06 \
  --conf-percentile 90
```

More conservative raycast occupancy:

```bash
python3 run_orbslam3_da3_realtime.py \
  --dataset-root dataset/dataset-corridor1_512_16 \
  --fusion-mode raycast \
  --penetration-tolerance-voxels 1 \
  --conf-percentile 90
```

Aggressive floating-point cleanup:

```bash
python3 run_orbslam3_da3_realtime.py \
  --dataset-root dataset/dataset-corridor1_512_16 \
  --fusion-mode tsdf_voxel \
  --camera-clearance-radius-m 0.35 \
  --camera-clearance-below-m 0.12
```

## Dependencies

For ORB-SLAM3, build the local binding first:

```bash
MAKE_JOBS=2 bash ./build_orbslam3_python.sh
```

For UniDepth backend, the current environment must be able to import the vendored UniDepth tree pointed to by `--unidepth-src`.

For DA3 backend, the current environment must be able to import `depth_anything_3`.

## Known Behavior

- The file name still says `da3` for historical reasons, but the runner is now backend-agnostic.
- The live occupancy viewer infers the vertical axis automatically for the top-down heatmap.
- The depth window shows the filtered depth that is actually used for fusion.
- Occupancy around the camera path is carved by default to suppress floating artifacts, with limited downward carving to avoid deleting the floor.
