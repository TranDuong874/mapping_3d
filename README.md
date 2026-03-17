# mapping_3d

Local ORB-SLAM3 Python bindings, a modernized vendored ORB-SLAM3 fork, a simple stereo-inertial dataset runner, and a dense monocular-inertial mapping runner with async depth fusion.

## Repo Layout

- `dependency/ORB_SLAM3/`: vendored ORB-SLAM3 fork used by the binding
- `pyorbslam3/`: Python package entrypoint
- `orbslam3_bindings.cpp`: pybind11 bridge
- `run_orbslam3_tumvi.py`: end-to-end TUM-VI style runner
- `run_orbslam3_da3_realtime.py`: dense monocular-inertial runner with UniDepthV2 or DA3 depth backends
- `examples/`: minimal binding examples
- `docs/`: build and API documentation

## Quick Start

Build the vendored ORB-SLAM3 and the Python extension:

```bash
cd mapping_3d
MAKE_JOBS=2 bash ./build_orbslam3_python.sh
```

Run the import smoke test:

```bash
cd mapping_3d
python3 test_orbslam3_import.py
```

Run the TUM-VI style stereo-inertial example:

```bash
cd mapping_3d
python3 run_orbslam3_tumvi.py \
  --dataset-root ../dataset/dataset-corridor1_512_16 \
  --output-dir outputs/corridor1_smoke \
  --max-frames 300
```

Outputs:

- `tracking_log.csv`
- `trajectory_tum.txt`
- `camera_path.xyz`
- `sparse_points.xyz`

Run the dense monocular-inertial runner:

```bash
cd mapping_3d
python3 run_orbslam3_da3_realtime.py \
  --dataset-root dataset/dataset-corridor1_512_16
```

Dense runner outputs include trajectory, sparse points, live occupancy exports, and a final fused occupancy map under `outputs/orbslam3_da3_realtime/`.

## Documentation

- `docs/BUILD.md`: prerequisites, build flow, rebuild notes
- `docs/API.md`: binding API summary and usage notes
- `docs/RUN_ORBSLAM3_DENSE.md`: dense runner usage, backends, outputs, and CLI option reference
- `examples/basic_tumvi_stereo_inertial.py`: minimal binding example over a TUM-VI style dataset

## Notes

- The build defaults to `MAKE_JOBS=2` to avoid memory spikes.
- The binding is ROS-free and intended for direct local Python use.
- `run_orbslam3_tumvi.py` is the simple stereo-inertial runner.
- `run_orbslam3_da3_realtime.py` is the dense monocular-inertial runner.
