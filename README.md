# mapping_3d

Local ORB-SLAM3 Python bindings, a modernized vendored ORB-SLAM3 fork, and a simple stereo-inertial dataset runner.

## Repo Layout

- `dependency/ORB_SLAM3/`: vendored ORB-SLAM3 fork used by the binding
- `pyorbslam3/`: Python package entrypoint
- `orbslam3_bindings.cpp`: pybind11 bridge
- `run_orbslam3_tumvi.py`: end-to-end TUM-VI style runner
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

## Documentation

- `docs/BUILD.md`: prerequisites, build flow, rebuild notes
- `docs/API.md`: binding API summary and usage notes
- `examples/basic_tumvi_stereo_inertial.py`: minimal binding example over a TUM-VI style dataset

## Notes

- The build defaults to `MAKE_JOBS=2` to avoid memory spikes.
- The binding is ROS-free and intended for direct local Python use.
- The runner currently targets stereo-inertial TUM-VI style datasets and exports trajectory plus sparse map points.
