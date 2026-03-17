# Python API

## Package

```python
import pyorbslam3
```

Exports:

- `pyorbslam3.System`
- `pyorbslam3.Sensor`
- `pyorbslam3.ImuMeasurement`

## Sensor Enum

Available sensors:

- `Sensor.MONOCULAR`
- `Sensor.STEREO`
- `Sensor.RGBD`
- `Sensor.IMU_MONOCULAR`
- `Sensor.IMU_STEREO`
- `Sensor.IMU_RGBD`

## System Construction

```python
slam = pyorbslam3.System(
    vocabulary_path,
    settings_path,
    pyorbslam3.Sensor.IMU_STEREO,
    use_viewer=False,
)
```

Arguments:

- `vocabulary_path`: path to `ORBvoc.txt`
- `settings_path`: path to the ORB-SLAM3 YAML config
- `sensor`: one of the `Sensor` enum values
- `use_viewer`: enable Pangolin viewer
- `init_frame`: optional initial frame id
- `sequence`: optional sequence name

## IMU Samples

```python
imu = pyorbslam3.ImuMeasurement(
    timestamp_s,
    (ax, ay, az),
    (gx, gy, gz),
)
```

The constructor takes acceleration first and angular velocity second.

## Tracking Calls

Monocular:

```python
result = slam.track_monocular(image_u8, timestamp_s, imu_measurements)
```

Stereo:

```python
result = slam.track_stereo(left_u8, right_u8, timestamp_s, imu_measurements)
```

RGB-D:

```python
result = slam.track_rgbd(image_u8, depth_f32, timestamp_s, imu_measurements)
```

Image requirements:

- grayscale `uint8` with shape `(H, W)`, or
- color `uint8` with shape `(H, W, 3)`

Depth requirements:

- `float32` or `float64`
- shape `(H, W)`

## Tracking Result

Each `track_*` call returns a dictionary with:

- `tracking_state`: integer ORB-SLAM3 tracking state
- `tracking_state_name`: readable state string
- `pose_valid`: `True` when the returned pose is finite
- `pose_matrix`: `4x4 float32` camera-world transform matrix or `None`
- `translation_xyz`: `(x, y, z)` tuple or `None`
- `quaternion_wxyz`: `(w, x, y, z)` tuple or `None`

The raw pose returned by ORB-SLAM3 is camera pose. The TUM-VI runner converts it to world-camera form before writing trajectory output.

## State and Map Access

Available methods:

- `slam.get_tracking_state()`
- `slam.get_tracking_state_name()`
- `slam.get_current_map_points()`
- `slam.get_tracked_keypoints()`
- `slam.get_tracked_observations()`
- `slam.get_image_scale()`
- `slam.reset()`
- `slam.reset_active_map()`
- `slam.shutdown()`
- `slam.is_shutdown()`

`get_current_map_points()` returns an `Nx3 float32` array of sparse world points from the current map.

`get_tracked_observations()` returns a dictionary with:

- `keypoints_uv`: `Nx2 float32` current-frame keypoint coordinates for valid tracked map points
- `world_points_xyz`: `Nx3 float32` corresponding sparse map points in world coordinates

## Minimal Example

See `examples/basic_tumvi_stereo_inertial.py` for a short example and `run_orbslam3_tumvi.py` for the full trajectory and sparse point export flow.
