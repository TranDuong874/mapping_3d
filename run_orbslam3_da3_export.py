#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# The pip cv2 wheel may inject Qt plugin env vars that collide with ORB-SLAM3's viewer thread.
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_FONTDIR", None)

import pyorbslam3


@dataclass(frozen=True)
class FrameRecord:
    timestamp_ns: int
    image_path: Path


@dataclass(frozen=True)
class ImuRecord:
    timestamp_ns: int
    gyro_xyz: tuple[float, float, float]
    accel_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class FisheyeCameraModel:
    intrinsics: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    prefix: str


class FisheyeUndistorter:
    def __init__(self, camera_model: FisheyeCameraModel, balance: float) -> None:
        width, height = camera_model.image_size
        identity = np.eye(3, dtype=np.float64)
        
        # If balance is negative, we manually preserve the focal length to avoid extreme stretching.
        if balance < 0:
            new_intrinsics = camera_model.intrinsics.copy().astype(np.float64)
            new_intrinsics[0, 2] = (width - 1) / 2.0
            new_intrinsics[1, 2] = (height - 1) / 2.0
        else:
            new_intrinsics = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                camera_model.intrinsics.astype(np.float64),
                camera_model.distortion.astype(np.float64),
                (width, height),
                identity,
                balance=balance,
                new_size=(width, height),
            )
            
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_model.intrinsics.astype(np.float64),
            camera_model.distortion.astype(np.float64),
            identity,
            new_intrinsics,
            (width, height),
            cv2.CV_32FC1,
        )
        self.input_intrinsics = camera_model.intrinsics.astype(np.float32)
        self.output_intrinsics = new_intrinsics.astype(np.float32)
        self.distortion = camera_model.distortion.astype(np.float32)
        self.map1 = map1
        self.map2 = map2

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)


def load_fisheye_camera_model(settings_path: Path) -> FisheyeCameraModel:
    storage = cv2.FileStorage(str(settings_path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Failed to open settings file: {settings_path}")
    try:
        node = storage.getNode("Camera1.fx")
        prefix = "Camera1"
        if node.empty():
            node = storage.getNode("Camera.fx")
            prefix = "Camera"
        
        if node.empty():
             raise ValueError(f"Could not find camera intrinsics in {settings_path}")

        fx = float(storage.getNode(f"{prefix}.fx").real())
        fy = float(storage.getNode(f"{prefix}.fy").real())
        cx = float(storage.getNode(f"{prefix}.cx").real())
        cy = float(storage.getNode(f"{prefix}.cy").real())
        k1 = float(storage.getNode(f"{prefix}.k1").real())
        k2 = float(storage.getNode(f"{prefix}.k2").real())
        k3 = float(storage.getNode(f"{prefix}.k3").real())
        k4 = float(storage.getNode(f"{prefix}.k4").real())
        width = int(storage.getNode("Camera.width").real())
        height = int(storage.getNode("Camera.height").real())
    finally:
        storage.release()

    return FisheyeCameraModel(
        intrinsics=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32),
        distortion=np.array([k1, k2, k3, k4], dtype=np.float32),
        image_size=(width, height),
        prefix=prefix
    )


def create_pinhole_settings(original_settings_path: Path, new_intrinsics: np.ndarray, temp_dir: Path) -> Path:
    with original_settings_path.open("r") as f:
        lines = f.readlines()

    new_lines = []
    prefix = "Camera1"
    for line in lines:
        if "Camera.fx" in line: prefix = "Camera"; break
        if "Camera1.fx" in line: prefix = "Camera1"; break

    processed_params = set()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Camera.type:"):
            new_lines.append('Camera.type: "PinHole"\n')
        elif stripped.startswith(f"{prefix}.fx:"):
            new_lines.append(f"{prefix}.fx: {new_intrinsics[0, 0]}\n")
            processed_params.add("fx")
        elif stripped.startswith(f"{prefix}.fy:"):
            new_lines.append(f"{prefix}.fy: {new_intrinsics[1, 1]}\n")
            processed_params.add("fy")
        elif stripped.startswith(f"{prefix}.cx:"):
            new_lines.append(f"{prefix}.cx: {new_intrinsics[0, 2]}\n")
            processed_params.add("cx")
        elif stripped.startswith(f"{prefix}.cy:"):
            new_lines.append(f"{prefix}.cy: {new_intrinsics[1, 2]}\n")
            processed_params.add("cy")
        elif any(stripped.startswith(f"{prefix}.k{i}:") for i in range(1, 5)):
            key = stripped.split(":")[0]
            new_lines.append(f"{key}: 0.0\n")
            processed_params.add(key.split(".")[1])
        elif stripped.startswith(f"{prefix}.p1:") or stripped.startswith(f"{prefix}.p2:"):
            key = stripped.split(":")[0]
            new_lines.append(f"{key}: 0.0\n")
            processed_params.add(key.split(".")[1])
        else:
            new_lines.append(line)

    for i in range(1, 5):
        if f"k{i}" not in processed_params: new_lines.append(f"{prefix}.k{i}: 0.0\n")
    if "p1" not in processed_params: new_lines.append(f"{prefix}.p1: 0.0\n")
    if "p2" not in processed_params: new_lines.append(f"{prefix}.p2: 0.0\n")

    temp_settings_path = temp_dir / "orbslam3_pinhole.yaml"
    with temp_settings_path.open("w") as f:
        f.writelines(new_lines)
    return temp_settings_path


def load_camera_stream(camera_dir: Path) -> list[FrameRecord]:
    data_csv = camera_dir / "data.csv"
    image_dir = camera_dir / "data"
    if not data_csv.exists(): raise FileNotFoundError(f"Missing camera csv: {data_csv}")
    frames = []
    with data_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2: continue
            frames.append(FrameRecord(timestamp_ns=int(row[0]), image_path=image_dir / row[1]))
    return frames


def load_imu_stream(imu_dir: Path) -> list[ImuRecord]:
    data_csv = imu_dir / "data.csv"
    if not data_csv.exists(): return []
    imu_samples = []
    with data_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 7: continue
            imu_samples.append(ImuRecord(timestamp_ns=int(row[0]), gyro_xyz=(float(row[1]), float(row[2]), float(row[3])), accel_xyz=(float(row[4]), float(row[5]), float(row[6]))))
    return imu_samples


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    orb_root = script_dir / "dependency" / "ORB_SLAM3"
    default_dataset = script_dir / "dataset" / "dataset-corridor1_512_16"
    if not default_dataset.exists():
        alt_dataset = script_dir.parent / "dataset"
        if alt_dataset.exists(): default_dataset = alt_dataset

    parser = argparse.ArgumentParser(description="Run monocular-inertial ORB-SLAM3 and export data for DA3 pipeline.")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument("--vocabulary-path", type=Path, default=orb_root / "Vocabulary" / "ORBvoc.txt")
    parser.add_argument("--settings-path", type=Path, default=orb_root / "Examples" / "Monocular-Inertial" / "TUM-VI.yaml")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs" / "da3_export")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--undistort-balance", type=float, default=-1.0)
    parser.add_argument("--native-tracking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only-keyframes", action="store_true", help="Export only ORB-SLAM3 KeyFrames. Recommended for clean mapping.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    vocabulary_path = args.vocabulary_path.resolve()
    settings_path = args.settings_path.resolve()
    output_dir = args.output_dir.resolve()
    da3_dir = output_dir / "da3"
    image_output_dir = da3_dir / "data"
    da3_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    camera_model = load_fisheye_camera_model(settings_path)
    undistorter = FisheyeUndistorter(camera_model, balance=args.undistort_balance)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        run_settings_path = settings_path if args.native_tracking else create_pinhole_settings(settings_path, undistorter.output_intrinsics, tmp_dir_path)

        frames = load_camera_stream(dataset_root / "mav0" / "cam0")
        imu_samples = load_imu_stream(dataset_root / "mav0" / "imu0")
        if not frames: raise RuntimeError(f"No frames found in {dataset_root}")
        if args.max_frames is not None: frames = frames[:args.max_frames]

        imu_timestamps = [sample.timestamp_ns for sample in imu_samples]
        imu_index = max(bisect.bisect_right(imu_timestamps, frames[0].timestamp_ns) - 1, 0) if imu_samples else 0
        use_clahe = not args.no_clahe
        clahe = cv2.createCLAHE(3.0, (8, 8)) if use_clahe else None

        sensor_type = pyorbslam3.Sensor.IMU_MONOCULAR if imu_samples else pyorbslam3.Sensor.MONOCULAR
        slam = pyorbslam3.System(str(vocabulary_path), str(run_settings_path), sensor_type, use_viewer=args.viewer)

        exported_keyframes = []
        all_extrinsics, all_intrinsics, all_timestamps, all_frame_indices = [], [], [], []
        processed_count, tracked_count = 0, 0

        try:
            for i, frame in enumerate(frames):
                raw_bgr = cv2.imread(str(frame.image_path))
                if raw_bgr is None: continue
                raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
                if use_clahe and clahe is not None: raw_gray = clahe.apply(raw_gray)
                slam_input = raw_gray if args.native_tracking else undistorter.undistort_image(raw_gray)

                imu_batch = []
                if sensor_type == pyorbslam3.Sensor.IMU_MONOCULAR:
                    while imu_index < len(imu_samples) and imu_samples[imu_index].timestamp_ns <= frame.timestamp_ns:
                        sample = imu_samples[imu_index]
                        imu_batch.append(pyorbslam3.ImuMeasurement(sample.timestamp_ns * 1e-9, sample.accel_xyz, sample.gyro_xyz))
                        imu_index += 1
                
                if i > 0 and sensor_type == pyorbslam3.Sensor.IMU_MONOCULAR and not imu_batch: continue

                result = slam.track_monocular(slam_input, frame.timestamp_ns * 1e-9, imu_batch)
                processed_count += 1
                
                is_keyframe = bool(result.get("is_keyframe", False))
                should_save = result["pose_valid"] and (not args.only_keyframes or is_keyframe)

                if should_save:
                    tracked_count += 1
                    image_filename = f"frame_{i:06d}.png"
                    cv2.imwrite(str(image_output_dir / image_filename), undistorter.undistort_image(raw_bgr))
                    exported_keyframes.append({"keyframe_index": len(exported_keyframes), "frame_index": i, "timestamp_ns": frame.timestamp_ns, "image_path": f"data/{image_filename}", "is_keyframe": int(is_keyframe)})
                    all_extrinsics.append(np.asarray(result["pose_matrix"], dtype=np.float32))
                    all_intrinsics.append(undistorter.output_intrinsics)
                    all_timestamps.append(frame.timestamp_ns)
                    all_frame_indices.append(i)

                if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
                    print(f"[{i + 1}/{len(frames)}] state={result['tracking_state_name']} tracked={tracked_count} kf={int(is_keyframe)}")
        finally:
            slam.shutdown()

    if not exported_keyframes: return 1
    with (da3_dir / "keyframes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["keyframe_index", "frame_index", "timestamp_ns", "image_path", "is_keyframe"])
        writer.writeheader(); writer.writerows(exported_keyframes)

    manifest = {"num_keyframes": len(exported_keyframes), "sensor": "monocular-inertial" if imu_samples else "monocular", "intrinsics_type": "pinhole", "tracking_mode": "native-fisheye" if args.native_tracking else "pinhole", "only_keyframes": args.only_keyframes}
    (da3_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    np.save(da3_dir / "extrinsics.npy", np.asarray(all_extrinsics, dtype=np.float32))
    np.save(da3_dir / "intrinsics.npy", np.asarray(all_intrinsics, dtype=np.float32))
    np.save(da3_dir / "timestamps_ns.npy", np.asarray(all_timestamps, dtype=np.int64))
    np.save(da3_dir / "frame_indices.npy", np.asarray(all_frame_indices, dtype=np.int32))
    print(f"Export complete. Processed: {processed_count}, Tracked: {tracked_count}. Outputs: {da3_dir}"); return 0

if __name__ == "__main__":
    sys.exit(main())
