#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

import pyorbslam3


@dataclass(frozen=True)
class FrameRecord:
    timestamp_ns: int
    image_path: Path


@dataclass(frozen=True)
class StereoFrameRecord:
    timestamp_ns: int
    left_image_path: Path
    right_image_path: Path


@dataclass(frozen=True)
class ImuRecord:
    timestamp_ns: int
    gyro_xyz: tuple[float, float, float]
    accel_xyz: tuple[float, float, float]


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    orb_root = script_dir / "dependency" / "ORB_SLAM3"

    parser = argparse.ArgumentParser(
        description="Run ORB-SLAM3 stereo-IMU through the local Python binding on a TUM-VI style dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing mav0/cam0, mav0/cam1, and mav0/imu0.",
    )
    parser.add_argument(
        "--vocabulary-path",
        type=Path,
        default=orb_root / "Vocabulary" / "ORBvoc.txt",
        help="Path to ORB vocabulary text file.",
    )
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=orb_root / "Examples" / "Stereo-Inertial" / "TUM-VI.yaml",
        help="Path to ORB-SLAM3 settings YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs" / "tumvi_run",
        help="Directory for tracking and sparse point cloud outputs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for a shorter smoke run.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N frames.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Enable Pangolin viewer.",
    )
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE preprocessing used by the native TUM-VI example.",
    )
    return parser


def load_camera_stream(camera_dir: Path) -> list[FrameRecord]:
    data_csv = camera_dir / "data.csv"
    image_dir = camera_dir / "data"
    if not data_csv.exists():
        raise FileNotFoundError(f"Missing camera csv: {data_csv}")

    frames: list[FrameRecord] = []
    with data_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            frames.append(
                FrameRecord(
                    timestamp_ns=int(row[0]),
                    image_path=image_dir / row[1],
                )
            )
    return frames


def pair_stereo_frames(left_frames: Iterable[FrameRecord], right_frames: Iterable[FrameRecord]) -> list[StereoFrameRecord]:
    right_by_ts = {frame.timestamp_ns: frame for frame in right_frames}
    stereo_frames: list[StereoFrameRecord] = []
    for left_frame in left_frames:
        right_frame = right_by_ts.get(left_frame.timestamp_ns)
        if right_frame is None:
            continue
        stereo_frames.append(
            StereoFrameRecord(
                timestamp_ns=left_frame.timestamp_ns,
                left_image_path=left_frame.image_path,
                right_image_path=right_frame.image_path,
            )
        )
    return stereo_frames


def load_imu_stream(imu_dir: Path) -> list[ImuRecord]:
    data_csv = imu_dir / "data.csv"
    if not data_csv.exists():
        raise FileNotFoundError(f"Missing imu csv: {data_csv}")

    imu_samples: list[ImuRecord] = []
    with data_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 7:
                continue
            imu_samples.append(
                ImuRecord(
                    timestamp_ns=int(row[0]),
                    gyro_xyz=(float(row[1]), float(row[2]), float(row[3])),
                    accel_xyz=(float(row[4]), float(row[5]), float(row[6])),
                )
            )
    return imu_samples


def read_grayscale_image(image_path: Path, use_clahe: bool, clahe: cv2.CLAHE | None) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    if use_clahe and clahe is not None:
        image = clahe.apply(image)
    return image


def quaternion_xyzw_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rotation[2, 1] - rotation[1, 2]) * s
        qy = (rotation[0, 2] - rotation[2, 0]) * s
        qz = (rotation[1, 0] - rotation[0, 1]) * s
        return (qx, qy, qz, qw)

    diagonal = np.diag(rotation)
    if diagonal[0] > diagonal[1] and diagonal[0] > diagonal[2]:
        s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
        return (qx, qy, qz, qw)
    if diagonal[1] > diagonal[2]:
        s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
        return (qx, qy, qz, qw)

    s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
    qw = (rotation[1, 0] - rotation[0, 1]) / s
    qx = (rotation[0, 2] + rotation[2, 0]) / s
    qy = (rotation[1, 2] + rotation[2, 1]) / s
    qz = 0.25 * s
    return (qx, qy, qz, qw)


def save_xyz(points: np.ndarray, output_path: Path) -> None:
    if points.size == 0:
        output_path.write_text("", encoding="utf-8")
        return
    np.savetxt(output_path, points, fmt="%.6f")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    vocabulary_path = args.vocabulary_path.resolve()
    settings_path = args.settings_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    left_frames = load_camera_stream(dataset_root / "mav0" / "cam0")
    right_frames = load_camera_stream(dataset_root / "mav0" / "cam1")
    stereo_frames = pair_stereo_frames(left_frames, right_frames)
    imu_samples = load_imu_stream(dataset_root / "mav0" / "imu0")
    if not stereo_frames:
        raise RuntimeError("No stereo frame pairs found.")
    if not imu_samples:
        raise RuntimeError("No IMU samples found.")

    if args.max_frames is not None:
        stereo_frames = stereo_frames[: args.max_frames]

    imu_timestamps = [sample.timestamp_ns for sample in imu_samples]
    imu_index = max(bisect.bisect_right(imu_timestamps, stereo_frames[0].timestamp_ns) - 1, 0)
    use_clahe = not args.no_clahe
    clahe = cv2.createCLAHE(3.0, (8, 8)) if use_clahe else None

    trajectory_lines: list[str] = []
    camera_path_points: list[np.ndarray] = []

    slam = pyorbslam3.System(
        str(vocabulary_path),
        str(settings_path),
        pyorbslam3.Sensor.IMU_STEREO,
        use_viewer=args.viewer,
    )

    tracking_log_path = output_dir / "tracking_log.csv"
    sparse_points_path = output_dir / "sparse_points.xyz"
    trajectory_path = output_dir / "trajectory_tum.txt"
    camera_path_path = output_dir / "camera_path.xyz"

    processed_frames = 0
    valid_poses = 0
    sparse_points = np.empty((0, 3), dtype=np.float32)

    try:
        with tracking_log_path.open("w", newline="", encoding="utf-8") as log_handle:
            writer = csv.writer(log_handle)
            writer.writerow(
                [
                    "frame_index",
                    "timestamp_ns",
                    "tracking_state",
                    "pose_valid",
                ]
            )

            for frame_index, frame in enumerate(stereo_frames):
                left_image = read_grayscale_image(frame.left_image_path, use_clahe, clahe)
                right_image = read_grayscale_image(frame.right_image_path, use_clahe, clahe)

                imu_batch: list[pyorbslam3.ImuMeasurement] = []
                if frame_index > 0:
                    while imu_index < len(imu_samples) and imu_samples[imu_index].timestamp_ns <= frame.timestamp_ns:
                        sample = imu_samples[imu_index]
                        imu_batch.append(
                            pyorbslam3.ImuMeasurement(
                                sample.timestamp_ns * 1e-9,
                                sample.accel_xyz,
                                sample.gyro_xyz,
                            )
                        )
                        imu_index += 1

                result = slam.track_stereo(
                    left_image,
                    right_image,
                    frame.timestamp_ns * 1e-9,
                    imu_batch,
                )
                writer.writerow(
                    [
                        frame_index,
                        frame.timestamp_ns,
                        result["tracking_state_name"],
                        int(bool(result["pose_valid"])),
                    ]
                )

                processed_frames += 1
                if result["pose_valid"]:
                    pose_cw = np.asarray(result["pose_matrix"], dtype=np.float64)
                    pose_wc = np.linalg.inv(pose_cw)
                    camera_position = pose_wc[:3, 3]
                    qx, qy, qz, qw = quaternion_xyzw_from_rotation(pose_wc[:3, :3])
                    trajectory_lines.append(
                        f"{frame.timestamp_ns * 1e-9:.9f} "
                        f"{camera_position[0]:.9f} {camera_position[1]:.9f} {camera_position[2]:.9f} "
                        f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
                    )
                    camera_path_points.append(camera_position)
                    valid_poses += 1

                if args.progress_every > 0 and (frame_index + 1) % args.progress_every == 0:
                    print(
                        f"[{frame_index + 1}/{len(stereo_frames)}] "
                        f"state={result['tracking_state_name']} "
                        f"pose_valid={bool(result['pose_valid'])}"
                    )

        sparse_points = np.asarray(slam.get_current_map_points(), dtype=np.float32)
    finally:
        slam.shutdown()

    trajectory_path.write_text("\n".join(trajectory_lines) + ("\n" if trajectory_lines else ""), encoding="utf-8")
    if camera_path_points:
        save_xyz(np.vstack(camera_path_points), camera_path_path)
    else:
        camera_path_path.write_text("", encoding="utf-8")
    save_xyz(sparse_points, sparse_points_path)

    print(f"Processed frames: {processed_frames}")
    print(f"Valid poses: {valid_poses}")
    print(f"Sparse map points: {int(sparse_points.shape[0])}")
    print(f"Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
