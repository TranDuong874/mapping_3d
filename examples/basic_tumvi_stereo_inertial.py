#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyorbslam3
from run_orbslam3_tumvi import (
    load_camera_stream,
    load_imu_stream,
    pair_stereo_frames,
    read_grayscale_image,
)


def build_arg_parser() -> argparse.ArgumentParser:
    orb_root = REPO_ROOT / "dependency" / "ORB_SLAM3"
    parser = argparse.ArgumentParser(
        description="Minimal stereo-inertial example using the local pyorbslam3 binding.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--vocabulary-path",
        type=Path,
        default=orb_root / "Vocabulary" / "ORBvoc.txt",
    )
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=orb_root / "Examples" / "Stereo-Inertial" / "TUM-VI.yaml",
    )
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--viewer", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    dataset_root = args.dataset_root.resolve()
    left_frames = load_camera_stream(dataset_root / "mav0" / "cam0")
    right_frames = load_camera_stream(dataset_root / "mav0" / "cam1")
    stereo_frames = pair_stereo_frames(left_frames, right_frames)
    imu_samples = load_imu_stream(dataset_root / "mav0" / "imu0")

    stereo_frames = stereo_frames[: args.max_frames]
    imu_timestamps = [sample.timestamp_ns for sample in imu_samples]
    imu_index = max(bisect.bisect_right(imu_timestamps, stereo_frames[0].timestamp_ns) - 1, 0)
    clahe = cv2.createCLAHE(3.0, (8, 8))

    slam = pyorbslam3.System(
        str(args.vocabulary_path.resolve()),
        str(args.settings_path.resolve()),
        pyorbslam3.Sensor.IMU_STEREO,
        use_viewer=args.viewer,
    )

    try:
        for frame_index, frame in enumerate(stereo_frames, start=1):
            left_image = read_grayscale_image(frame.left_image_path, True, clahe)
            right_image = read_grayscale_image(frame.right_image_path, True, clahe)

            imu_batch: list[pyorbslam3.ImuMeasurement] = []
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
            print(
                f"[{frame_index}/{len(stereo_frames)}] "
                f"state={result['tracking_state_name']} "
                f"pose_valid={bool(result['pose_valid'])}"
            )

        sparse_points = np.asarray(slam.get_current_map_points(), dtype=np.float32)
        print(f"sparse_map_points={sparse_points.shape[0]}")
    finally:
        slam.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
