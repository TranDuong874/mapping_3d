#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

SCRIPT_DIR = Path(__file__).resolve().parent
DA3_SRC = SCRIPT_DIR / "dependency" / "Depth-Anything-3" / "src"
if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))

# The pip cv2 wheel sets Qt plugin env vars inside virtualenvs. That collides with
# ORB-SLAM3's native viewer thread, which uses the system OpenCV/Qt stack.
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_FONTDIR", None)

import pyorbslam3
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.glb import (
    _depths_to_world_points_with_colors,
    get_conf_thresh,
)


@dataclass(frozen=True)
class FrameRecord:
    timestamp_ns: int
    image_path: Path


@dataclass(frozen=True)
class KeyframePacket:
    frame_index: int
    timestamp_ns: int
    image_rgb: np.ndarray
    pose_cw: np.ndarray
    intrinsics: np.ndarray


@dataclass(frozen=True)
class FisheyeCameraModel:
    intrinsics: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]


def depth_to_preview(depth_map: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(depth_map)
    if not np.any(finite_mask):
        return np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)

    valid_depth = depth_map[finite_mask]
    lo = float(np.percentile(valid_depth, 2.0))
    hi = float(np.percentile(valid_depth, 98.0))
    if hi <= lo:
        hi = lo + 1e-6
    clipped = np.clip(depth_map, lo, hi)
    normalized = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def make_da3_preview(image_rgb: np.ndarray, depth_map: np.ndarray, metadata_text: str) -> np.ndarray:
    rgb_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    depth_vis = depth_to_preview(depth_map)
    if rgb_bgr.shape[:2] != depth_vis.shape[:2]:
        rgb_bgr = cv2.resize(rgb_bgr, (depth_vis.shape[1], depth_vis.shape[0]), interpolation=cv2.INTER_AREA)
    panel = np.hstack([rgb_bgr, depth_vis])
    cv2.putText(
        panel,
        metadata_text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


class PreviewWindow:
    def __init__(self, title: str) -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root)
        self.label.pack()
        self._photo: ImageTk.PhotoImage | None = None
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def update(self, frame_bgr: np.ndarray) -> None:
        if self._closed:
            return
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self._photo = ImageTk.PhotoImage(image=image)
            self.label.configure(image=self._photo)
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    orb_root = SCRIPT_DIR / "dependency" / "ORB_SLAM3"

    parser = argparse.ArgumentParser(
        description=(
            "Run monocular ORB-SLAM3 in realtime on a TUM-VI style dataset and "
            "feed keyframes to a background Depth Anything 3 worker for pose-conditioned "
            "depth and point-cloud generation."
        ),
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
        default=orb_root / "Examples" / "Monocular" / "TUM-VI.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs" / "orbslam3_da3")
    parser.add_argument("--viewer", action="store_true", help="Enable ORB-SLAM3 Pangolin viewer.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Run as fast as possible instead of sleeping to match dataset timestamps.",
    )
    parser.add_argument(
        "--da3-model",
        default="depth-anything/da3-small",
        help="Hugging Face model id or local DA3 checkpoint name for DepthAnything3.from_pretrained().",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for DA3 inference.",
    )
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--da3-min-views",
        type=int,
        default=3,
        help="Minimum buffered keyframes before running DA3 in the worker.",
    )
    parser.add_argument(
        "--da3-max-keyframes",
        type=int,
        default=24,
        help="Maximum number of recent keyframes used per DA3 inference call.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=32,
        help="Maximum number of pending keyframes for the DA3 worker.",
    )
    parser.add_argument(
        "--use-ray-pose",
        action="store_true",
        help="Use DA3 ray-pose mode instead of the default camera decoder.",
    )
    parser.add_argument(
        "--conf-thresh-percentile",
        type=float,
        default=40.0,
        help="Confidence percentile used when filtering the DA3 point cloud.",
    )
    parser.add_argument(
        "--num-max-points",
        type=int,
        default=1_000_000,
        help="Maximum number of points written to the DA3 PLY output.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.02,
        help="Voxel size in world units used to merge dense point contributions into one global cloud.",
    )
    parser.add_argument(
        "--no-da3-window",
        action="store_true",
        help="Disable the realtime DA3 preview window.",
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
            frames.append(FrameRecord(timestamp_ns=int(row[0]), image_path=image_dir / row[1]))
    return frames


def read_grayscale_image(image_path: Path, use_clahe: bool, clahe: cv2.CLAHE | None) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    if use_clahe and clahe is not None:
        image = clahe.apply(image)
    return image


def read_rgb_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_camera_intrinsics(settings_path: Path) -> np.ndarray:
    storage = cv2.FileStorage(str(settings_path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Failed to open settings file: {settings_path}")
    try:
        fx = float(storage.getNode("Camera1.fx").real())
        fy = float(storage.getNode("Camera1.fy").real())
        cx = float(storage.getNode("Camera1.cx").real())
        cy = float(storage.getNode("Camera1.cy").real())
    finally:
        storage.release()

    intrinsics = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intrinsics


def load_fisheye_camera_model(settings_path: Path) -> FisheyeCameraModel:
    storage = cv2.FileStorage(str(settings_path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Failed to open settings file: {settings_path}")
    try:
        fx = float(storage.getNode("Camera1.fx").real())
        fy = float(storage.getNode("Camera1.fy").real())
        cx = float(storage.getNode("Camera1.cx").real())
        cy = float(storage.getNode("Camera1.cy").real())
        k1 = float(storage.getNode("Camera1.k1").real())
        k2 = float(storage.getNode("Camera1.k2").real())
        k3 = float(storage.getNode("Camera1.k3").real())
        k4 = float(storage.getNode("Camera1.k4").real())
        width = int(storage.getNode("Camera.width").real())
        height = int(storage.getNode("Camera.height").real())
    finally:
        storage.release()

    return FisheyeCameraModel(
        intrinsics=np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        distortion=np.array([k1, k2, k3, k4], dtype=np.float32),
        image_size=(width, height),
    )


class FisheyeUndistorter:
    def __init__(self, camera_model: FisheyeCameraModel, balance: float = 0.0) -> None:
        width, height = camera_model.image_size
        identity = np.eye(3, dtype=np.float64)
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
        self.map1 = map1
        self.map2 = map2
        self.output_intrinsics = new_intrinsics.astype(np.float32)

    def undistort(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)


def write_point_cloud_ply(points: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=False):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def downsample_points(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_points:
        return points, colors
    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[indices], colors[indices]


def voxel_downsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0 or points.shape[0] == 0:
        return points, colors
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return points[unique_indices], colors[unique_indices]


class DepthEstimationService:
    def __init__(
        self,
        model_name: str,
        device: str,
        output_dir: Path,
        process_res: int,
        min_views: int,
        max_keyframes: int,
        use_ray_pose: bool,
        conf_thresh_percentile: float,
        num_max_points: int,
        queue_size: int,
        voxel_size: float,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.output_dir = output_dir
        self.process_res = process_res
        self.min_views = min_views
        self.max_keyframes = max_keyframes
        self.use_ray_pose = use_ray_pose
        self.conf_thresh_percentile = conf_thresh_percentile
        self.num_max_points = num_max_points
        self.voxel_size = voxel_size
        self.queue: queue.Queue[KeyframePacket | None] = queue.Queue(maxsize=queue_size)
        self.worker = threading.Thread(target=self._run, name="da3-worker", daemon=True)
        self.exception: Exception | None = None
        self.lock = threading.Lock()
        self.processed_keyframes = 0
        self.latest_ply_path: Path | None = None
        self.latest_preview: np.ndarray | None = None
        self.dropped_keyframes = 0
        self.global_points = np.empty((0, 3), dtype=np.float32)
        self.global_colors = np.empty((0, 3), dtype=np.uint8)
        self.last_integrated_frame_index: int | None = None
        self.last_frame_point_count = 0

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.worker.start()

    def submit(self, packet: KeyframePacket) -> bool:
        try:
            self.queue.put_nowait(packet)
            return True
        except queue.Full:
            pass

        # Keep the queue bounded and biased toward fresh keyframes so SLAM never blocks.
        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass
        else:
            with self.lock:
                self.dropped_keyframes += 1

        try:
            self.queue.put_nowait(packet)
            return True
        except queue.Full:
            with self.lock:
                self.dropped_keyframes += 1
            return False

    def get_latest_preview(self) -> np.ndarray | None:
        with self.lock:
            if self.latest_preview is None:
                return None
            return self.latest_preview.copy()

    def get_stats(self) -> dict[str, int]:
        with self.lock:
            return {
                "processed_keyframes": self.processed_keyframes,
                "dropped_keyframes": self.dropped_keyframes,
                "global_num_points": int(self.global_points.shape[0]),
            }

    def finish(self) -> None:
        self.queue.put(None)
        self.worker.join()
        if self.exception is not None:
            raise self.exception

    def _run(self) -> None:
        try:
            model = DepthAnything3.from_pretrained(self.model_name).to(self.device)
            input_processor = model.input_processor

            def single_worker_input_processor(image, extrinsics=None, intrinsics=None, process_res=504, process_res_method="upper_bound_resize"):
                return input_processor(
                    image,
                    extrinsics,
                    intrinsics,
                    process_res,
                    process_res_method,
                    num_workers=1,
                    sequential=True,
                )

            model.input_processor = single_worker_input_processor
            buffered: list[KeyframePacket] = []
            while True:
                packet = self.queue.get()
                if packet is None:
                    break
                buffered.append(packet)
                if len(buffered) > self.max_keyframes:
                    buffered = buffered[-self.max_keyframes :]
                if len(buffered) < self.min_views:
                    continue
                self._infer_and_export(model, buffered, suffix="latest")

            if buffered:
                if self.last_integrated_frame_index != buffered[-1].frame_index or self.processed_keyframes == 0:
                    self._infer_and_export(model, buffered, suffix="final")
                else:
                    self._export_global_cloud(
                        suffix="final",
                        latest_packet=buffered[-1],
                        window_keyframes=len(buffered),
                        latest_frame_points=self.last_frame_point_count,
                    )
        except Exception as exc:  # pragma: no cover - surfaced back to caller
            self.exception = exc

    def _export_global_cloud(
        self,
        suffix: str,
        latest_packet: KeyframePacket,
        window_keyframes: int,
        latest_frame_points: int,
    ) -> Path:
        with self.lock:
            export_points = self.global_points.copy()
            export_colors = self.global_colors.copy()
            global_num_points = int(export_points.shape[0])

        ply_path = self.output_dir / f"da3_global_pointcloud_{suffix}.ply"
        write_point_cloud_ply(export_points, export_colors, ply_path)

        metadata = {
            "window_keyframes": window_keyframes,
            "global_num_points": global_num_points,
            "latest_frame_points": latest_frame_points,
            "latest_timestamp_ns": latest_packet.timestamp_ns,
            "latest_frame_index": latest_packet.frame_index,
            "model_name": self.model_name,
            "voxel_size": self.voxel_size,
        }
        metadata_path = self.output_dir / f"da3_global_pointcloud_{suffix}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return ply_path

    def _infer_and_export(self, model: DepthAnything3, buffered: list[KeyframePacket], suffix: str) -> None:
        images = [packet.image_rgb for packet in buffered]
        extrinsics = np.stack([packet.pose_cw for packet in buffered]).astype(np.float32)
        intrinsics = np.stack([packet.intrinsics for packet in buffered]).astype(np.float32)

        prediction = model.inference(
            image=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=True,
            process_res=self.process_res,
            use_ray_pose=self.use_ray_pose,
            ref_view_strategy="middle",
        )

        if prediction.conf is None:
            conf_thr = 0.0
        else:
            conf_thr = get_conf_thresh(
                prediction,
                getattr(prediction, "sky_mask", None),
                conf_thresh=1.05,
                conf_thresh_percentile=self.conf_thresh_percentile,
                ensure_thresh_percentile=90.0,
            )

        latest_depth = prediction.depth[-1:]
        latest_intrinsics = prediction.intrinsics[-1:]
        latest_extrinsics = prediction.extrinsics[-1:]
        latest_images = prediction.processed_images[-1:]
        latest_conf = None if prediction.conf is None else prediction.conf[-1:]

        points, colors = _depths_to_world_points_with_colors(
            latest_depth,
            latest_intrinsics,
            latest_extrinsics,
            latest_images,
            latest_conf,
            conf_thr,
        )
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        colors = colors[finite]
        with self.lock:
            if self.global_points.shape[0] == 0:
                merged_points = points.astype(np.float32, copy=False)
                merged_colors = colors.astype(np.uint8, copy=False)
            else:
                merged_points = np.concatenate(
                    [self.global_points, points.astype(np.float32, copy=False)],
                    axis=0,
                )
                merged_colors = np.concatenate(
                    [self.global_colors, colors.astype(np.uint8, copy=False)],
                    axis=0,
                )
            merged_points, merged_colors = voxel_downsample_points(
                merged_points,
                merged_colors,
                self.voxel_size,
            )
            merged_points, merged_colors = downsample_points(
                merged_points,
                merged_colors,
                self.num_max_points,
            )
            self.global_points = merged_points
            self.global_colors = merged_colors
            global_num_points = int(self.global_points.shape[0])

        preview = make_da3_preview(
            buffered[-1].image_rgb,
            prediction.depth[-1],
            (
                f"DA3 window={len(buffered)} "
                f"frame_pts={points.shape[0]} "
                f"global_pts={global_num_points} "
                f"ts={buffered[-1].timestamp_ns * 1e-9:.3f}s"
            ),
        )

        with self.lock:
            self.processed_keyframes += 1
            self.last_integrated_frame_index = buffered[-1].frame_index
            self.last_frame_point_count = int(points.shape[0])
            self.latest_preview = preview
        ply_path = self._export_global_cloud(
            suffix=suffix,
            latest_packet=buffered[-1],
            window_keyframes=len(buffered),
            latest_frame_points=int(points.shape[0]),
        )
        metadata_path = self.output_dir / f"da3_global_pointcloud_{suffix}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["is_metric"] = to_jsonable(prediction.is_metric)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with self.lock:
            self.latest_ply_path = ply_path
        print(
            f"[DA3] integrated frame {buffered[-1].frame_index} "
            f"with {points.shape[0]} points; global cloud now has {global_num_points} points "
            f"at {ply_path}"
        )


def main() -> int:
    args = build_arg_parser().parse_args()

    dataset_root = args.dataset_root.resolve()
    vocabulary_path = args.vocabulary_path.resolve()
    settings_path = args.settings_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_camera_stream(dataset_root / "mav0" / "cam0")
    if not frames:
        raise RuntimeError("No monocular frames found in mav0/cam0.")
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    use_clahe = not args.no_clahe
    clahe = cv2.createCLAHE(3.0, (8, 8)) if use_clahe else None
    camera_model = load_fisheye_camera_model(settings_path)
    undistorter = FisheyeUndistorter(camera_model)
    intrinsics = undistorter.output_intrinsics

    depth_service = DepthEstimationService(
        model_name=args.da3_model,
        device=args.device,
        output_dir=output_dir / "da3",
        process_res=args.process_res,
        min_views=max(1, args.da3_min_views),
        max_keyframes=max(1, args.da3_max_keyframes),
        use_ray_pose=args.use_ray_pose,
        conf_thresh_percentile=args.conf_thresh_percentile,
        num_max_points=args.num_max_points,
        queue_size=max(1, args.queue_size),
        voxel_size=max(0.0, args.voxel_size),
    )
    depth_service.start()

    slam = pyorbslam3.System(
        str(vocabulary_path),
        str(settings_path),
        pyorbslam3.Sensor.MONOCULAR,
        use_viewer=args.viewer,
    )

    tracking_log_path = output_dir / "tracking_log.csv"
    wall_start = time.perf_counter()
    sequence_start_ts = frames[0].timestamp_ns

    processed_frames = 0
    keyframes_submitted = 0
    keyframes_dropped = 0
    preview_window = PreviewWindow("DA3 Depth") if not args.no_da3_window else None

    try:
        with tracking_log_path.open("w", newline="", encoding="utf-8") as log_handle:
            writer = csv.writer(log_handle)
            writer.writerow(
                [
                    "frame_index",
                    "timestamp_ns",
                    "tracking_state",
                    "pose_valid",
                    "is_keyframe",
                ]
            )

            for frame_index, frame in enumerate(frames):
                if not args.no_realtime:
                    elapsed_dataset_s = (frame.timestamp_ns - sequence_start_ts) * 1e-9
                    elapsed_wall_s = time.perf_counter() - wall_start
                    sleep_s = elapsed_dataset_s - elapsed_wall_s
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                image = read_grayscale_image(frame.image_path, use_clahe, clahe)
                result = slam.track_monocular(image, frame.timestamp_ns * 1e-9)
                writer.writerow(
                    [
                        frame_index,
                        frame.timestamp_ns,
                        result["tracking_state_name"],
                        int(bool(result["pose_valid"])),
                        int(bool(result["is_keyframe"])),
                    ]
                )

                processed_frames += 1
                if result["pose_valid"] and result["is_keyframe"]:
                    image_rgb = undistorter.undistort(read_rgb_image(frame.image_path))
                    submitted = depth_service.submit(
                        KeyframePacket(
                            frame_index=frame_index,
                            timestamp_ns=frame.timestamp_ns,
                            image_rgb=image_rgb,
                            pose_cw=np.asarray(result["pose_matrix"], dtype=np.float32),
                            intrinsics=intrinsics,
                        )
                    )
                    if submitted:
                        keyframes_submitted += 1
                    else:
                        keyframes_dropped += 1

                if preview_window is not None:
                    preview = depth_service.get_latest_preview()
                    if preview is not None:
                        preview_window.update(preview)

                if args.progress_every > 0 and (frame_index + 1) % args.progress_every == 0:
                    print(
                        f"[{frame_index + 1}/{len(frames)}] "
                        f"state={result['tracking_state_name']} "
                        f"pose_valid={bool(result['pose_valid'])} "
                        f"is_keyframe={bool(result['is_keyframe'])}"
                    )
    finally:
        slam.shutdown()
        try:
            depth_service.finish()
        finally:
            if preview_window is not None:
                preview = depth_service.get_latest_preview()
                if preview is not None:
                    preview_window.update(preview)
                preview_window.close()

    summary = {
        "processed_frames": processed_frames,
        "keyframes_submitted": keyframes_submitted,
        "keyframes_dropped_by_submit": keyframes_dropped,
        "da3_worker_stats": depth_service.get_stats(),
        "da3_latest_pointcloud": str(depth_service.latest_ply_path) if depth_service.latest_ply_path else None,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Processed frames: {processed_frames}")
    print(f"Keyframes submitted to DA3: {keyframes_submitted}")
    print(f"Keyframes dropped at enqueue: {keyframes_dropped}")
    if depth_service.latest_ply_path is not None:
        print(f"DA3 point cloud: {depth_service.latest_ply_path}")
    print(f"Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
