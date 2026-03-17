#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
import traceback
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
UNIDEPTH_SRC_DEFAULT = Path("/home/tranduong/dev/thesis_prototype/spatial_mapping_demo/UniDepth")

# The pip cv2 wheel may inject Qt plugin env vars that collide with Pangolin.
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_FONTDIR", None)

import pyorbslam3
from depth_anything_3.api import DepthAnything3


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


@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    timestamp_ns: int
    image_path: str
    pose_cw: np.ndarray
    sparse_keypoints_uv: np.ndarray
    sparse_world_points_xyz: np.ndarray


@dataclass(frozen=True)
class ProcessedFramePacket:
    frame_index: int
    timestamp_ns: int
    image_rgb: np.ndarray
    pose_cw: np.ndarray
    intrinsics: np.ndarray
    sparse_keypoints_uv: np.ndarray
    sparse_world_points_xyz: np.ndarray


@dataclass(frozen=True)
class DepthWorkerConfig:
    depth_backend: str
    fusion_mode: str
    model_name: str
    unidepth_src: str
    unidepth_resolution_level: int
    device: str
    output_dir: str
    settings_path: str
    undistort_balance: float
    process_res: int
    process_res_method: str
    pose_conditioned: bool
    context_keyframes: int
    reprojection_stride: int
    min_depth_m: float
    max_depth_m: float
    conf_percentile: float
    sky_threshold: float
    min_scale_correspondences: int
    voxel_size: float
    tsdf_sdf_trunc_m: float
    penetration_tolerance_voxels: int
    camera_clearance_radius_m: float
    camera_clearance_below_m: float
    max_map_voxels: int
    live_export_interval_sec: float
    live_point_budget: int
    final_point_budget: int


@dataclass(frozen=True)
class DepthWorkerUpdate:
    kind: str
    stats: dict[str, int | float | str | None]
    preview_bgr: np.ndarray | None = None
    message: str | None = None


class PreviewWindow:
    _app_root: tk.Tk | None = None
    _num_windows = 0

    def __init__(self, title: str) -> None:
        if PreviewWindow._app_root is None:
            PreviewWindow._app_root = tk.Tk()
            self.window = PreviewWindow._app_root
        else:
            self.window = tk.Toplevel(PreviewWindow._app_root)
        PreviewWindow._num_windows += 1
        self.window.title(title)
        self.label = tk.Label(self.window)
        self.label.pack()
        self._photo: ImageTk.PhotoImage | None = None
        self._closed = False
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def update(self, frame_bgr: np.ndarray) -> None:
        if self._closed:
            return
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self._photo = ImageTk.PhotoImage(image=image)
            self.label.configure(image=self._photo)
            self.window.update_idletasks()
            self.window.update()
        except tk.TclError:
            self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.window.destroy()
        except tk.TclError:
            pass


class MapViewerProcess:
    def __init__(self, title: str, output_dir: Path) -> None:
        self.title = title
        self.output_dir = output_dir
        self.process: subprocess.Popen[str] | None = None

    @property
    def enabled(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> None:
        viewer_script = SCRIPT_DIR / "live_occupancy_viewer.py"
        self.process = subprocess.Popen(
            [sys.executable, str(viewer_script), str(self.output_dir), self.title],
            text=True,
        )

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        self.process = None


class FisheyeUndistorter:
    def __init__(self, camera_model: FisheyeCameraModel, balance: float) -> None:
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
        self.input_intrinsics = camera_model.intrinsics.astype(np.float32)
        self.output_intrinsics = new_intrinsics.astype(np.float32)
        self.distortion = camera_model.distortion.astype(np.float32)
        self.map1 = map1
        self.map2 = map2

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

    def undistort_points(self, points_uv: np.ndarray) -> np.ndarray:
        if points_uv.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        points = points_uv.reshape(-1, 1, 2).astype(np.float32, copy=False)
        undistorted = cv2.fisheye.undistortPoints(
            points,
            self.input_intrinsics,
            self.distortion,
            P=self.output_intrinsics,
        )
        return undistorted.reshape(-1, 2).astype(np.float32, copy=False)


def camera_position_from_pose_cw(pose_cw: np.ndarray) -> np.ndarray:
    rotation_cw = pose_cw[:3, :3].astype(np.float64, copy=False)
    translation_cw = pose_cw[:3, 3].astype(np.float64, copy=False)
    rotation_wc = rotation_cw.T
    return (-rotation_wc @ translation_cw).astype(np.float32, copy=False)


def infer_vertical_axis_from_positions(camera_positions: list[np.ndarray]) -> int:
    if len(camera_positions) < 2:
        return 1
    points = np.asarray(camera_positions, dtype=np.float32)
    lower = np.percentile(points, 5.0, axis=0)
    upper = np.percentile(points, 95.0, axis=0)
    spans = np.maximum(upper - lower, 1e-6)
    return int(np.argmin(spans))


def clearance_voxels_for_segment(
    start_world: np.ndarray | None,
    end_world: np.ndarray,
    voxel_size: float,
    clearance_radius_m: float,
    clearance_below_m: float,
    vertical_axis: int,
) -> set[tuple[int, int, int]]:
    radius_m = float(max(clearance_radius_m, 0.0))
    below_m = float(max(clearance_below_m, 0.0))
    if radius_m <= 0.0:
        return set()

    start = end_world.astype(np.float32, copy=False) if start_world is None else start_world.astype(np.float32, copy=False)
    end = end_world.astype(np.float32, copy=False)
    distance_m = float(np.linalg.norm(end - start))
    step_m = max(min(radius_m * 0.5, voxel_size), 1e-3)
    num_samples = max(1, int(np.ceil(distance_m / step_m)))

    horizontal_axes = [axis for axis in range(3) if axis != vertical_axis]
    radius_grid = max(1, int(np.ceil(radius_m / voxel_size)))
    below_grid = max(0, int(np.ceil(below_m / voxel_size)))
    clearance_keys: set[tuple[int, int, int]] = set()

    for alpha in np.linspace(0.0, 1.0, num_samples + 1, dtype=np.float32):
        sample_world = (1.0 - alpha) * start + alpha * end
        center_key = np.floor(sample_world / voxel_size).astype(np.int32)
        for dx in range(-radius_grid, radius_grid + 1):
            for dy in range(-radius_grid, radius_grid + 1):
                for dz in range(-below_grid, radius_grid + 1):
                    offset = np.array([dx, dy, dz], dtype=np.int32)
                    voxel_key = center_key + offset
                    voxel_center = ((voxel_key.astype(np.float32) + 0.5) * voxel_size).astype(np.float32, copy=False)
                    delta = voxel_center - sample_world
                    vertical_delta = float(delta[vertical_axis])
                    if vertical_delta < -below_m or vertical_delta > radius_m:
                        continue
                    vertical_scale = radius_m if vertical_delta >= 0.0 else max(below_m, voxel_size * 0.5)
                    horizontal_sq = float(np.sum(delta[horizontal_axes] ** 2))
                    normalized = (horizontal_sq / max(radius_m * radius_m, 1e-6)) + (
                        (vertical_delta / vertical_scale) ** 2
                    )
                    if normalized <= 1.0:
                        clearance_keys.add((int(voxel_key[0]), int(voxel_key[1]), int(voxel_key[2])))
    return clearance_keys


class RaycastOccupancyFusion:
    def __init__(
        self,
        voxel_size: float,
        penetration_tolerance_voxels: int,
        max_voxels: int,
        camera_clearance_radius_m: float,
        camera_clearance_below_m: float,
    ) -> None:
        self.voxel_size = float(voxel_size)
        self.penetration_tolerance_voxels = max(0, int(penetration_tolerance_voxels))
        self.max_voxels = max(0, int(max_voxels))
        self.camera_clearance_radius_m = float(max(camera_clearance_radius_m, 0.0))
        self.camera_clearance_below_m = float(max(camera_clearance_below_m, 0.0))
        self.integrated_frames = 0
        self.occupied_threshold = 1.0
        self.max_score = 12.0
        self.remove_threshold = 0.05
        self.hit_score_delta = 1.0
        self.blocked_wall_score_delta = 0.25
        self.pass_through_decay = 0.10
        self._scores: dict[tuple[int, int, int], float] = {}
        self._color_sums: dict[tuple[int, int, int], np.ndarray] = {}
        self._hit_counts: dict[tuple[int, int, int], int] = {}
        self._camera_positions: list[np.ndarray] = []
        self._clearance_voxels: set[tuple[int, int, int]] = set()

    def num_occupied_voxels(self) -> int:
        return sum(
            1
            for key, score in self._scores.items()
            if score >= self.occupied_threshold and key not in self._clearance_voxels
        )

    def _apply_camera_clearance(self, camera_world: np.ndarray) -> int:
        if self.camera_clearance_radius_m <= 0.0:
            return 0
        previous = self._camera_positions[-1] if self._camera_positions else None
        current = camera_world.astype(np.float32, copy=False)
        self._camera_positions.append(current.copy())
        vertical_axis = infer_vertical_axis_from_positions(self._camera_positions)
        carved_keys = clearance_voxels_for_segment(
            previous,
            current,
            self.voxel_size,
            self.camera_clearance_radius_m,
            self.camera_clearance_below_m,
            vertical_axis,
        )
        if not carved_keys:
            return 0
        self._clearance_voxels.update(carved_keys)
        removed = 0
        for voxel_key in carved_keys:
            if voxel_key in self._scores:
                removed += 1
            self._scores.pop(voxel_key, None)
            self._color_sums.pop(voxel_key, None)
            self._hit_counts.pop(voxel_key, None)
        return removed

    def update_camera_pose(self, pose_cw: np.ndarray) -> int:
        return self._apply_camera_clearance(camera_position_from_pose_cw(pose_cw))

    def _increase_score(self, key: tuple[int, int, int], delta: float) -> None:
        updated = min(self.max_score, self._scores.get(key, 0.0) + float(delta))
        if updated <= self.remove_threshold:
            self._scores.pop(key, None)
            self._color_sums.pop(key, None)
            self._hit_counts.pop(key, None)
            return
        self._scores[key] = updated

    def _decrease_score(self, key: tuple[int, int, int], delta: float) -> None:
        current = self._scores.get(key)
        if current is None:
            return
        updated = current - float(delta)
        if updated <= self.remove_threshold:
            self._scores.pop(key, None)
            self._color_sums.pop(key, None)
            self._hit_counts.pop(key, None)
        else:
            self._scores[key] = updated

    def _register_hit(self, key: tuple[int, int, int], color_rgb_u8: np.ndarray) -> None:
        self._increase_score(key, self.hit_score_delta)
        if key not in self._color_sums:
            self._color_sums[key] = color_rgb_u8.astype(np.float64, copy=True)
            self._hit_counts[key] = 1
        else:
            self._color_sums[key] += color_rgb_u8.astype(np.float64, copy=False)
            self._hit_counts[key] += 1

    def _trace_voxels(
        self,
        origin_xyz: np.ndarray,
        endpoint_xyz: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        origin_grid = origin_xyz.astype(np.float64, copy=False) / self.voxel_size
        endpoint_grid = endpoint_xyz.astype(np.float64, copy=False) / self.voxel_size
        direction = endpoint_grid - origin_grid
        current = np.floor(origin_grid).astype(np.int32)
        target = np.floor(endpoint_grid).astype(np.int32)

        start_key = (int(current[0]), int(current[1]), int(current[2]))
        voxels = [start_key]
        if np.array_equal(current, target):
            return voxels

        step = np.sign(direction).astype(np.int32)
        t_max = np.empty((3,), dtype=np.float64)
        t_delta = np.empty((3,), dtype=np.float64)
        for axis in range(3):
            if abs(direction[axis]) < 1e-9:
                t_max[axis] = np.inf
                t_delta[axis] = np.inf
                continue
            next_boundary = float(np.floor(origin_grid[axis]) + (1 if step[axis] > 0 else 0))
            t_max[axis] = max((next_boundary - origin_grid[axis]) / direction[axis], 0.0)
            t_delta[axis] = 1.0 / abs(direction[axis])

        max_steps = int(np.ceil(np.linalg.norm(endpoint_xyz - origin_xyz) / self.voxel_size) * 3) + 4
        for _ in range(max_steps):
            if np.array_equal(current, target):
                break
            axis = int(np.argmin(t_max))
            current[axis] += step[axis]
            t_max[axis] += t_delta[axis]
            voxels.append((int(current[0]), int(current[1]), int(current[2])))
            if np.array_equal(current, target):
                break
        return voxels

    def integrate_rays(
        self,
        origin_world: np.ndarray,
        endpoints_world: np.ndarray,
        colors_rgb_u8: np.ndarray,
    ) -> dict[str, int]:
        accepted_hits = 0
        blocked_rays = 0
        decayed_voxels = 0

        origin_world = origin_world.astype(np.float64, copy=False)
        for endpoint_world, color_rgb_u8 in zip(endpoints_world, colors_rgb_u8, strict=False):
            traversed_voxels = self._trace_voxels(origin_world, endpoint_world.astype(np.float64, copy=False))
            if not traversed_voxels:
                continue

            if len(traversed_voxels) == 1:
                self._register_hit(traversed_voxels[0], color_rgb_u8)
                accepted_hits += 1
                continue

            occupied_run: list[tuple[int, int, int]] = []
            wall_voxel: tuple[int, int, int] | None = None
            for voxel_key in traversed_voxels[1:-1]:
                score = self._scores.get(voxel_key, 0.0)
                if score >= self.occupied_threshold:
                    occupied_run.append(voxel_key)
                    if len(occupied_run) > self.penetration_tolerance_voxels:
                        wall_voxel = voxel_key
                        break
                else:
                    for prior_key in occupied_run:
                        self._decrease_score(prior_key, self.pass_through_decay)
                        decayed_voxels += 1
                    occupied_run.clear()

            if wall_voxel is not None:
                blocked_rays += 1
                self._increase_score(wall_voxel, self.blocked_wall_score_delta)
                continue

            for prior_key in occupied_run:
                self._decrease_score(prior_key, self.pass_through_decay)
                decayed_voxels += 1

            endpoint_key = traversed_voxels[-1]
            self._register_hit(endpoint_key, color_rgb_u8)
            accepted_hits += 1

        self.integrated_frames += 1
        if self.max_voxels > 0 and len(self._scores) > self.max_voxels:
            self._prune()

        return {
            "accepted_hits": accepted_hits,
            "blocked_rays": blocked_rays,
            "decayed_voxels": decayed_voxels,
        }

    def _prune(self) -> None:
        scored_keys = sorted(
            self._scores.keys(),
            key=lambda key: (self._scores[key], self._hit_counts.get(key, 0)),
            reverse=True,
        )
        keep = set(scored_keys[: self.max_voxels])
        self._scores = {key: self._scores[key] for key in keep}
        self._color_sums = {key: value for key, value in self._color_sums.items() if key in keep}
        self._hit_counts = {key: value for key, value in self._hit_counts.items() if key in keep}

    def snapshot(
        self,
        point_budget: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        occupied_keys = [
            key
            for key, score in self._scores.items()
            if score >= self.occupied_threshold and key not in self._clearance_voxels
        ]
        if not occupied_keys:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        voxel_keys = np.asarray(occupied_keys, dtype=np.int32)
        scores = np.asarray([self._scores[key] for key in occupied_keys], dtype=np.float32)
        hit_counts = np.asarray([self._hit_counts.get(key, 0) for key in occupied_keys], dtype=np.int32)
        colors_rgb = np.zeros((len(occupied_keys), 3), dtype=np.uint8)
        for index, key in enumerate(occupied_keys):
            count = max(self._hit_counts.get(key, 0), 1)
            color_sum = self._color_sums.get(key)
            if color_sum is None:
                colors_rgb[index] = np.array([220, 220, 220], dtype=np.uint8)
            else:
                colors_rgb[index] = np.clip(color_sum / count, 0.0, 255.0).astype(np.uint8)

        if point_budget is not None and point_budget > 0 and voxel_keys.shape[0] > point_budget:
            ordering = np.lexsort((voxel_keys[:, 2], voxel_keys[:, 1], voxel_keys[:, 0]))
            selected = ordering[np.linspace(0, ordering.shape[0] - 1, point_budget, dtype=np.int32)]
            voxel_keys = voxel_keys[selected]
            scores = scores[selected]
            hit_counts = hit_counts[selected]
            colors_rgb = colors_rgb[selected]

        points_xyz = ((voxel_keys.astype(np.float32) + 0.5) * self.voxel_size).astype(np.float32, copy=False)
        return points_xyz, colors_rgb, scores, hit_counts

    def export_live(self, live_cloud_path: Path, point_budget: int) -> int:
        points_xyz, colors_rgb_u8, _, _ = self.snapshot(point_budget=point_budget)
        write_colored_xyz(points_xyz, colors_rgb_u8, live_cloud_path)
        return int(points_xyz.shape[0])

    def export_final(
        self,
        pointcloud_path: Path,
        metadata_path: Path,
        point_budget: int,
        mesh_path: Path | None = None,
    ) -> int:
        points_xyz, colors_rgb_u8, scores, hit_counts = self.snapshot(point_budget=point_budget)
        write_point_cloud_ply(points_xyz, colors_rgb_u8, pointcloud_path)
        np.savez_compressed(
            metadata_path,
            points_xyz=points_xyz,
            colors_rgb_u8=colors_rgb_u8,
            scores=scores,
            hit_counts=hit_counts,
            voxel_size=np.array(self.voxel_size, dtype=np.float32),
            penetration_tolerance_voxels=np.array(self.penetration_tolerance_voxels, dtype=np.int32),
            camera_clearance_radius_m=np.array(self.camera_clearance_radius_m, dtype=np.float32),
            camera_clearance_below_m=np.array(self.camera_clearance_below_m, dtype=np.float32),
            integrated_frames=np.array(self.integrated_frames, dtype=np.int32),
        )
        return int(points_xyz.shape[0])


class TSDFFusedOccupancy:
    def __init__(
        self,
        voxel_size: float,
        sdf_trunc_m: float,
        max_depth_m: float,
        max_voxels: int,
        camera_clearance_radius_m: float,
        camera_clearance_below_m: float,
    ) -> None:
        import open3d as o3d

        self.o3d = o3d
        self.voxel_size = float(voxel_size)
        self.sdf_trunc_m = float(max(sdf_trunc_m, self.voxel_size * 1.1))
        self.max_depth_m = float(max_depth_m)
        self.max_voxels = max(0, int(max_voxels))
        self.camera_clearance_radius_m = float(max(camera_clearance_radius_m, 0.0))
        self.camera_clearance_below_m = float(max(camera_clearance_below_m, 0.0))
        self.integrated_frames = 0
        self.volume = self.o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc_m,
            color_type=self.o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self._dirty = True
        self._cached_points = np.empty((0, 3), dtype=np.float32)
        self._cached_colors = np.empty((0, 3), dtype=np.uint8)
        self._cached_counts = np.empty((0,), dtype=np.int32)
        self._cached_mesh_vertices = 0
        self._camera_positions: list[np.ndarray] = []
        self._clearance_voxels: set[tuple[int, int, int]] = set()

    def _update_camera_clearance(self, camera_world: np.ndarray) -> None:
        if self.camera_clearance_radius_m <= 0.0:
            return
        previous = self._camera_positions[-1] if self._camera_positions else None
        current = camera_world.astype(np.float32, copy=False)
        self._camera_positions.append(current.copy())
        vertical_axis = infer_vertical_axis_from_positions(self._camera_positions)
        self._clearance_voxels.update(
            clearance_voxels_for_segment(
                previous,
                current,
                self.voxel_size,
                self.camera_clearance_radius_m,
                self.camera_clearance_below_m,
                vertical_axis,
            )
        )

    def update_camera_pose(self, pose_cw: np.ndarray) -> int:
        before = len(self._clearance_voxels)
        self._update_camera_clearance(camera_position_from_pose_cw(pose_cw))
        self._dirty = True
        return len(self._clearance_voxels) - before

    def integrate_depth(
        self,
        depth_map: np.ndarray,
        image_rgb: np.ndarray,
        intrinsics: np.ndarray,
        pose_cw: np.ndarray,
    ) -> None:
        height, width = depth_map.shape[:2]
        depth_o3d = self.o3d.geometry.Image(np.ascontiguousarray(depth_map.astype(np.float32, copy=False)))
        color_o3d = self.o3d.geometry.Image(np.ascontiguousarray(image_rgb.astype(np.uint8, copy=False)))
        rgbd = self.o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=self.max_depth_m,
            convert_rgb_to_intensity=False,
        )
        intrinsics_o3d = self.o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(intrinsics[0, 0]),
            float(intrinsics[1, 1]),
            float(intrinsics[0, 2]),
            float(intrinsics[1, 2]),
        )
        self.volume.integrate(
            rgbd,
            intrinsics_o3d,
            np.ascontiguousarray(pose_cw.astype(np.float64, copy=False)),
        )
        self.integrated_frames += 1
        self._dirty = True

    def _refresh_cache(self) -> None:
        if not self._dirty:
            return

        mesh = self.volume.extract_triangle_mesh()
        self._cached_mesh_vertices = int(len(mesh.vertices))
        if self._cached_mesh_vertices == 0:
            self._cached_points = np.empty((0, 3), dtype=np.float32)
            self._cached_colors = np.empty((0, 3), dtype=np.uint8)
            self._cached_counts = np.empty((0,), dtype=np.int32)
            self._dirty = False
            return

        sample_count = min(max(50_000, self._cached_mesh_vertices * 3), 250_000)
        point_cloud = mesh.sample_points_uniformly(number_of_points=sample_count)
        points_xyz = np.asarray(point_cloud.points, dtype=np.float32)
        if points_xyz.size == 0:
            self._cached_points = np.empty((0, 3), dtype=np.float32)
            self._cached_colors = np.empty((0, 3), dtype=np.uint8)
            self._cached_counts = np.empty((0,), dtype=np.int32)
            self._dirty = False
            return

        if point_cloud.has_colors():
            colors_rgb = np.clip(np.asarray(point_cloud.colors), 0.0, 1.0)
            colors_rgb_u8 = (colors_rgb * 255.0).astype(np.uint8)
        else:
            colors_rgb_u8 = np.full((points_xyz.shape[0], 3), 200, dtype=np.uint8)

        voxel_keys = np.floor(points_xyz / self.voxel_size).astype(np.int32, copy=False)
        unique_voxels, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
        counts = np.bincount(inverse).astype(np.int32, copy=False)
        color_sums = np.zeros((unique_voxels.shape[0], 3), dtype=np.float64)
        np.add.at(color_sums, inverse, colors_rgb_u8.astype(np.float64, copy=False))
        colors = np.clip(color_sums / counts[:, None], 0.0, 255.0).astype(np.uint8)

        if self._clearance_voxels:
            keep_mask = np.fromiter(
                (
                    (int(key[0]), int(key[1]), int(key[2])) not in self._clearance_voxels
                    for key in unique_voxels
                ),
                dtype=bool,
                count=unique_voxels.shape[0],
            )
            unique_voxels = unique_voxels[keep_mask]
            counts = counts[keep_mask]
            colors = colors[keep_mask]

        if self.max_voxels > 0 and unique_voxels.shape[0] > self.max_voxels:
            keep = np.argsort(counts)[::-1][: self.max_voxels]
            unique_voxels = unique_voxels[keep]
            counts = counts[keep]
            colors = colors[keep]

        self._cached_points = ((unique_voxels.astype(np.float32) + 0.5) * self.voxel_size).astype(
            np.float32,
            copy=False,
        )
        self._cached_colors = colors
        self._cached_counts = counts
        self._dirty = False

    def num_occupied_voxels(self) -> int:
        self._refresh_cache()
        return int(self._cached_points.shape[0])

    def mesh_vertex_count(self) -> int:
        self._refresh_cache()
        return int(self._cached_mesh_vertices)

    def snapshot(
        self,
        point_budget: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._refresh_cache()
        if self._cached_points.size == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int32),
            )

        if point_budget is None or point_budget <= 0 or self._cached_points.shape[0] <= point_budget:
            return (
                self._cached_points.copy(),
                self._cached_colors.copy(),
                self._cached_counts.copy(),
            )

        ordering = np.lexsort(
            (
                np.floor(self._cached_points[:, 2] / self.voxel_size).astype(np.int32),
                np.floor(self._cached_points[:, 1] / self.voxel_size).astype(np.int32),
                np.floor(self._cached_points[:, 0] / self.voxel_size).astype(np.int32),
            )
        )
        selected = ordering[np.linspace(0, ordering.shape[0] - 1, point_budget, dtype=np.int32)]
        return (
            self._cached_points[selected].copy(),
            self._cached_colors[selected].copy(),
            self._cached_counts[selected].copy(),
        )

    def export_live(self, live_cloud_path: Path, point_budget: int) -> int:
        points_xyz, colors_rgb_u8, _ = self.snapshot(point_budget=point_budget)
        write_colored_xyz(points_xyz, colors_rgb_u8, live_cloud_path)
        return int(points_xyz.shape[0])

    def export_final(
        self,
        pointcloud_path: Path,
        metadata_path: Path,
        point_budget: int,
        mesh_path: Path | None = None,
    ) -> int:
        mesh = self.volume.extract_triangle_mesh()
        if mesh_path is not None:
            if len(mesh.vertices) > 0:
                self.o3d.io.write_triangle_mesh(
                    str(mesh_path),
                    mesh,
                    write_ascii=True,
                    write_vertex_colors=True,
                )
            elif mesh_path.exists():
                mesh_path.unlink()

        points_xyz, colors_rgb_u8, counts = self.snapshot(point_budget=point_budget)
        write_point_cloud_ply(points_xyz, colors_rgb_u8, pointcloud_path)
        np.savez_compressed(
            metadata_path,
            points_xyz=points_xyz,
            colors_rgb_u8=colors_rgb_u8,
            hit_counts=counts,
            voxel_size=np.array(self.voxel_size, dtype=np.float32),
            tsdf_sdf_trunc_m=np.array(self.sdf_trunc_m, dtype=np.float32),
            camera_clearance_radius_m=np.array(self.camera_clearance_radius_m, dtype=np.float32),
            camera_clearance_below_m=np.array(self.camera_clearance_below_m, dtype=np.float32),
            integrated_frames=np.array(self.integrated_frames, dtype=np.int32),
        )
        return int(points_xyz.shape[0])


def depth_backend_entry(
    config: DepthWorkerConfig,
    task_queue: mp.Queue,
    update_queue: mp.Queue,
) -> None:
    try:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        live_cloud_path = output_dir / "occupancy_live.xyz"
        backend_status_path = output_dir / "backend_status.txt"
        pointcloud_path = output_dir / "occupancy_map.ply"
        occupancy_path = output_dir / "occupancy_map.npz"
        mesh_path = output_dir / "tsdf_mesh.ply"

        device = torch.device(config.device)
        if device.type == "cuda":
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
        else:
            device_name = str(device)

        processed_frames = 0
        last_scale: float | None = None
        last_scale_correspondences = 0
        last_scale_source = "none"
        last_status = "loading_model"
        last_dense_points = 0
        last_occupancy_voxels = 0
        last_exported_voxels = 0
        last_blocked_rays = 0
        last_mesh_vertices = 0
        last_live_export_time = 0.0
        depth_backend = str(config.depth_backend)
        pose_conditioned_enabled = bool(config.pose_conditioned and depth_backend == "da3")
        fusion_mode = str(config.fusion_mode)

        def emit_update(preview_bgr: np.ndarray | None = None, message: str | None = None) -> None:
            stats = {
                "depth_backend": depth_backend,
                "fusion_mode": fusion_mode,
                "processed_frames": processed_frames,
                "integrated_frames": fusion.integrated_frames,
                "occupancy_voxels": last_occupancy_voxels,
                "exported_voxels": last_exported_voxels,
                "blocked_rays": last_blocked_rays,
                "mesh_vertices": last_mesh_vertices,
                "last_dense_points": last_dense_points,
                "last_scale_correspondences": last_scale_correspondences,
                "last_scale": last_scale,
                "last_scale_source": last_scale_source,
                "last_status": last_status,
                "device": device_name,
                "pose_conditioned": pose_conditioned_enabled,
                "context_keyframes": config.context_keyframes,
            }
            update_queue.put(
                DepthWorkerUpdate(
                    kind="error" if message is not None else "status",
                    stats=stats,
                    preview_bgr=preview_bgr,
                    message=message,
                )
            )

        if fusion_mode == "tsdf_voxel":
            fusion = TSDFFusedOccupancy(
                voxel_size=config.voxel_size,
                sdf_trunc_m=config.tsdf_sdf_trunc_m,
                max_depth_m=config.max_depth_m,
                max_voxels=config.max_map_voxels,
                camera_clearance_radius_m=config.camera_clearance_radius_m,
                camera_clearance_below_m=config.camera_clearance_below_m,
            )
        else:
            fusion = RaycastOccupancyFusion(
                voxel_size=config.voxel_size,
                penetration_tolerance_voxels=config.penetration_tolerance_voxels,
                max_voxels=config.max_map_voxels,
                camera_clearance_radius_m=config.camera_clearance_radius_m,
                camera_clearance_below_m=config.camera_clearance_below_m,
            )
        camera_model = load_fisheye_camera_model(Path(config.settings_path))
        undistorter = FisheyeUndistorter(camera_model, balance=config.undistort_balance)

        emit_update(
            preview_bgr=make_status_preview(
                [
                    f"Loading depth model: {config.model_name}",
                    f"backend={depth_backend}",
                    f"device={device_name}",
                    f"fusion_mode={fusion_mode}",
                    f"pose_conditioned_request={int(config.pose_conditioned)} window={config.context_keyframes}",
                ]
            )
        )
        write_status_file(
            backend_status_path,
            {
                "frame_index": 0,
                "status": last_status,
                "depth_backend": depth_backend,
                "fusion_mode": fusion_mode,
                "processed_frames": 0,
                "integrated_frames": 0,
                "occupancy_voxels": 0,
                "exported_voxels": 0,
                "mesh_vertices": 0,
                "device": device_name,
            },
        )

        model = None
        if depth_backend == "unidepth_v2":
            model = load_unidepth_model(
                config.model_name,
                device,
                Path(config.unidepth_src),
                config.unidepth_resolution_level,
            )
            print(
                f"[UniDepth] model={config.model_name} device={device_name} "
                f"resolution_level={config.unidepth_resolution_level}"
            )
        elif depth_backend == "da3":
            model = DepthAnything3.from_pretrained(config.model_name).to(device)
            original_align_to_input = model._align_to_input_extrinsics_intrinsics

            # In this pipeline, ORB-SLAM3 already provides the poses we trust.
            # DA3's default Umeyama alignment can fail on short/degenerate motion windows,
            # so bypass it and keep the input poses as-is whenever poses are provided.
            def passthrough_input_pose_alignment(
                extrinsics,
                intrinsics,
                prediction,
                align_to_input_ext_scale=True,
                ransac_view_thresh=10,
            ):
                if extrinsics is None:
                    return original_align_to_input(
                        extrinsics,
                        intrinsics,
                        prediction,
                        align_to_input_ext_scale=align_to_input_ext_scale,
                        ransac_view_thresh=ransac_view_thresh,
                    )
                if intrinsics is not None:
                    prediction.intrinsics = intrinsics.numpy()
                prediction.extrinsics = extrinsics[..., :3, :].numpy()
                return prediction

            model._align_to_input_extrinsics_intrinsics = passthrough_input_pose_alignment
            model_supports_pose_conditioning = (
                getattr(model.model, "cam_enc", None) is not None
                and getattr(model.model, "cam_dec", None) is not None
            )
            if pose_conditioned_enabled and not model_supports_pose_conditioning:
                pose_conditioned_enabled = False
                warning_message = (
                    f"[DA3] model={config.model_name} does not support camera pose conditioning; "
                    "falling back to single-frame depth"
                )
                print(warning_message)
                last_status = "pose_conditioning_unsupported"
                emit_update(
                    preview_bgr=make_status_preview(
                        [
                            f"DA3 model loaded: {config.model_name}",
                            "Pose conditioning disabled for this checkpoint",
                            "Continuing with single-frame depth inference",
                        ]
                    )
                )
            print(
                f"[DA3] model={config.model_name} device={device_name} "
                f"pose_conditioned={int(pose_conditioned_enabled)} "
                f"context_keyframes={config.context_keyframes}"
            )

            input_processor = model.input_processor

            def single_worker_input_processor(
                image,
                extrinsics=None,
                intrinsics=None,
                process_res=504,
                process_res_method="upper_bound_resize",
            ):
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
        else:
            raise RuntimeError(f"Unsupported depth backend: {depth_backend}")

        buffered_packets: list[ProcessedFramePacket] = []
        last_status = "idle"

        def export_live_surface(force: bool = False) -> None:
            nonlocal last_live_export_time
            nonlocal last_occupancy_voxels
            nonlocal last_exported_voxels
            nonlocal last_mesh_vertices

            now = time.perf_counter()
            if not force and config.live_export_interval_sec > 0.0:
                if (now - last_live_export_time) < config.live_export_interval_sec:
                    return

            last_exported_voxels = fusion.export_live(
                live_cloud_path,
                config.live_point_budget,
            )
            last_occupancy_voxels = fusion.num_occupied_voxels()
            if hasattr(fusion, "mesh_vertex_count"):
                last_mesh_vertices = int(fusion.mesh_vertex_count())
            else:
                last_mesh_vertices = 0
            last_live_export_time = now

        emit_update()

        frame_log_path = output_dir / "dense_frames.csv"
        with frame_log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "frame_index",
                    "timestamp_ns",
                    "status",
                    "pose_conditioned_used",
                    "window_size",
                    "scale",
                    "scale_source",
                    "scale_correspondences",
                    "sampled_rays",
                    "integrated_frames",
                    "occupancy_voxels",
                    "exported_voxels",
                    "blocked_rays",
                    "prediction_is_metric",
                ]
            )

            while True:
                task = task_queue.get()
                if task is None:
                    break

                raw_rgb = read_rgb_image(Path(task.image_path))
                undistorted_rgb = undistorter.undistort_image(raw_rgb).astype(np.uint8, copy=False)
                undistorted_keypoints = undistorter.undistort_points(task.sparse_keypoints_uv)
                packet = ProcessedFramePacket(
                    frame_index=task.frame_index,
                    timestamp_ns=task.timestamp_ns,
                    image_rgb=undistorted_rgb,
                    pose_cw=task.pose_cw.astype(np.float32, copy=False),
                    intrinsics=undistorter.output_intrinsics.astype(np.float32, copy=False),
                    sparse_keypoints_uv=undistorted_keypoints,
                    sparse_world_points_xyz=task.sparse_world_points_xyz.astype(np.float32, copy=False),
                )

                buffered_packets.append(packet)
                if len(buffered_packets) > config.context_keyframes:
                    buffered_packets = buffered_packets[-config.context_keyframes :]

                pose_conditioned_used = False
                inference_packets = [packet]
                processed_image = packet.image_rgb
                processed_intrinsics = packet.intrinsics
                sky_map = None
                prediction_is_metric = True

                if depth_backend == "da3":
                    pose_conditioned_used = pose_conditioned_enabled and len(buffered_packets) >= 2
                    inference_packets = buffered_packets if pose_conditioned_used else [packet]
                    inference_kwargs: dict[str, object] = {
                        "image": [item.image_rgb for item in inference_packets],
                        "intrinsics": np.asarray([item.intrinsics for item in inference_packets], dtype=np.float32),
                        "process_res": config.process_res,
                        "process_res_method": config.process_res_method,
                    }
                    if pose_conditioned_used:
                        inference_kwargs["extrinsics"] = np.asarray(
                            [item.pose_cw for item in inference_packets],
                            dtype=np.float32,
                        )
                        inference_kwargs["align_to_input_ext_scale"] = True
                        inference_kwargs["ref_view_strategy"] = "middle"

                    try:
                        prediction = model.inference(**inference_kwargs)
                    except Exception as exc:
                        pose_conditioning_failed = (
                            pose_conditioned_used
                            and (
                                ("NoneType" in str(exc) and "callable" in str(exc))
                                or ("Degenerate covariance rank" in str(exc))
                                or ("Umeyama alignment is not possible" in str(exc))
                            )
                        )
                        if pose_conditioning_failed:
                            pose_conditioned_enabled = False
                            pose_conditioned_used = False
                            inference_packets = [packet]
                            inference_kwargs = {
                                "image": [packet.image_rgb],
                                "intrinsics": np.asarray([packet.intrinsics], dtype=np.float32),
                                "process_res": config.process_res,
                                "process_res_method": config.process_res_method,
                            }
                            last_status = "pose_conditioning_unsupported"
                            print(
                                f"[DA3] disabling pose conditioning after model failure on frame {packet.frame_index}: {exc}"
                            )
                            emit_update(
                                preview_bgr=make_status_preview(
                                    [
                                        "Pose conditioning failed for this DA3 checkpoint",
                                        "Falling back to single-frame inference",
                                        f"frame={packet.frame_index}",
                                    ]
                                )
                            )
                            prediction = model.inference(**inference_kwargs)
                        else:
                            raise

                    processed_image = np.asarray(
                        prediction.processed_images[-1]
                        if prediction.processed_images is not None
                        else inference_packets[-1].image_rgb,
                        dtype=np.uint8,
                    )
                    processed_intrinsics = np.asarray(
                        prediction.intrinsics[-1]
                        if prediction.intrinsics is not None
                        else inference_packets[-1].intrinsics,
                        dtype=np.float32,
                    )
                    depth_map = np.asarray(prediction.depth[-1], dtype=np.float32)
                    conf_map = (
                        None if prediction.conf is None else np.asarray(prediction.conf[-1], dtype=np.float32)
                    )
                    sky_map = (
                        None if prediction.sky is None else np.asarray(prediction.sky[-1], dtype=np.float32)
                    )
                    prediction_is_metric = bool(prediction.is_metric)
                else:
                    processed_image, processed_intrinsics, depth_map, conf_map = predict_unidepth(
                        packet.image_rgb,
                        packet.intrinsics,
                        model,
                        device,
                    )

                sparse_uv_processed = remap_pixels_between_intrinsics(
                    packet.sparse_keypoints_uv,
                    packet.intrinsics,
                    processed_intrinsics,
                )
                scale, num_scale_corr = estimate_depth_scale(
                    sparse_uv_processed,
                    packet.sparse_world_points_xyz,
                    packet.pose_cw,
                    depth_map,
                    config.min_scale_correspondences,
                )
                scale_source = "sparse_correspondence"
                if scale is None:
                    if prediction_is_metric:
                        scale = 1.0
                        scale_source = "metric_default"
                    elif last_scale is not None:
                        scale = last_scale
                        scale_source = "last_valid"

                processed_frames += 1
                last_scale_correspondences = int(num_scale_corr)
                last_scale_source = scale_source
                camera_cleared_voxels = 0
                if hasattr(fusion, "update_camera_pose"):
                    camera_cleared_voxels = int(fusion.update_camera_pose(packet.pose_cw))

                if scale is None or not np.isfinite(scale) or scale <= 0.0:
                    last_dense_points = 0
                    last_status = "skipped_no_scale"
                    if camera_cleared_voxels > 0:
                        export_live_surface(force=(processed_frames == 1))
                    preview = make_depth_preview(
                        processed_image,
                        depth_map,
                        (
                            f"frame={packet.frame_index} scale=missing "
                            f"sparse={num_scale_corr} pose={int(pose_conditioned_used)} "
                            f"window={len(inference_packets)} status=skipped"
                        ),
                    )
                    writer.writerow(
                        [
                            packet.frame_index,
                            packet.timestamp_ns,
                            last_status,
                            int(pose_conditioned_used),
                            len(inference_packets),
                            "",
                            "missing",
                            num_scale_corr,
                            0,
                            fusion.integrated_frames,
                            last_occupancy_voxels,
                            last_exported_voxels,
                            0,
                            int(prediction_is_metric),
                        ]
                    )
                    handle.flush()
                    write_status_file(
                        backend_status_path,
                        {
                            "frame_index": packet.frame_index,
                            "status": last_status,
                            "depth_backend": depth_backend,
                            "fusion_mode": fusion_mode,
                            "processed_frames": processed_frames,
                            "integrated_frames": fusion.integrated_frames,
                            "occupancy_voxels": last_occupancy_voxels,
                            "exported_voxels": last_exported_voxels,
                            "mesh_vertices": last_mesh_vertices,
                            "scale_source": "missing",
                            "device": device_name,
                        },
                    )
                    emit_update(preview_bgr=preview)
                    continue

                last_scale = float(scale)
                scaled_depth = depth_map * float(scale)
                filtered_depth, _ = filter_depth_for_integration(
                    scaled_depth,
                    config.min_depth_m,
                    config.max_depth_m,
                    conf_map,
                    config.conf_percentile,
                    sky_map,
                    config.sky_threshold,
                )
                if fusion_mode == "raycast":
                    origin_world, endpoints_world, sampled_colors_rgb = build_raycast_samples(
                        filtered_depth,
                        processed_image,
                        processed_intrinsics,
                        packet.pose_cw,
                        config.reprojection_stride,
                        config.min_depth_m,
                        config.max_depth_m,
                        None,
                        config.conf_percentile,
                        None,
                        config.sky_threshold,
                    )
                    last_dense_points = int(endpoints_world.shape[0])
                else:
                    last_dense_points = count_valid_depth_samples(
                        filtered_depth,
                        config.reprojection_stride,
                        config.min_depth_m,
                        config.max_depth_m,
                        None,
                        config.conf_percentile,
                        None,
                        config.sky_threshold,
                    )
                if last_dense_points > 0:
                    if fusion_mode == "raycast":
                        integration_stats = fusion.integrate_rays(
                            origin_world,
                            endpoints_world,
                            sampled_colors_rgb,
                        )
                        last_blocked_rays = int(integration_stats["blocked_rays"])
                    else:
                        fusion.integrate_depth(
                            filtered_depth,
                            processed_image,
                            processed_intrinsics,
                            packet.pose_cw,
                        )
                        last_blocked_rays = 0
                    export_live_surface(force=(processed_frames == 1))
                    last_status = "ok"
                else:
                    last_status = "skipped_empty_depth"
                    last_blocked_rays = 0
                    if camera_cleared_voxels > 0:
                        export_live_surface(force=(processed_frames == 1))

                preview = make_depth_preview(
                    processed_image,
                    filtered_depth,
                    (
                        f"frame={packet.frame_index} scale={scale:.3f} "
                        f"src={scale_source} sparse={num_scale_corr} "
                        f"pose={int(pose_conditioned_used)} window={len(inference_packets)} "
                        f"samples={last_dense_points} occ={last_occupancy_voxels} "
                        f"blocked={last_blocked_rays} mode={fusion_mode}"
                    ),
                )
                writer.writerow(
                    [
                        packet.frame_index,
                        packet.timestamp_ns,
                        last_status,
                        int(pose_conditioned_used),
                        len(inference_packets),
                        f"{scale:.6f}",
                        scale_source,
                        num_scale_corr,
                        last_dense_points,
                        fusion.integrated_frames,
                        last_occupancy_voxels,
                        last_exported_voxels,
                        last_blocked_rays,
                        int(prediction_is_metric),
                    ]
                )
                handle.flush()
                write_status_file(
                        backend_status_path,
                        {
                            "frame_index": packet.frame_index,
                            "status": last_status,
                            "depth_backend": depth_backend,
                            "fusion_mode": fusion_mode,
                            "processed_frames": processed_frames,
                            "integrated_frames": fusion.integrated_frames,
                            "occupancy_voxels": last_occupancy_voxels,
                            "exported_voxels": last_exported_voxels,
                            "sampled_rays": last_dense_points,
                            "blocked_rays": last_blocked_rays,
                            "mesh_vertices": last_mesh_vertices,
                            "scale": f"{scale:.6f}",
                            "scale_source": scale_source,
                            "device": device_name,
                        },
                    )
                emit_update(preview_bgr=preview)

            export_live_surface(force=True)
            last_exported_voxels = fusion.export_final(
                pointcloud_path,
                occupancy_path,
                config.final_point_budget,
                mesh_path=mesh_path if fusion_mode == "tsdf_voxel" else None,
            )
            last_occupancy_voxels = fusion.num_occupied_voxels()
            if hasattr(fusion, "mesh_vertex_count"):
                last_mesh_vertices = int(fusion.mesh_vertex_count())
            else:
                last_mesh_vertices = 0
            write_status_file(
                backend_status_path,
                {
                    "frame_index": processed_frames,
                    "status": "finished",
                    "depth_backend": depth_backend,
                    "fusion_mode": fusion_mode,
                    "processed_frames": processed_frames,
                    "integrated_frames": fusion.integrated_frames,
                    "occupancy_voxels": last_occupancy_voxels,
                    "exported_voxels": last_exported_voxels,
                    "mesh_vertices": last_mesh_vertices,
                    "device": device_name,
                },
            )
            emit_update()
    except Exception as exc:  # pragma: no cover
        emit_message = "".join(traceback.format_exception(exc))
        try:
            update_queue.put(DepthWorkerUpdate(kind="error", stats={}, message=emit_message))
        except Exception:
            pass
        raise


class DepthEstimationService:
    def __init__(
        self,
        depth_backend: str,
        fusion_mode: str,
        model_name: str,
        unidepth_src: Path,
        unidepth_resolution_level: int,
        device: str,
        output_dir: Path,
        settings_path: Path,
        undistort_balance: float,
        process_res: int,
        process_res_method: str,
        queue_size: int,
        pose_conditioned: bool,
        context_keyframes: int,
        reprojection_stride: int,
        min_depth_m: float,
        max_depth_m: float,
        conf_percentile: float,
        sky_threshold: float,
        min_scale_correspondences: int,
        voxel_size: float,
        tsdf_sdf_trunc_m: float,
        penetration_tolerance_voxels: int,
        camera_clearance_radius_m: float,
        camera_clearance_below_m: float,
        max_map_voxels: int,
        live_export_interval_sec: float,
        live_point_budget: int,
        final_point_budget: int,
    ) -> None:
        self.output_dir = output_dir
        self.ctx = mp.get_context("spawn")
        self.task_queue: mp.Queue = self.ctx.Queue(maxsize=max(1, queue_size))
        self.update_queue: mp.Queue = self.ctx.Queue()
        self.process: mp.Process | None = None
        self.latest_preview: np.ndarray | None = None
        self.error_message: str | None = None
        self.stats: dict[str, int | float | str | None] = {
            "depth_backend": str(depth_backend),
            "fusion_mode": str(fusion_mode),
            "processed_frames": 0,
            "dropped_frames": 0,
            "integrated_frames": 0,
            "occupancy_voxels": 0,
            "exported_voxels": 0,
            "blocked_rays": 0,
            "mesh_vertices": 0,
            "last_dense_points": 0,
            "last_scale_correspondences": 0,
            "last_scale": None,
            "last_scale_source": "none",
            "last_status": "idle",
            "device": None,
            "pose_conditioned": bool(pose_conditioned),
            "context_keyframes": int(context_keyframes),
        }
        self.config = DepthWorkerConfig(
            depth_backend=str(depth_backend),
            fusion_mode=str(fusion_mode),
            model_name=model_name,
            unidepth_src=str(unidepth_src),
            unidepth_resolution_level=int(unidepth_resolution_level),
            device=device,
            output_dir=str(output_dir),
            settings_path=str(settings_path),
            undistort_balance=float(undistort_balance),
            process_res=int(process_res),
            process_res_method=process_res_method,
            pose_conditioned=bool(pose_conditioned),
            context_keyframes=max(1, int(context_keyframes)),
            reprojection_stride=max(1, int(reprojection_stride)),
            min_depth_m=float(min_depth_m),
            max_depth_m=float(max_depth_m),
            conf_percentile=float(conf_percentile),
            sky_threshold=float(sky_threshold),
            min_scale_correspondences=max(3, int(min_scale_correspondences)),
            voxel_size=float(voxel_size),
            tsdf_sdf_trunc_m=float(max(tsdf_sdf_trunc_m, voxel_size * 1.1)),
            penetration_tolerance_voxels=max(0, int(penetration_tolerance_voxels)),
            camera_clearance_radius_m=float(max(0.0, camera_clearance_radius_m)),
            camera_clearance_below_m=float(max(0.0, camera_clearance_below_m)),
            max_map_voxels=max(1_000, int(max_map_voxels)),
            live_export_interval_sec=float(max(0.0, live_export_interval_sec)),
            live_point_budget=max(500, int(live_point_budget)),
            final_point_budget=max(1_000, int(final_point_budget)),
        )

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.process = self.ctx.Process(
            target=depth_backend_entry,
            args=(self.config, self.task_queue, self.update_queue),
            name="depth-backend",
            daemon=True,
        )
        self.process.start()

    def submit(self, packet: FramePacket) -> bool:
        try:
            self.task_queue.put_nowait(packet)
            return True
        except queue.Full:
            pass

        try:
            self.task_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self.stats["dropped_frames"] = int(self.stats["dropped_frames"] or 0) + 1

        try:
            self.task_queue.put_nowait(packet)
            return True
        except queue.Full:
            self.stats["dropped_frames"] = int(self.stats["dropped_frames"] or 0) + 1
            return False

    def get_queue_size(self) -> int:
        try:
            return int(self.task_queue.qsize())
        except (NotImplementedError, AttributeError):
            return -1

    def poll_updates(self) -> None:
        while True:
            try:
                update: DepthWorkerUpdate = self.update_queue.get_nowait()
            except queue.Empty:
                break
            if update.kind == "error":
                self.error_message = update.message
            else:
                if update.preview_bgr is not None:
                    self.latest_preview = update.preview_bgr
                self.stats.update(update.stats)

    def get_latest_preview(self) -> np.ndarray | None:
        self.poll_updates()
        if self.latest_preview is None:
            return None
        return self.latest_preview.copy()

    def get_stats(self) -> dict[str, int | float | str | None]:
        self.poll_updates()
        return dict(self.stats)

    def finish(self) -> None:
        if self.process is not None and self.process.is_alive():
            while True:
                try:
                    self.task_queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    if self.process is None or not self.process.is_alive():
                        break
                    time.sleep(0.05)
            self.process.join()
        self.poll_updates()
        if self.error_message is not None:
            raise RuntimeError(self.error_message)
        if self.process is not None and self.process.exitcode not in (0, None):
            raise RuntimeError(f"Depth backend exited with code {self.process.exitcode}")

    def export_outputs(self) -> tuple[Path, Path, Path | None]:
        mesh_path = self.output_dir / "tsdf_mesh.ply"
        if self.config.fusion_mode != "tsdf_voxel":
            mesh_path = None
        return (
            self.output_dir / "occupancy_map.ply",
            self.output_dir / "occupancy_map.npz",
            mesh_path,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    orb_root = SCRIPT_DIR / "dependency" / "ORB_SLAM3"
    default_dataset = SCRIPT_DIR / "dataset" / "dataset-corridor1_512_16"

    parser = argparse.ArgumentParser(
        description=(
            "Run monocular-inertial ORB-SLAM3 on cam0, stream all tracked frames to a "
            "background depth worker, scale depth using ORB sparse "
            "correspondences, and fuse the result into an occupancy map using either "
            "TSDF voxelization or raycast occupancy."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset,
        help="Dataset root containing mav0/cam0 and optionally mav0/imu0.",
    )
    parser.add_argument(
        "--vocabulary-path",
        type=Path,
        default=orb_root / "Vocabulary" / "ORBvoc.txt",
    )
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=orb_root / "Examples" / "Monocular-Inertial" / "TUM-VI.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "orbslam3_da3_realtime",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--no-orb-viewer", action="store_true")
    parser.add_argument("--no-input-window", action="store_true")
    parser.add_argument("--no-depth-window", action="store_true")
    parser.add_argument("--no-map-window", action="store_true")
    parser.add_argument(
        "--fusion-mode",
        choices=("tsdf_voxel", "raycast"),
        default="tsdf_voxel",
        help="Dense fusion backend. 'tsdf_voxel' extracts a TSDF surface then voxelizes it; 'raycast' uses occupancy ray marching.",
    )
    parser.add_argument(
        "--depth-backend",
        choices=("unidepth_v2", "da3"),
        default="unidepth_v2",
        help="Depth model backend. UniDepthV2 is the default; DA3 remains available as a fallback.",
    )
    parser.add_argument(
        "--depth-model",
        default=None,
        help="Hugging Face repo or local model directory for the selected depth backend.",
    )
    parser.add_argument(
        "--da3-model",
        dest="depth_model",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--unidepth-src",
        type=Path,
        default=UNIDEPTH_SRC_DEFAULT,
        help="Path to the UniDepth source tree used for local imports.",
    )
    parser.add_argument(
        "--unidepth-resolution-level",
        type=int,
        default=0,
        help="UniDepthV2 resolution level in [0, 9]. Lower is faster.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--da3-pose-conditioned",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a rolling frame window with ORB poses/intrinsics for DA3 pose conditioning.",
    )
    parser.add_argument(
        "--da3-context-keyframes",
        "--da3-context-frames",
        type=int,
        dest="da3_context_keyframes",
        default=8,
        help="Number of recent frames kept in the DA3 conditioning window when enabled.",
    )
    parser.add_argument("--process-res", type=int, default=512)
    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=2,
        help="Latest-frame-wins backend queue size. Small values minimize frontend stalls.",
    )
    parser.add_argument(
        "--reprojection-stride",
        type=int,
        default=4,
        help="Pixel stride used to sample depth rays for occupancy integration.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="Occupancy voxel size in meters.",
    )
    parser.add_argument(
        "--tsdf-sdf-trunc-m",
        type=float,
        default=0.08,
        help="TSDF truncation band in meters when --fusion-mode=tsdf_voxel.",
    )
    parser.add_argument(
        "--penetration-tolerance-voxels",
        type=int,
        default=2,
        help="How many occupied voxels a ray may penetrate before being treated as blocked by a wall when --fusion-mode=raycast.",
    )
    parser.add_argument(
        "--camera-clearance-radius-m",
        type=float,
        default=0.30,
        help="Carve occupied voxels around the camera path within this radius to remove floating points. Set 0 to disable.",
    )
    parser.add_argument(
        "--camera-clearance-below-m",
        type=float,
        default=0.18,
        help="How far below the camera the carve reaches. Keep this small to preserve the floor.",
    )
    parser.add_argument(
        "--max-map-voxels",
        type=int,
        default=350_000,
        help="Maximum number of scored occupancy voxels kept in memory.",
    )
    parser.add_argument(
        "--live-export-interval-sec",
        type=float,
        default=0.75,
        help="Minimum interval between live occupancy exports for the viewer.",
    )
    parser.add_argument(
        "--live-point-budget",
        type=int,
        default=25_000,
        help="Maximum number of occupied voxels exported for the live viewer.",
    )
    parser.add_argument(
        "--final-point-budget",
        type=int,
        default=100_000,
        help="Maximum number of occupied voxels exported in the final point cloud.",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.25)
    parser.add_argument("--max-depth-m", type=float, default=20.0)
    parser.add_argument(
        "--conf-percentile",
        type=float,
        default=85.0,
        help="Keep only depth pixels at or above this model confidence percentile before fusion.",
    )
    parser.add_argument("--sky-threshold", type=float, default=0.30)
    parser.add_argument("--min-scale-correspondences", type=int, default=24)
    parser.add_argument("--undistort-balance", type=float, default=0.0)
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


def load_imu_stream(imu_dir: Path) -> list[ImuRecord]:
    data_csv = imu_dir / "data.csv"
    if not data_csv.exists():
        return []

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


def read_rgb_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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


def save_xyz(points_xyz: np.ndarray, output_path: Path) -> None:
    if points_xyz.size == 0:
        output_path.write_text("", encoding="utf-8")
        return
    np.savetxt(output_path, points_xyz, fmt="%.6f")


def write_point_cloud_ply(points_xyz: np.ndarray, colors_rgb_u8: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points_xyz.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points_xyz, colors_rgb_u8, strict=False):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_colored_xyz(points_xyz: np.ndarray, colors_rgb_u8: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if points_xyz.size == 0:
        output_path.write_text("", encoding="utf-8")
        return
    cloud = np.concatenate(
        [points_xyz.astype(np.float32, copy=False), colors_rgb_u8.astype(np.uint8, copy=False)],
        axis=1,
    )
    np.savetxt(
        output_path,
        cloud,
        fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"],
    )


def write_status_file(path: Path, values: dict[str, object]) -> None:
    lines = [f"{key}={value}" for key, value in values.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def add_overlay_lines(image_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    output = image_bgr.copy()
    for index, line in enumerate(lines):
        y = 28 + index * 24
        cv2.putText(
            output,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def make_depth_preview(image_rgb: np.ndarray, depth_map: np.ndarray, status_text: str) -> np.ndarray:
    depth_vis = depth_to_preview(depth_map)
    return add_overlay_lines(depth_vis, [status_text])


def make_status_preview(lines: list[str], height: int = 512, width: int = 1024) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    return add_overlay_lines(canvas, lines)


def filter_depth_for_integration(
    depth_map: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    conf_map: np.ndarray | None,
    conf_percentile: float,
    sky_map: np.ndarray | None,
    sky_threshold: float,
) -> tuple[np.ndarray, float | None]:
    filtered_depth = depth_map.astype(np.float32, copy=True)
    valid = np.isfinite(filtered_depth) & (filtered_depth >= min_depth_m) & (filtered_depth <= max_depth_m)

    conf_threshold: float | None = None
    if conf_map is not None:
        finite_conf = conf_map[np.isfinite(conf_map)]
        if finite_conf.size > 0:
            conf_threshold = float(np.percentile(finite_conf, np.clip(conf_percentile, 0.0, 100.0)))
            valid &= np.isfinite(conf_map) & (conf_map >= conf_threshold)

    if sky_map is not None:
        valid &= np.isfinite(sky_map) & (sky_map < sky_threshold)

    filtered_depth[~valid] = 0.0
    return filtered_depth, conf_threshold


def load_unidepth_model(
    model_name: str,
    device: torch.device,
    source_dir: Path,
    resolution_level: int,
):
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise RuntimeError(f"UniDepth source directory not found: {source_dir}")
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    try:
        from unidepth.models import UniDepthV2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import UniDepthV2. Install the missing UniDepth dependency "
            f"'{exc.name}' in the current environment."
        ) from exc

    model = UniDepthV2.from_pretrained(model_name).to(device)
    model.eval()
    model.resolution_level = int(np.clip(resolution_level, 0, 9))
    return model


def predict_unidepth(
    image_rgb: np.ndarray,
    intrinsics: np.ndarray,
    model,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    rgb = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).to(device)
    camera = torch.from_numpy(np.ascontiguousarray(intrinsics.astype(np.float32, copy=False))).to(device)

    with torch.inference_mode():
        outputs = model.infer(rgb, camera)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    depth_map = outputs["depth"].squeeze().detach().cpu().numpy().astype(np.float32)
    output_intrinsics = outputs.get("intrinsics")
    if output_intrinsics is None:
        processed_intrinsics = intrinsics.astype(np.float32, copy=True)
    else:
        processed_intrinsics = (
            output_intrinsics.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
        )

    confidence_tensor = outputs.get("confidence")
    conf_score = None
    if confidence_tensor is not None:
        confidence_error = confidence_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
        conf_score = np.full_like(confidence_error, np.nan, dtype=np.float32)
        valid = np.isfinite(confidence_error) & (confidence_error > 0.0)
        conf_score[valid] = 1.0 / np.maximum(confidence_error[valid], 1e-6)

    return image_rgb, processed_intrinsics, depth_map, conf_score


def make_input_preview(
    image_gray: np.ndarray,
    frame_index: int,
    timestamp_ns: int,
    tracking_state_name: str,
    pose_valid: bool,
    is_keyframe: bool,
    queued: int,
    submitted: int,
    dropped: int,
) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    return add_overlay_lines(
        image_bgr,
        [
            f"frame={frame_index} ts={timestamp_ns * 1e-9:.3f}s",
            f"state={tracking_state_name} pose={int(bool(pose_valid))} orb_keyframe={int(bool(is_keyframe))}",
            f"frames queued={queued} submitted={submitted} dropped={dropped}",
        ],
    )


def remap_pixels_between_intrinsics(
    pixels_uv: np.ndarray,
    source_intrinsics: np.ndarray,
    target_intrinsics: np.ndarray,
) -> np.ndarray:
    if pixels_uv.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    homogeneous = np.concatenate(
        [pixels_uv.astype(np.float32, copy=False), np.ones((pixels_uv.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    rays = (np.linalg.inv(source_intrinsics) @ homogeneous.T).T
    projected = (target_intrinsics @ rays.T).T
    return (projected[:, :2] / projected[:, 2:3]).astype(np.float32, copy=False)


def sample_depth_nearest(depth_map: np.ndarray, pixels_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pixels_uv.size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=bool)
    height, width = depth_map.shape[:2]
    u = np.rint(pixels_uv[:, 0]).astype(np.int32)
    v = np.rint(pixels_uv[:, 1]).astype(np.int32)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    samples = np.full((pixels_uv.shape[0],), np.nan, dtype=np.float32)
    samples[valid] = depth_map[v[valid], u[valid]]
    return samples, valid


def estimate_depth_scale(
    sparse_uv: np.ndarray,
    world_points_xyz: np.ndarray,
    pose_cw: np.ndarray,
    depth_map: np.ndarray,
    min_correspondences: int,
) -> tuple[float | None, int]:
    if sparse_uv.shape[0] == 0 or world_points_xyz.shape[0] == 0:
        return None, 0
    num_entries = min(sparse_uv.shape[0], world_points_xyz.shape[0])
    sparse_uv = sparse_uv[:num_entries]
    world_points_xyz = world_points_xyz[:num_entries]

    rotation_cw = pose_cw[:3, :3].astype(np.float64, copy=False)
    translation_cw = pose_cw[:3, 3].astype(np.float64, copy=False)
    points_cam = (rotation_cw @ world_points_xyz.T).T + translation_cw
    reference_depth = points_cam[:, 2].astype(np.float32, copy=False)

    predicted_depth, inside = sample_depth_nearest(depth_map, sparse_uv)
    valid = (
        inside
        & np.isfinite(reference_depth)
        & np.isfinite(predicted_depth)
        & (reference_depth > 1e-3)
        & (predicted_depth > 1e-3)
    )
    if int(valid.sum()) < min_correspondences:
        return None, int(valid.sum())

    ratios = reference_depth[valid] / predicted_depth[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 1e-3)]
    if ratios.size < min_correspondences:
        return None, int(ratios.size)

    median = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - median)))
    if mad > 1e-6:
        inlier_mask = np.abs(ratios - median) <= (3.5 * mad)
        filtered = ratios[inlier_mask]
    else:
        filtered = ratios

    if filtered.size < min_correspondences:
        return None, int(filtered.size)
    return float(np.median(filtered)), int(filtered.size)


def count_valid_depth_samples(
    depth_map: np.ndarray,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
    conf_map: np.ndarray | None,
    conf_percentile: float,
    sky_map: np.ndarray | None,
    sky_threshold: float,
) -> int:
    height, width = depth_map.shape[:2]
    rows = np.arange(0, height, stride, dtype=np.int32)
    cols = np.arange(0, width, stride, dtype=np.int32)
    grid_u, grid_v = np.meshgrid(cols, rows, indexing="xy")

    sampled_depth = depth_map[grid_v, grid_u]
    valid = np.isfinite(sampled_depth) & (sampled_depth >= min_depth_m) & (sampled_depth <= max_depth_m)

    if conf_map is not None:
        finite_conf = conf_map[np.isfinite(conf_map)]
        if finite_conf.size > 0:
            conf_threshold = float(np.percentile(finite_conf, conf_percentile))
            valid &= conf_map[grid_v, grid_u] >= conf_threshold

    if sky_map is not None:
        valid &= sky_map[grid_v, grid_u] < sky_threshold

    return int(np.count_nonzero(valid))


def build_raycast_samples(
    depth_map: np.ndarray,
    image_rgb: np.ndarray,
    intrinsics: np.ndarray,
    pose_cw: np.ndarray,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
    conf_map: np.ndarray | None,
    conf_percentile: float,
    sky_map: np.ndarray | None,
    sky_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = depth_map.shape[:2]
    rows = np.arange(0, height, stride, dtype=np.int32)
    cols = np.arange(0, width, stride, dtype=np.int32)
    grid_u, grid_v = np.meshgrid(cols, rows, indexing="xy")

    sampled_depth = depth_map[grid_v, grid_u]
    valid = np.isfinite(sampled_depth) & (sampled_depth >= min_depth_m) & (sampled_depth <= max_depth_m)

    if conf_map is not None:
        finite_conf = conf_map[np.isfinite(conf_map)]
        if finite_conf.size > 0:
            conf_threshold = float(np.percentile(finite_conf, conf_percentile))
            valid &= conf_map[grid_v, grid_u] >= conf_threshold

    if sky_map is not None:
        valid &= sky_map[grid_v, grid_u] < sky_threshold

    if not np.any(valid):
        rotation_cw = pose_cw[:3, :3].astype(np.float32, copy=False)
        translation_cw = pose_cw[:3, 3].astype(np.float32, copy=False)
        rotation_wc = rotation_cw.T
        origin_world = (-rotation_wc @ translation_cw).astype(np.float32, copy=False)
        return (
            origin_world,
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )

    u = grid_u[valid].astype(np.float32)
    v = grid_v[valid].astype(np.float32)
    z = sampled_depth[valid].astype(np.float32)

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_cam = np.stack([x, y, z], axis=1)

    rotation_cw = pose_cw[:3, :3].astype(np.float32, copy=False)
    translation_cw = pose_cw[:3, 3].astype(np.float32, copy=False)
    rotation_wc = rotation_cw.T
    origin_world = (-rotation_wc @ translation_cw).astype(np.float32, copy=False)
    endpoints_world = (rotation_wc @ points_cam.T).T + origin_world
    colors_rgb = image_rgb[grid_v[valid], grid_u[valid]].astype(np.uint8, copy=False)
    return origin_world, endpoints_world.astype(np.float32, copy=False), colors_rgb


def main() -> int:
    args = build_arg_parser().parse_args()

    dataset_root = args.dataset_root.resolve()
    vocabulary_path = args.vocabulary_path.resolve()
    settings_path = args.settings_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_camera_stream(dataset_root / "mav0" / "cam0")
    if not frames:
        raise RuntimeError("No frames found in mav0/cam0.")
    imu_samples = load_imu_stream(dataset_root / "mav0" / "imu0")
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    resolved_depth_model = args.depth_model
    if resolved_depth_model is None:
        resolved_depth_model = (
            "lpiccinelli/unidepth-v2-vits14"
            if args.depth_backend == "unidepth_v2"
            else "depth-anything/DA3-SMALL"
        )

    depth_service = DepthEstimationService(
        depth_backend=args.depth_backend,
        fusion_mode=args.fusion_mode,
        model_name=resolved_depth_model,
        unidepth_src=args.unidepth_src,
        unidepth_resolution_level=args.unidepth_resolution_level,
        device=args.device,
        output_dir=output_dir / "depth",
        settings_path=settings_path,
        undistort_balance=args.undistort_balance,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        queue_size=args.queue_size,
        pose_conditioned=args.da3_pose_conditioned,
        context_keyframes=args.da3_context_keyframes,
        reprojection_stride=args.reprojection_stride,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        conf_percentile=args.conf_percentile,
        sky_threshold=args.sky_threshold,
        min_scale_correspondences=args.min_scale_correspondences,
        voxel_size=max(1e-4, args.voxel_size),
        tsdf_sdf_trunc_m=args.tsdf_sdf_trunc_m,
        penetration_tolerance_voxels=args.penetration_tolerance_voxels,
        camera_clearance_radius_m=args.camera_clearance_radius_m,
        camera_clearance_below_m=args.camera_clearance_below_m,
        max_map_voxels=args.max_map_voxels,
        live_export_interval_sec=args.live_export_interval_sec,
        live_point_budget=args.live_point_budget,
        final_point_budget=args.final_point_budget,
    )
    depth_service.start()

    slam = pyorbslam3.System(
        str(vocabulary_path),
        str(settings_path),
        pyorbslam3.Sensor.IMU_MONOCULAR,
        use_viewer=not args.no_orb_viewer,
    )

    use_clahe = not args.no_clahe
    clahe = cv2.createCLAHE(3.0, (8, 8)) if use_clahe else None
    imu_timestamps = [sample.timestamp_ns for sample in imu_samples]
    imu_index = (
        max(bisect.bisect_right(imu_timestamps, frames[0].timestamp_ns) - 1, 0)
        if imu_timestamps
        else 0
    )

    input_window = None if args.no_input_window else PreviewWindow("ORB-SLAM3 Input")
    depth_window_title = "UniDepth V2" if args.depth_backend == "unidepth_v2" else "Depth Anything 3"
    depth_window = None if args.no_depth_window else PreviewWindow(depth_window_title)
    viewer_title = "Occupancy Map (TSDF Voxels)" if args.fusion_mode == "tsdf_voxel" else "Occupancy Map (Raycast)"
    map_window = None if args.no_map_window else MapViewerProcess(viewer_title, output_dir)
    if map_window is not None:
        map_window.start()
    if depth_window is not None:
        depth_window.update(
            make_status_preview(
                [
                    f"Depth window ready: backend={args.depth_backend}",
                    f"model={resolved_depth_model}",
                    f"fusion_mode={args.fusion_mode}",
                    f"device request={args.device}",
                    f"clearance={args.camera_clearance_radius_m:.2f}m below={args.camera_clearance_below_m:.2f}m",
                    f"pose_conditioned={int(args.da3_pose_conditioned and args.depth_backend == 'da3')} window={args.da3_context_keyframes}",
                    "Waiting for first usable ORB-SLAM3 frame...",
                ]
            )
        )

    tracking_log_path = output_dir / "tracking_log.csv"
    trajectory_path = output_dir / "trajectory_tum.txt"
    camera_path_path = output_dir / "camera_path.xyz"
    sparse_points_path = output_dir / "sparse_points.xyz"
    live_status_path = output_dir / "live_status.txt"
    camera_path_path.write_text("", encoding="utf-8")
    write_status_file(
        live_status_path,
        {
            "frame_index": 0,
            "tracking_state": "WAITING",
            "depth_backend": args.depth_backend,
            "fusion_mode": args.fusion_mode,
            "occupancy_voxels": 0,
            "exported_voxels": 0,
            "integrated_frames": 0,
            "backend_queue": 0,
        },
    )

    processed_frames = 0
    valid_poses = 0
    frames_submitted = 0
    frames_dropped = 0
    sparse_points = np.empty((0, 3), dtype=np.float32)
    trajectory_lines: list[str] = []
    camera_positions: list[np.ndarray] = []

    wall_start = time.perf_counter()
    sequence_start_ts = frames[0].timestamp_ns
    last_path_flush_time = time.perf_counter()
    last_status_write_time = 0.0

    try:
        with tracking_log_path.open("w", newline="", encoding="utf-8") as handle, camera_path_path.open(
            "a", encoding="utf-8"
        ) as camera_path_handle:
            writer = csv.writer(handle)
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
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)

                image_gray = read_grayscale_image(frame.image_path, use_clahe, clahe)
                imu_batch: list[pyorbslam3.ImuMeasurement] = []
                if frame_index > 0 and imu_samples:
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

                result = slam.track_monocular(
                    image_gray,
                    frame.timestamp_ns * 1e-9,
                    imu_batch,
                )
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
                tracking_ok = str(result["tracking_state_name"]) in {"OK", "OK_KLT"}
                usable_pose = bool(result["pose_valid"]) and tracking_ok
                depth_service.poll_updates()
                if depth_service.error_message is not None:
                    raise RuntimeError(depth_service.error_message)

                if usable_pose:
                    pose_cw = np.asarray(result["pose_matrix"], dtype=np.float64)
                    pose_wc = np.linalg.inv(pose_cw)
                    camera_position = pose_wc[:3, 3]
                    qx, qy, qz, qw = quaternion_xyzw_from_rotation(pose_wc[:3, :3])
                    trajectory_lines.append(
                        f"{frame.timestamp_ns * 1e-9:.9f} "
                        f"{camera_position[0]:.9f} {camera_position[1]:.9f} {camera_position[2]:.9f} "
                        f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
                    )
                    camera_positions.append(camera_position.astype(np.float32, copy=False))
                    camera_path_handle.write(
                        f"{camera_position[0]:.6f} {camera_position[1]:.6f} {camera_position[2]:.6f}\n"
                    )
                    if (time.perf_counter() - last_path_flush_time) >= 0.20:
                        camera_path_handle.flush()
                        last_path_flush_time = time.perf_counter()
                    valid_poses += 1

                if usable_pose:
                    tracked = slam.get_tracked_observations()
                    sparse_keypoints = np.asarray(tracked["keypoints_uv"], dtype=np.float32)
                    sparse_world_points = np.asarray(tracked["world_points_xyz"], dtype=np.float32)
                    submitted = depth_service.submit(
                        FramePacket(
                            frame_index=frame_index,
                            timestamp_ns=frame.timestamp_ns,
                            image_path=str(frame.image_path.resolve()),
                            pose_cw=np.asarray(result["pose_matrix"], dtype=np.float32),
                            sparse_keypoints_uv=sparse_keypoints,
                            sparse_world_points_xyz=sparse_world_points,
                        )
                    )
                    if submitted:
                        frames_submitted += 1
                    else:
                        frames_dropped += 1

                backend_queue = max(depth_service.get_queue_size(), 0)

                if input_window is not None:
                    input_window.update(
                        make_input_preview(
                            image_gray,
                            frame_index,
                            frame.timestamp_ns,
                            str(result["tracking_state_name"]),
                            usable_pose,
                            bool(result["is_keyframe"]),
                            backend_queue,
                            frames_submitted,
                            frames_dropped,
                        )
                    )

                if depth_window is not None:
                    preview = depth_service.get_latest_preview()
                    if preview is not None:
                        depth_window.update(preview)

                backend_stats = depth_service.get_stats()
                if (time.perf_counter() - last_status_write_time) >= 0.20 or (frame_index + 1) == len(frames):
                    write_status_file(
                        live_status_path,
                        {
                            "frame_index": frame_index,
                            "tracking_state": str(result["tracking_state_name"]),
                            "depth_backend": args.depth_backend,
                            "fusion_mode": args.fusion_mode,
                            "processed_frames": processed_frames,
                            "valid_poses": valid_poses,
                            "frames_submitted": frames_submitted,
                            "frames_dropped": frames_dropped,
                            "integrated_frames": int(backend_stats.get("integrated_frames", 0) or 0),
                            "occupancy_voxels": int(backend_stats.get("occupancy_voxels", 0) or 0),
                            "exported_voxels": int(backend_stats.get("exported_voxels", 0) or 0),
                            "backend_queue": backend_queue,
                            "backend_status": backend_stats.get("last_status", "idle"),
                        },
                    )
                    last_status_write_time = time.perf_counter()

                if args.progress_every > 0 and (frame_index + 1) % args.progress_every == 0:
                    print(
                        f"[{frame_index + 1}/{len(frames)}] "
                        f"state={result['tracking_state_name']} "
                        f"pose_valid={usable_pose} "
                        f"is_keyframe={bool(result['is_keyframe'])} "
                        f"occ={int(backend_stats.get('occupancy_voxels', 0) or 0)} "
                        f"blocked={int(backend_stats.get('blocked_rays', 0) or 0)}"
                    )

            sparse_points = np.asarray(slam.get_current_map_points(), dtype=np.float32)
    finally:
        slam.shutdown()
        try:
            depth_service.finish()
        finally:
            if input_window is not None:
                input_window.close()
            if depth_window is not None:
                depth_window.close()
            if map_window is not None:
                map_window.close()

    trajectory_path.write_text("\n".join(trajectory_lines) + ("\n" if trajectory_lines else ""), encoding="utf-8")
    if camera_positions:
        save_xyz(np.vstack(camera_positions), camera_path_path)
    else:
        camera_path_path.write_text("", encoding="utf-8")
    save_xyz(sparse_points, sparse_points_path)
    occupancy_ply_path, occupancy_npz_path, tsdf_mesh_path = depth_service.export_outputs()

    summary = {
        "depth_backend": args.depth_backend,
        "depth_model": resolved_depth_model,
        "fusion_mode": args.fusion_mode,
        "processed_frames": processed_frames,
        "valid_poses": valid_poses,
        "frames_submitted": frames_submitted,
        "frames_dropped": frames_dropped,
        "sparse_map_points": int(sparse_points.shape[0]),
        "integrated_frames": int(depth_service.get_stats().get("integrated_frames", 0) or 0),
        "occupancy_voxels": int(depth_service.get_stats().get("occupancy_voxels", 0) or 0),
        "exported_voxels": int(depth_service.get_stats().get("exported_voxels", 0) or 0),
        "trajectory_path": str(trajectory_path),
        "occupancy_map_ply": str(occupancy_ply_path),
        "occupancy_map_npz": str(occupancy_npz_path),
        "tsdf_mesh_ply": None if tsdf_mesh_path is None else str(tsdf_mesh_path),
        "depth_worker": depth_service.get_stats(),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Processed frames: {processed_frames}")
    print(f"Depth backend: {args.depth_backend}")
    print(f"Depth model: {resolved_depth_model}")
    print(f"Fusion mode: {args.fusion_mode}")
    print(f"Valid poses: {valid_poses}")
    print(f"Frames submitted to depth backend: {frames_submitted}")
    print(f"Frames dropped at enqueue: {frames_dropped}")
    print(f"Sparse map points: {int(sparse_points.shape[0])}")
    print(f"Integrated occupancy frames: {int(depth_service.get_stats().get('integrated_frames', 0) or 0)}")
    print(f"Occupied voxels: {int(depth_service.get_stats().get('occupancy_voxels', 0) or 0)}")
    print(f"Exported voxels: {int(depth_service.get_stats().get('exported_voxels', 0) or 0)}")
    print(f"Occupancy map: {occupancy_ply_path}")
    if tsdf_mesh_path is not None:
        print(f"TSDF mesh: {tsdf_mesh_path}")
    print(f"Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
