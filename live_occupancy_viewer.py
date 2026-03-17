#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


POLL_INTERVAL_MS = 500
MAX_RENDER_POINTS = 12_000
HEATMAP_MIN_BINS = 12
HEATMAP_MAX_BINS = 56


def load_xyz(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 3), dtype=float)
    try:
        points = np.loadtxt(path, dtype=float)
    except (ValueError, OSError):
        return np.empty((0, 3), dtype=float)
    if points.size == 0:
        return np.empty((0, 3), dtype=float)
    if points.ndim == 1:
        return points.reshape(1, -1)
    return points


def load_status(path: Path) -> dict[str, str]:
    status: dict[str, str] = {}
    if not path.exists():
        return status
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        status[key] = value
    return status


def xyz_only(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=float)
    return points[:, :3]


def point_colors(points: np.ndarray) -> np.ndarray | None:
    if points.size == 0 or points.shape[1] < 6:
        return None
    return np.clip(points[:, 3:6] / 255.0, 0.0, 1.0)


def decimate_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.size == 0 or points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int32)
    return points[indices]


def set_axes_equal(ax: plt.Axes, points_xyz: np.ndarray, path_points: np.ndarray) -> None:
    if points_xyz.size == 0 and path_points.size == 0:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        return

    if points_xyz.size == 0:
        all_points = path_points
    elif path_points.size == 0:
        all_points = points_xyz
    else:
        all_points = np.vstack((points_xyz, path_points))

    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 0.5)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def combined_points(points_xyz: np.ndarray, path_points: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0 and path_points.size == 0:
        return np.empty((0, 3), dtype=float)
    if points_xyz.size == 0:
        return path_points
    if path_points.size == 0:
        return points_xyz
    return np.vstack((points_xyz, path_points))


def infer_vertical_axis(points_xyz: np.ndarray, path_points: np.ndarray) -> int:
    all_points = combined_points(points_xyz, path_points)
    if all_points.size == 0:
        return 1

    lower = np.percentile(all_points, 2.0, axis=0)
    upper = np.percentile(all_points, 98.0, axis=0)
    spans = np.maximum(upper - lower, 1e-6)
    return int(np.argmin(spans))


def infer_view_dims(points_xyz: np.ndarray, path_points: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    vertical_axis = infer_vertical_axis(points_xyz, path_points)
    horizontal_axes = [axis for axis in range(3) if axis != vertical_axis]

    all_points = combined_points(points_xyz, path_points)
    if all_points.size != 0:
        lower = np.percentile(all_points, 2.0, axis=0)
        upper = np.percentile(all_points, 98.0, axis=0)
        spans = upper - lower
        horizontal_axes.sort(key=lambda axis: spans[axis], reverse=True)

    top_dims = (horizontal_axes[0], horizontal_axes[1])
    side_dims = (horizontal_axes[0], vertical_axis)
    return top_dims, side_dims


def planar_bounds(
    points_xyz: np.ndarray,
    path_points: np.ndarray,
    dims: tuple[int, int],
) -> tuple[float, float, float, float]:
    if points_xyz.size == 0 and path_points.size == 0:
        return (-1.0, 1.0, -1.0, 1.0)

    if points_xyz.size == 0:
        all_points = path_points[:, list(dims)]
    elif path_points.size == 0:
        all_points = points_xyz[:, list(dims)]
    else:
        all_points = np.vstack((points_xyz[:, list(dims)], path_points[:, list(dims)]))

    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 0.5)
    return (
        center[0] - radius,
        center[0] + radius,
        center[1] - radius,
        center[1] + radius,
    )


def axis_name(axis: int) -> str:
    return ("x", "y", "z")[axis]


def render_heatmap(
    ax: plt.Axes,
    points_xyz: np.ndarray,
    path_points: np.ndarray,
    dims: tuple[int, int],
    axis_labels: tuple[str, str],
    title: str,
) -> None:
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_aspect("equal", adjustable="box")

    xmin, xmax, ymin, ymax = planar_bounds(points_xyz, path_points, dims)
    if points_xyz.size:
        bins = max(HEATMAP_MIN_BINS, min(HEATMAP_MAX_BINS, int(np.sqrt(len(points_xyz)))))
        heatmap, _, _ = np.histogram2d(
            points_xyz[:, dims[0]],
            points_xyz[:, dims[1]],
            bins=bins,
            range=[[xmin, xmax], [ymin, ymax]],
        )
        if np.any(heatmap > 0):
            ax.imshow(
                heatmap.T,
                origin="lower",
                extent=[xmin, xmax, ymin, ymax],
                cmap="inferno",
                interpolation="nearest",
                aspect="equal",
            )
        else:
            ax.scatter(
                points_xyz[:, dims[0]],
                points_xyz[:, dims[1]],
                s=3.0,
                c="#1f77b4",
                alpha=0.6,
            )

    if path_points.size:
        ax.plot(
            path_points[:, dims[0]],
            path_points[:, dims[1]],
            color="#00bcd4",
            linewidth=1.5,
        )
        ax.scatter(
            path_points[-1:, dims[0]],
            path_points[-1:, dims[1]],
            s=35.0,
            c="#111111",
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)


def main() -> int:
    output_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    title = sys.argv[2] if len(sys.argv) > 2 else "Occupancy Map"

    points_path = output_dir / "depth" / "occupancy_live.xyz"
    if not points_path.exists():
        legacy_points_path = output_dir / "da3" / "occupancy_live.xyz"
        if legacy_points_path.exists():
            points_path = legacy_points_path
    camera_path = output_dir / "camera_path.xyz"
    status_path = output_dir / "live_status.txt"

    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(131, projection="3d")
    ax_top = fig.add_subplot(132)
    ax_side = fig.add_subplot(133)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax_top.set_xlabel("x")
    ax_top.set_ylabel("y")
    ax_top.set_aspect("equal", adjustable="box")
    ax_side.set_xlabel("x")
    ax_side.set_ylabel("z")
    ax_side.set_aspect("equal", adjustable="box")
    ax.view_init(elev=72, azim=-90)

    last_mtimes = {"points": 0.0, "path": 0.0, "status": 0.0}
    cached_points = np.empty((0, 3), dtype=float)
    cached_path = np.empty((0, 3), dtype=float)
    cached_status: dict[str, str] = {}

    def update(_frame_index: int) -> None:
        nonlocal cached_points, cached_path, cached_status

        if points_path.exists():
            mtime = points_path.stat().st_mtime
            if mtime != last_mtimes["points"]:
                loaded_points = load_xyz(points_path)
                if loaded_points.size != 0 or points_path.stat().st_size == 0:
                    cached_points = decimate_points(loaded_points, MAX_RENDER_POINTS)
                    last_mtimes["points"] = mtime

        if camera_path.exists():
            mtime = camera_path.stat().st_mtime
            if mtime != last_mtimes["path"]:
                loaded_path = load_xyz(camera_path)
                if loaded_path.size != 0 or camera_path.stat().st_size == 0:
                    cached_path = loaded_path
                    last_mtimes["path"] = mtime

        if status_path.exists():
            mtime = status_path.stat().st_mtime
            if mtime != last_mtimes["status"]:
                cached_status = load_status(status_path)
                last_mtimes["status"] = mtime

        ax.cla()
        ax_top.cla()
        ax_side.cla()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax_top.set_xlabel("x")
        ax_top.set_ylabel("y")
        ax_top.set_aspect("equal", adjustable="box")
        ax_side.set_xlabel("x")
        ax_side.set_ylabel("z")
        ax_side.set_aspect("equal", adjustable="box")
        ax.view_init(elev=72, azim=-90)

        colors = point_colors(cached_points)
        points_xyz = xyz_only(cached_points)

        if points_xyz.size:
            ax.scatter(
                points_xyz[:, 0],
                points_xyz[:, 1],
                points_xyz[:, 2],
                s=2.0,
                c=colors if colors is not None else "#1f77b4",
                alpha=0.7,
            )

        if cached_path.size:
            ax.plot(
                cached_path[:, 0],
                cached_path[:, 1],
                cached_path[:, 2],
                color="#d62728",
                linewidth=1.5,
            )
            ax.scatter(
                cached_path[-1:, 0],
                cached_path[-1:, 1],
                cached_path[-1:, 2],
                s=35.0,
                c="#111111",
            )

        set_axes_equal(ax, points_xyz, cached_path)
        top_dims, side_dims = infer_view_dims(points_xyz, cached_path)
        vertical_axis = next(axis for axis in range(3) if axis not in top_dims)
        render_heatmap(
            ax_top,
            points_xyz,
            cached_path,
            dims=top_dims,
            axis_labels=(axis_name(top_dims[0]), axis_name(top_dims[1])),
            title=f"Top-Down Density ({axis_name(top_dims[0])}-{axis_name(top_dims[1])})",
        )
        render_heatmap(
            ax_side,
            points_xyz,
            cached_path,
            dims=side_dims,
            axis_labels=(axis_name(side_dims[0]), axis_name(side_dims[1])),
            title=f"Side Density ({axis_name(side_dims[0])}-{axis_name(side_dims[1])})",
        )

        frame_index = cached_status.get("frame_index", "?")
        tracking_state = cached_status.get("tracking_state", "WAITING")
        occupancy_voxels = cached_status.get("occupancy_voxels", "0")
        exported_voxels = cached_status.get("exported_voxels", "0")
        integrated_frames = cached_status.get("integrated_frames", "0")
        backend_status = cached_status.get("backend_status", "idle")
        ax.set_title(
            f"{title}  frame={frame_index}  state={tracking_state}  "
            f"frames={integrated_frames}  occ={occupancy_voxels}  "
            f"live={exported_voxels}  backend={backend_status}  up={axis_name(vertical_axis)}"
        )

    animation = FuncAnimation(fig, update, interval=POLL_INTERVAL_MS, cache_frame_data=False)
    fig._live_animation = animation
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
