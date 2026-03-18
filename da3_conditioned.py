#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DA3_SRC = SCRIPT_DIR / "dependency" / "Depth-Anything-3" / "src"
if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export.glb import export_to_glb


@dataclass(frozen=True)
class Da3InputBundle:
    da3_dir: Path
    image_paths: list[Path]
    keyframe_indices: np.ndarray
    frame_indices: np.ndarray
    timestamps_ns: np.ndarray
    extrinsics: np.ndarray
    intrinsics: np.ndarray


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Depth Anything 3 on ORB-SLAM3-exported frames with Global Depth Alignment "
            "to create consistent 3D maps for Octomap/Occupancy mapping."
        ),
    )
    parser.add_argument(
        "--orb-output-dir",
        type=Path,
        default=None,
        help="Input directory (bundle). Auto-detected on Kaggle if not set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="depth-anything/DA3-SMALL",
        help="DA3 checkpoint (e.g. SMALL, BASE, or METRIC-LARGE).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Number of frames per window. Lower this (e.g. 4 or 6) if you hit VRAM OOM.",
    )
    parser.add_argument(
        "--window-overlap",
        type=int,
        default=3,
        help="Number of overlapping frames. Must be less than window-size.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only first N frames.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Temporal stride when sampling from the bundle.",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=512,
        help="DA3 processing resolution.",
    )
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=2,
        help="Backprojection stride for the dense point cloud.",
    )
    parser.add_argument(
        "--min-depth-m",
        type=float,
        default=0.1,
        help="Minimum accepted depth.",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=50.0,
        help="Maximum accepted depth.",
    )
    parser.add_argument(
        "--conf-thresh-percentile",
        type=float,
        default=40.0,
        help="Per-frame confidence percentile to keep points.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2_000_000,
        help="Cap the final dense point cloud size.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="Voxel size in meters for downsampling. 0.05 = 5cm.",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="mini_npz-glb",
        help="Subset of {mini_npz,npz,glb} joined by '-'.",
    )
    return parser


def detect_kaggle_bundle() -> Path | None:
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists(): return None
    for bundle_dir in kaggle_input.rglob("*"):
        if bundle_dir.is_dir() and ((bundle_dir / "manifest.json").exists() or (bundle_dir / "keyframes.csv").exists()):
             return bundle_dir
    return None


def resolve_da3_dir(orb_output_dir: Path | None) -> Path:
    if orb_output_dir is None:
        detected = detect_kaggle_bundle()
        if detected: return detected
        raise ValueError("Missing --orb-output-dir and could not auto-detect Kaggle input.")
    da3_dir = orb_output_dir / "da3"
    return da3_dir if da3_dir.exists() else orb_output_dir.resolve()


def load_bundle(da3_dir: Path, max_frames: int | None = None, stride: int = 1) -> Da3InputBundle:
    keyframes_csv = da3_dir / "keyframes.csv"
    extrinsics_path = da3_dir / "extrinsics.npy"
    intrinsics_path = da3_dir / "intrinsics.npy"

    rows = []
    with keyframes_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader: rows.append(row)

    stride = max(1, stride)
    rows = rows[::stride]
    if max_frames is not None: rows = rows[:max_frames]

    sel = [int(r.get("keyframe_index", i*stride)) for i, r in enumerate(rows)]
    image_paths = [(da3_dir / r["image_path"]).resolve() for r in rows]
    keyframe_indices = np.asarray([int(r["keyframe_index"]) for r in rows], dtype=np.int32)
    extrinsics = np.load(extrinsics_path).astype(np.float32, copy=False)[sel]
    intrinsics = np.load(intrinsics_path).astype(np.float32, copy=False)[sel]
    
    return Da3InputBundle(
        da3_dir=da3_dir, image_paths=image_paths, keyframe_indices=keyframe_indices,
        frame_indices=keyframe_indices,
        timestamps_ns=keyframe_indices,
        extrinsics=extrinsics, intrinsics=intrinsics,
    )


def configure_model(model: DepthAnything3) -> None:
    original_align = model._align_to_input_extrinsics_intrinsics

    def to_numpy(value):
        if value is None: return None
        if isinstance(value, torch.Tensor): return value.detach().cpu().numpy()
        return np.asarray(value)

    def passthrough_input_pose_alignment(extrinsics, intrinsics, prediction, align_to_input_ext_scale=True, ransac_view_thresh=10):
        if extrinsics is not None:
            prediction.extrinsics = to_numpy(extrinsics)
            if intrinsics is not None:
                prediction.intrinsics = to_numpy(intrinsics)
            return prediction
        return original_align(extrinsics, intrinsics, prediction, align_to_input_ext_scale, ransac_view_thresh)

    model._align_to_input_extrinsics_intrinsics = passthrough_input_pose_alignment


def align_depth_least_squares(target_depth, source_depth, conf=None):
    mask = np.isfinite(target_depth) & np.isfinite(source_depth)
    if conf is not None:
        mask &= (conf > np.percentile(conf, 50))
    y, x = target_depth[mask], source_depth[mask]
    if len(y) < 100: return 1.0, 0.0
    A = np.vstack([x, np.ones(len(x))]).T
    s, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(s), float(b)


def backproject_frame(depth_map, image_rgb, intrinsic, extrinsic_w2c, conf_map, conf_threshold, pixel_stride, min_depth_m, max_depth_m):
    height, width = depth_map.shape
    step = max(int(pixel_stride), 1)
    sampled_depth = depth_map[::step, ::step]
    valid = np.isfinite(sampled_depth) & (sampled_depth > float(min_depth_m)) & (sampled_depth < float(max_depth_m))
    if conf_map is not None and conf_threshold is not None:
        valid &= conf_map[::step, ::step] >= conf_threshold
    if not np.any(valid): return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    us, vs = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))
    u, v, z = us[valid].astype(np.float32), vs[valid].astype(np.float32), sampled_depth[valid].astype(np.float32)
    fx, fy, cx, cy = float(intrinsic[0, 0]), float(intrinsic[1, 1]), float(intrinsic[0, 2]), float(intrinsic[1, 2])
    x, y = ((u - cx) / fx) * z, ((v - cy) / fy) * z
    points_cam = np.stack([x, y, z], axis=1)
    pose_wc = np.linalg.inv(extrinsic_w2c.astype(np.float64))
    points_world = (points_cam @ pose_wc[:3, :3].T) + pose_wc[:3, 3]
    colors_rgb = image_rgb[::step, ::step][valid].astype(np.uint8)
    return points_world.astype(np.float32), colors_rgb


def voxel_downsample(points, colors, voxel_size=0.05):
    if len(points) == 0: return points, colors
    coords = (points / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    return points[unique_indices], colors[unique_indices]


def save_prediction_npz(prediction: Prediction, output_dir: Path, include_images: bool) -> None:
    export_dir = output_dir / "exports" / ("npz" if include_images else "mini_npz")
    export_dir.mkdir(parents=True, exist_ok=True)
    save_dict: dict[str, np.ndarray] = {"depth": np.round(prediction.depth, 8)}
    if include_images and prediction.processed_images is not None:
        save_dict["image"] = prediction.processed_images
    if prediction.conf is not None:
        save_dict["conf"] = np.round(prediction.conf, 2)
    if prediction.extrinsics is not None:
        save_dict["extrinsics"] = prediction.extrinsics
    if prediction.intrinsics is not None:
        save_dict["intrinsics"] = prediction.intrinsics
    np.savez_compressed(export_dir / "results.npz", **save_dict)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    da3_dir = resolve_da3_dir(args.orb_output_dir)
    bundle = load_bundle(da3_dir, max_frames=args.max_frames, stride=args.stride)
    output_dir = args.output_dir.resolve() if args.output_dir else (Path("/kaggle/working/da3_results") if Path("/kaggle/working").exists() else da3_dir.parent / "da3_dense")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    model = DepthAnything3.from_pretrained(args.model_name).to(torch.device(args.device))
    configure_model(model)
    
    num_views = len(bundle.image_paths)
    window_size = args.window_size
    window_overlap = args.window_overlap
    stride = window_size - window_overlap
    
    global_depths = [None] * num_views
    global_confs = [None] * num_views
    global_images = [None] * num_views

    print(f"Processing {num_views} frames with Global Alignment...")

    for start in range(0, num_views, stride):
        end = min(start + window_size, num_views)
        if start >= num_views: break
        indices = slice(start, end)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            prediction = model.inference(
                image=[str(p) for p in bundle.image_paths[indices]],
                extrinsics=bundle.extrinsics[indices],
                intrinsics=bundle.intrinsics[indices],
                align_to_input_ext_scale=True,
                process_res=args.process_res,
            )

        depths = np.asarray(prediction.depth, dtype=np.float32)
        confs = np.asarray(prediction.conf, dtype=np.float32) if prediction.conf is not None else None
        images = np.asarray(prediction.processed_images, dtype=np.uint8)

        s, b = 1.0, 0.0
        overlap_found = False
        for i in range(start, min(start + window_overlap, num_views)):
            if global_depths[i] is not None:
                local_idx = i - start
                s_i, b_i = align_depth_least_squares(global_depths[i], depths[local_idx], confs[local_idx] if confs is not None else None)
                s, b = s_i, b_i
                overlap_found = True
                break
        
        if overlap_found:
            print(f" Window {start}:{end} -> Aligned (s={s:.3f}, b={b:.3f})")
            depths = s * depths + b
        else:
            print(f" Window {start}:{end} -> Anchor")

        for i in range(start, end):
            local_idx = i - start
            if global_depths[i] is None or (i >= start + window_overlap // 2):
                global_depths[i] = depths[local_idx]
                global_images[i] = images[local_idx]
                if confs is not None: global_confs[i] = confs[local_idx]

    # FINAL BACKPROJECTION & MERGE
    point_chunks, color_chunks = [], []
    print(f"Backprojecting {num_views} frames...")
    valid_indices = [i for i in range(num_views) if global_depths[i] is not None]
    
    for i in valid_indices:
        conf_threshold = float(np.percentile(global_confs[i], args.conf_thresh_percentile)) if global_confs[i] is not None else None
        pts, clrs = backproject_frame(global_depths[i], global_images[i], bundle.intrinsics[i], bundle.extrinsics[i], global_confs[i], conf_threshold, args.pixel_stride, args.min_depth_m, args.max_depth_m)
        point_chunks.append(pts)
        color_chunks.append(clrs)

    dense_points = np.concatenate(point_chunks, axis=0) if point_chunks else np.empty((0, 3), dtype=np.float32)
    dense_colors = np.concatenate(color_chunks, axis=0) if color_chunks else np.empty((0, 3), dtype=np.uint8)
    
    print(f"Applying voxel downsampling (size={args.voxel_size}m)...")
    dense_points, dense_colors = voxel_downsample(dense_points, dense_colors, args.voxel_size)

    if dense_points.shape[0] > args.max_points:
        idx = np.random.choice(dense_points.shape[0], args.max_points, replace=False)
        dense_points, dense_colors = dense_points[idx], dense_colors[idx]

    # Export Manual PLY
    print(f"Writing {dense_points.shape[0]} points to consistent_map.ply...")
    with (output_dir / "consistent_map.ply").open("w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {dense_points.shape[0]}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(dense_points, dense_colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    # Export Standard DA3 Formats (including GLB)
    export_formats = {item for item in args.export_format.split("-") if item}
    if export_formats:
        print("Exporting additional formats (including GLB)...")
        combined_prediction = Prediction(
            depth=np.stack([global_depths[i] for i in valid_indices]),
            is_metric=1, # We forced consistency
            conf=np.stack([global_confs[i] for i in valid_indices]) if global_confs[0] is not None else None,
            extrinsics=bundle.extrinsics[valid_indices],
            intrinsics=bundle.intrinsics[valid_indices],
            processed_images=np.stack([global_images[i] for i in valid_indices]),
            aux={}
        )
        if "mini_npz" in export_formats: save_prediction_npz(combined_prediction, output_dir, include_images=False)
        if "npz" in export_formats: save_prediction_npz(combined_prediction, output_dir, include_images=True)
        if "glb" in export_formats:
            export_to_glb(
                combined_prediction,
                str(output_dir / "exports" / "glb"),
                conf_thresh_percentile=args.conf_thresh_percentile,
                num_max_points=args.max_points,
                show_cameras=True,
            )

    print(f"Done. Results saved to {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
