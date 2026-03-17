from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Iterable

import numpy as np
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEPTH_ANYTHING_REPO = WORKSPACE_ROOT / "dependency" / "Depth-Anything-3"
DEPTH_ANYTHING_SRC = DEPTH_ANYTHING_REPO / "src"

if str(DEPTH_ANYTHING_SRC) not in sys.path:
    sys.path.insert(0, str(DEPTH_ANYTHING_SRC))

from depth_anything_3.api import DepthAnything3  # noqa: E402


CANDIDATE_MODELS = [
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "depth-anything/DA3-BASE",
    "depth-anything/DA3-SMALL",
    "depth-anything/DA3-LARGE-1.1",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
TUM_INTRINSICS_PATH = WORKSPACE_ROOT / "dependency" / "ORB_SLAM3" / "Examples" / "Monocular" / "TUM-VI.yaml"
DEFAULT_POSE_DIR = WORKSPACE_ROOT / "outputs" / "corridor_4"
DEFAULT_DATASET_DIR = WORKSPACE_ROOT / "dataset"
DEFAULT_JSON_OUTPUT = Path(__file__).resolve().with_name("da3_cpu_benchmark.json")
DEFAULT_CSV_OUTPUT = Path(__file__).resolve().with_name("da3_cpu_benchmark.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Depth Anything 3 models on CPU with pose conditioning."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset root. Images are searched recursively. Default: ../dataset relative to depth_utils.",
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=DEFAULT_POSE_DIR,
        help="Directory containing corridor_4 trajectory_tum.txt and tracking_log.csv.",
    )
    parser.add_argument(
        "--intrinsics-file",
        type=Path,
        default=TUM_INTRINSICS_PATH,
        help="YAML file used to read Camera1 intrinsics defaults.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images to sample from the matched pose/image set.",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Depth Anything processing resolution.",
    )
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        help="Depth Anything resize method.",
    )
    parser.add_argument(
        "--time-tolerance-ns",
        type=int,
        default=5_000_000,
        help="Nearest timestamp tolerance used when filenames do not exactly match poses.",
    )
    parser.add_argument(
        "--align-to-input-ext-scale",
        action="store_true",
        help="Rescale predicted depth to the input pose scale after inference.",
    )
    parser.add_argument(
        "--use-ray-pose",
        action="store_true",
        help="Use the slower ray-based pose head during inference.",
    )
    parser.add_argument(
        "--ref-view-strategy",
        type=str,
        default="saddle_balanced",
        help="Reference view strategy forwarded to Depth Anything 3.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=CANDIDATE_MODELS,
        help="Model repo IDs or local model directories.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Where to save the JSON benchmark summary.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help="Where to save the CSV benchmark summary.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional CPU thread count for PyTorch.",
    )
    return parser


def patch_cpu_forward(model: DepthAnything3) -> None:
    original_forward = model.forward

    @torch.inference_mode()
    def cpu_safe_forward(
        self: DepthAnything3,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, torch.Tensor]:
        if image.device.type == "cpu":
            with torch.no_grad():
                return self.model(
                    image,
                    extrinsics,
                    intrinsics,
                    export_feat_layers,
                    infer_gs,
                    use_ray_pose,
                    ref_view_strategy,
                )
        return original_forward(
            image,
            extrinsics,
            intrinsics,
            export_feat_layers,
            infer_gs,
            use_ray_pose,
            ref_view_strategy,
        )

    model.forward = MethodType(cpu_safe_forward, model)


def normalize_scene_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def infer_scene_token_from_pose_dir(pose_dir: Path) -> str:
    normalized = normalize_scene_name(pose_dir.name)
    normalized = normalized.replace("outputs", "")
    return normalized


def guess_dataset_dir(dataset_dir: Path, pose_dir: Path) -> Path:
    if dataset_dir.exists():
        return dataset_dir

    scene_token = infer_scene_token_from_pose_dir(pose_dir)
    candidate_roots = [
        dataset_dir,
        WORKSPACE_ROOT / "dataset",
        WORKSPACE_ROOT.parent / "dataset",
        Path.home() / "dataset",
        Path.home() / "machine_perception_system" / "dataset",
    ]

    guesses: list[Path] = []
    for root in candidate_roots:
        if not root.exists():
            continue

        if root.is_dir() and (root / "mav0" / "cam0" / "data").exists():
            guesses.append(root / "mav0" / "cam0" / "data")

        pattern = f"*{scene_token}*"
        for candidate in root.glob(pattern):
            if not candidate.is_dir():
                continue
            cam0_dir = candidate / "mav0" / "cam0" / "data"
            if cam0_dir.exists():
                guesses.append(cam0_dir)
            else:
                guesses.append(candidate)

    if guesses:
        guesses = sorted(dict.fromkeys(guesses))
        return guesses[0]

    return dataset_dir


def load_intrinsics_matrix(intrinsics_file: Path) -> np.ndarray:
    if not intrinsics_file.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_file}")

    text = intrinsics_file.read_text(encoding="utf-8")
    values = {}
    for key in ("Camera1.fx", "Camera1.fy", "Camera1.cx", "Camera1.cy"):
        match = re.search(rf"^{re.escape(key)}:\s*([-+0-9.eE]+)", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"Could not find {key} in {intrinsics_file}")
        values[key] = float(match.group(1))

    return np.array(
        [
            [values["Camera1.fx"], 0.0, values["Camera1.cx"]],
            [0.0, values["Camera1.fy"], values["Camera1.cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = q / norm

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def parse_tracking_states(tracking_log_path: Path) -> set[int]:
    if not tracking_log_path.exists():
        return set()

    valid_timestamps: set[int] = set()
    with tracking_log_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp_ns = int(row["timestamp_ns"])
            tracking_state = row["tracking_state"].strip()
            pose_valid = row["pose_valid"].strip()
            if tracking_state == "OK" and pose_valid == "1":
                valid_timestamps.add(timestamp_ns)
    return valid_timestamps


def parse_trajectory(trajectory_path: Path, valid_timestamps: set[int]) -> dict[int, np.ndarray]:
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    extrinsics_by_timestamp: dict[int, np.ndarray] = {}
    for line in trajectory_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        values = stripped.split()
        if len(values) != 8:
            continue

        timestamp_ns = int(round(float(values[0]) * 1_000_000_000.0))
        if valid_timestamps and timestamp_ns not in valid_timestamps:
            continue

        tx, ty, tz, qx, qy, qz, qw = map(float, values[1:])
        rotation_c2w = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        translation_c2w = np.array([tx, ty, tz], dtype=np.float64)

        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = rotation_c2w.T
        extrinsic[:3, 3] = -rotation_c2w.T @ translation_c2w

        if valid_timestamps or not np.allclose(extrinsic, np.eye(4), atol=1e-9):
            extrinsics_by_timestamp[timestamp_ns] = extrinsic.astype(np.float32)

    if not extrinsics_by_timestamp:
        raise RuntimeError(f"No usable poses found in {trajectory_path}")
    return extrinsics_by_timestamp


def collect_images(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Point --dataset-dir to the folder that contains your corridor_4 images."
        )

    images = sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No image files found under {dataset_dir}")
    return images


def parse_timestamp_from_path(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return None
    return int(matches[-1])


def select_evenly_spaced(items: list[dict], count: int) -> list[dict]:
    if count <= 0:
        raise ValueError("--num-images must be positive")
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]

    indices = []
    max_index = len(items) - 1
    for i in range(count):
        idx = int(round(i * max_index / (count - 1)))
        if not indices or idx != indices[-1]:
            indices.append(idx)

    while len(indices) < count:
        for idx in range(len(items)):
            if idx not in indices:
                indices.append(idx)
                if len(indices) == count:
                    break

    return [items[idx] for idx in sorted(indices[:count])]


def match_images_to_poses(
    image_paths: Iterable[Path],
    extrinsics_by_timestamp: dict[int, np.ndarray],
    tolerance_ns: int,
) -> list[dict]:
    matched = []
    sorted_pose_timestamps = sorted(extrinsics_by_timestamp)
    remaining_pose_timestamps = set(sorted_pose_timestamps)

    for image_path in image_paths:
        image_timestamp = parse_timestamp_from_path(image_path)
        if image_timestamp is None:
            continue

        pose_timestamp = None
        if image_timestamp in extrinsics_by_timestamp:
            pose_timestamp = image_timestamp
        else:
            nearest = min(
                sorted_pose_timestamps,
                key=lambda ts: abs(ts - image_timestamp),
                default=None,
            )
            if nearest is not None and abs(nearest - image_timestamp) <= tolerance_ns:
                pose_timestamp = nearest

        if pose_timestamp is None or pose_timestamp not in remaining_pose_timestamps:
            continue

        remaining_pose_timestamps.remove(pose_timestamp)
        matched.append(
            {
                "image_path": image_path,
                "image_timestamp_ns": image_timestamp,
                "pose_timestamp_ns": pose_timestamp,
                "extrinsic": extrinsics_by_timestamp[pose_timestamp],
            }
        )

    if matched:
        return sorted(matched, key=lambda item: item["pose_timestamp_ns"])

    image_paths = list(image_paths)
    usable_count = min(len(image_paths), len(sorted_pose_timestamps))
    if usable_count == 0:
        raise RuntimeError(
            "Could not match image timestamps to pose timestamps and there are no images "
            "available for a fallback sequential pairing."
        )

    fallback = []
    for image_path, pose_timestamp in zip(sorted(image_paths)[:usable_count], sorted_pose_timestamps[:usable_count]):
        fallback.append(
            {
                "image_path": image_path,
                "image_timestamp_ns": parse_timestamp_from_path(image_path),
                "pose_timestamp_ns": pose_timestamp,
                "extrinsic": extrinsics_by_timestamp[pose_timestamp],
            }
        )
    return fallback


def summarize_array(array: np.ndarray) -> dict[str, float | list[int]]:
    finite = np.isfinite(array)
    finite_values = array[finite]
    if finite_values.size == 0:
        return {
            "shape": list(array.shape),
            "finite_ratio": 0.0,
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
        }

    return {
        "shape": list(array.shape),
        "finite_ratio": float(finite.mean()),
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "mean": float(finite_values.mean()),
        "median": float(np.median(finite_values)),
        "std": float(finite_values.std()),
    }


def benchmark_model(
    model_name: str,
    image_paths: list[str],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    image_records: list[dict],
) -> dict:
    result: dict[str, object] = {
        "model": model_name,
        "device": "cpu",
        "num_images": len(image_paths),
    }

    load_start = time.perf_counter()
    model = DepthAnything3.from_pretrained(model_name).to("cpu")
    patch_cpu_forward(model)
    load_time = time.perf_counter() - load_start

    infer_start = time.perf_counter()
    prediction = model.inference(
        image=image_paths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        align_to_input_ext_scale=args.align_to_input_ext_scale,
        infer_gs=False,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
    )
    inference_time = time.perf_counter() - infer_start

    result.update(
        {
            "load_time_seconds": load_time,
            "inference_time_seconds": inference_time,
            "seconds_per_image": inference_time / len(image_paths),
            "images_per_second": len(image_paths) / inference_time if inference_time else math.inf,
            "depth": summarize_array(prediction.depth),
            "conf": summarize_array(prediction.conf) if prediction.conf is not None else None,
            "has_extrinsics": prediction.extrinsics is not None,
            "has_intrinsics": prediction.intrinsics is not None,
            "has_processed_images": prediction.processed_images is not None,
            "is_metric": bool(prediction.is_metric),
        }
    )

    per_image = []
    for index, record in enumerate(image_records):
        row = {
            "index": index,
            "image_path": str(record["image_path"]),
            "pose_timestamp_ns": int(record["pose_timestamp_ns"]),
            "image_timestamp_ns": (
                int(record["image_timestamp_ns"]) if record["image_timestamp_ns"] is not None else None
            ),
            "depth": summarize_array(prediction.depth[index]),
        }
        if prediction.conf is not None:
            row["conf"] = summarize_array(prediction.conf[index])
        per_image.append(row)
    result["per_image"] = per_image

    del prediction
    del model
    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "status",
                "num_images",
                "load_time_seconds",
                "inference_time_seconds",
                "seconds_per_image",
                "images_per_second",
                "depth_mean",
                "depth_std",
                "depth_min",
                "depth_max",
                "conf_mean",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            depth = result.get("depth") or {}
            conf = result.get("conf") or {}
            writer.writerow(
                {
                    "model": result.get("model"),
                    "status": result.get("status"),
                    "num_images": result.get("num_images"),
                    "load_time_seconds": result.get("load_time_seconds"),
                    "inference_time_seconds": result.get("inference_time_seconds"),
                    "seconds_per_image": result.get("seconds_per_image"),
                    "images_per_second": result.get("images_per_second"),
                    "depth_mean": depth.get("mean"),
                    "depth_std": depth.get("std"),
                    "depth_min": depth.get("min"),
                    "depth_max": depth.get("max"),
                    "conf_mean": conf.get("mean"),
                    "error": result.get("error"),
                }
            )


def print_run_header(args: argparse.Namespace, selected_records: list[dict]) -> None:
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Pose dir: {args.pose_dir}")
    print(f"Intrinsics file: {args.intrinsics_file}")
    print(f"Selected images: {len(selected_records)}")
    for record in selected_records:
        print(
            "  "
            f"{record['pose_timestamp_ns']} -> {record['image_path']}"
        )


def main() -> int:
    args = build_parser().parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))

    args.dataset_dir = guess_dataset_dir(args.dataset_dir, args.pose_dir)

    intrinsics_matrix = load_intrinsics_matrix(args.intrinsics_file)
    valid_timestamps = parse_tracking_states(args.pose_dir / "tracking_log.csv")
    extrinsics_by_timestamp = parse_trajectory(args.pose_dir / "trajectory_tum.txt", valid_timestamps)
    images = collect_images(args.dataset_dir)
    matched_records = match_images_to_poses(images, extrinsics_by_timestamp, args.time_tolerance_ns)
    selected_records = select_evenly_spaced(matched_records, args.num_images)
    if not selected_records:
        raise RuntimeError("No image/pose pairs were selected for benchmarking.")

    if len(selected_records) < args.num_images:
        print(
            f"Only found {len(selected_records)} matched images with poses. "
            f"Requested {args.num_images}.",
            file=sys.stderr,
        )

    image_paths = [str(record["image_path"]) for record in selected_records]
    extrinsics = np.stack([record["extrinsic"] for record in selected_records], axis=0).astype(np.float32)
    intrinsics = np.repeat(intrinsics_matrix[None, ...], len(selected_records), axis=0)

    print_run_header(args, selected_records)

    results = []
    for model_name in args.models:
        print(f"\nBenchmarking {model_name} on CPU...")
        try:
            result = benchmark_model(
                model_name=model_name,
                image_paths=image_paths,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                args=args,
                image_records=selected_records,
            )
            result["status"] = "ok"
            results.append(result)
            print(
                f"  load={result['load_time_seconds']:.3f}s "
                f"infer={result['inference_time_seconds']:.3f}s "
                f"ips={result['images_per_second']:.3f} "
                f"depth_mean={result['depth']['mean']:.6f}"
            )
        except Exception as exc:
            error_result = {
                "model": model_name,
                "status": "error",
                "num_images": len(image_paths),
                "error": f"{type(exc).__name__}: {exc}",
            }
            results.append(error_result)
            print(f"  failed: {error_result['error']}", file=sys.stderr)

    payload = {
        "device": "cpu",
        "dataset_dir": str(args.dataset_dir),
        "pose_dir": str(args.pose_dir),
        "intrinsics_file": str(args.intrinsics_file),
        "num_images": len(selected_records),
        "process_res": args.process_res,
        "process_res_method": args.process_res_method,
        "align_to_input_ext_scale": bool(args.align_to_input_ext_scale),
        "use_ray_pose": bool(args.use_ray_pose),
        "ref_view_strategy": args.ref_view_strategy,
        "selected_images": [
            {
                "image_path": str(record["image_path"]),
                "image_timestamp_ns": record["image_timestamp_ns"],
                "pose_timestamp_ns": record["pose_timestamp_ns"],
            }
            for record in selected_records
        ],
        "results": results,
    }

    write_json(args.output_json, payload)
    write_csv(args.output_csv, results)

    print(f"\nSaved JSON summary to {args.output_json}")
    print(f"Saved CSV summary to {args.output_csv}")

    return 0 if all(result.get("status") == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
