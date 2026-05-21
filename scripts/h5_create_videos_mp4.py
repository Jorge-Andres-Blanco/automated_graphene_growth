#!/usr/bin/env python3
"""Stream HDF5 image frames, center-crop, and save as MP4.

Why this script:
- HDF5 image stacks can be huge (many GB), so loading everything at once is risky.
- This script reads one frame at a time, crops to the center, converts to 8-bit,
  and writes directly to video.

Default behavior:
- Auto-detect image dataset path.
- Crop to 640x640 from image center.
- Process all frames (or selected range).
- Save MP4 using OpenCV.

Example:
    python h5_stream_crop_to_mp4.py /path/to/file.h5
    python h5_stream_crop_to_mp4.py /path/to/file.h5 --out movie.mp4 --start 0 --stop 1000 --step 2
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

try:
    # Optional: registers common HDF5 compression filters (useful on clusters).
    import hdf5plugin  # noqa: F401
except Exception:
    hdf5plugin = None

import cv2
import h5py
import numpy as np


IMAGE_CANDIDATES = [
    "/1.1/measurement/basler",
    "/1.1/instrument/basler/image",
    "/1.1/instrument/basler/data",
]

CH4_CANDIDATES = [
    "/1.1/measurement/CH4",
    "/1.1/instrument/CH4/signal",
]

TIME_CANDIDATES = [
    "/1.1/measurement/elapsed_time",
    "/1.1/instrument/elapsed_time/value",
    "/1.1/measurement/epoch",
    "/1.1/instrument/epoch/value",
]

EPOCH_CANDIDATES = [
    "/1.1/measurement/epoch",
    "/1.1/instrument/epoch/value",
]


def resolve_h5_path(path: Path) -> Path:
    """Resolve input path to one concrete .h5 file."""
    if path.is_file():
        return path

    if path.is_dir():
        files = sorted(path.glob("*.h5"))
        if len(files) == 1:
            return files[0]
        if len(files) == 0:
            raise FileNotFoundError(f"No .h5 files found in directory: {path}")
        raise ValueError(
            f"Multiple .h5 files found in directory {path}. Pass a file path directly."
        )

    raise FileNotFoundError(f"Input path does not exist: {path}")


def find_image_dataset(f: h5py.File, dataset_path: str | None) -> h5py.Dataset:
    """Find and return image dataset either from explicit path or fallback candidates."""
    if dataset_path is not None:
        if dataset_path not in f:
            raise KeyError(f"Dataset path not found: {dataset_path}")
        obj = f[dataset_path]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"Path exists but is not a dataset: {dataset_path}")
        return obj

    for path in IMAGE_CANDIDATES:
        if path in f and isinstance(f[path], h5py.Dataset):
            return f[path]

    raise KeyError(
        "Could not auto-detect image dataset. Tried: " + ", ".join(IMAGE_CANDIDATES)
    )


def find_ch4_dataset(f: h5py.File, dataset_path: str | None) -> h5py.Dataset:
    """Find and return CH4 flow dataset."""
    if dataset_path is not None:
        if dataset_path not in f:
            raise KeyError(f"CH4 dataset path not found: {dataset_path}")
        obj = f[dataset_path]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"CH4 path exists but is not a dataset: {dataset_path}")
        return obj

    for path in CH4_CANDIDATES:
        if path in f and isinstance(f[path], h5py.Dataset):
            return f[path]

    raise KeyError(
        "Could not auto-detect CH4 dataset. Tried: " + ", ".join(CH4_CANDIDATES)
    )


def find_time_dataset(f: h5py.File, dataset_path: str | None) -> h5py.Dataset:
    """Find and return elapsed-time/epoch dataset."""
    if dataset_path is not None:
        if dataset_path not in f:
            raise KeyError(f"Time dataset path not found: {dataset_path}")
        obj = f[dataset_path]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"Time path exists but is not a dataset: {dataset_path}")
        return obj

    for path in TIME_CANDIDATES:
        if path in f and isinstance(f[path], h5py.Dataset):
            return f[path]

    raise KeyError(
        "Could not auto-detect time dataset. Tried: " + ", ".join(TIME_CANDIDATES)
    )


def find_optional_epoch_dataset(f: h5py.File) -> h5py.Dataset | None:
    """Find epoch dataset if present, otherwise return None."""
    for path in EPOCH_CANDIDATES:
        if path in f and isinstance(f[path], h5py.Dataset):
            return f[path]
    return None


def center_crop_2d(frame: np.ndarray, crop_size: int) -> np.ndarray:
    """Return center crop of a 2D frame."""
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D frame, got shape={frame.shape}")

    h, w = frame.shape
    if crop_size > h or crop_size > w:
        raise ValueError(
            f"Crop size {crop_size} is larger than frame size ({h}, {w})."
        )

    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return frame[y0 : y0 + crop_size, x0 : x0 + crop_size]


def maybe_resize_2d(frame: np.ndarray, resize_to: int | None) -> np.ndarray:
    """Resize a 2D frame to (resize_to, resize_to) if requested."""
    if resize_to is None:
        return frame

    if resize_to <= 0:
        raise ValueError("--resize must be >= 1")

    if frame.ndim != 2:
        raise ValueError(f"Expected 2D frame for resize, got shape={frame.shape}")

    if frame.shape == (resize_to, resize_to):
        return frame

    # INTER_AREA is generally a good default for downsampling.
    return cv2.resize(frame, (resize_to, resize_to), interpolation=cv2.INTER_AREA)


def to_uint8(frame: np.ndarray, mode: str) -> np.ndarray:
    """Convert a single-frame array to uint8 for video writing.

    mode:
    - "per-frame": scale each frame by its own min/max (fast, but can flicker).
    - "fixed-4095": fixed scaling 0..4095 -> 0..255 (consistent across frames).
    - "none": only valid for uint8 input; no scaling done.
    """
    if mode == "none":
        if frame.dtype != np.uint8:
            raise ValueError("--scale none requires uint8 image data")
        return frame

    if mode == "fixed-4095":
        f32 = frame.astype(np.float32)
        return np.clip((f32 / 4095.0) * 255.0, 0, 255).astype(np.uint8)

    f32 = frame.astype(np.float32)
    lo = float(np.min(f32))
    hi = float(np.max(f32))

    # Avoid division-by-zero for constant frames.
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.uint8)

    norm = (f32 - lo) / (hi - lo)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def map_frame_to_series_index(frame_idx: int, n_frames: int, n_series: int) -> int:
    """Map frame index to a nearest index in a 1D time series."""
    if n_series <= 1 or n_frames <= 1:
        return 0
    return int(round(frame_idx * (n_series - 1) / (n_frames - 1)))


def draw_ch4_overlay(
    frame_bgr: np.ndarray,
    series: np.ndarray,
    current_idx: int,
    window: int,
    width: int,
    height: int,
    margin: int,
) -> None:
    """Draw trailing CH4 line plot in the bottom-right corner."""
    h, w = frame_bgr.shape[:2]
    plot_w = max(20, min(width, w - 2 * margin))
    plot_h = max(20, min(height, h - 2 * margin))
    x0 = w - margin - plot_w
    y0 = h - margin - plot_h

    start = max(0, current_idx - window + 1)
    seg = np.asarray(series[start : current_idx + 1], dtype=np.float32)
    if seg.size == 0:
        return

    # Dynamic y-range in the visible window improves local contrast.
    y_min = float(np.min(seg))
    y_max = float(np.max(seg))
    if y_max <= y_min:
        y_norm = np.full(seg.shape, 0.5, dtype=np.float32)
    else:
        y_norm = (seg - y_min) / (y_max - y_min)

    xs = np.linspace(x0, x0 + plot_w - 1, num=seg.size, dtype=np.float32)
    ys = y0 + (1.0 - y_norm) * (plot_h - 1)
    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)

    # Draw axes (no background box).
    axis_color = (230, 230, 230)  # BGR
    x_axis_y = y0 + plot_h - 1
    y_axis_x = x0
    cv2.line(
        frame_bgr,
        (y_axis_x, y0),
        (y_axis_x, x_axis_y),
        axis_color,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        frame_bgr,
        (y_axis_x, x_axis_y),
        (x0 + plot_w - 1, x_axis_y),
        axis_color,
        1,
        cv2.LINE_AA,
    )

    # Min/max labels for the y-axis.
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame_bgr,
        f"{y_max:.2f}",
        (x0 + 4, y0 + 11),
        font,
        0.35,
        axis_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"{y_min:.2f}",
        (x0 + 4, x_axis_y - 4),
        font,
        0.35,
        axis_color,
        1,
        cv2.LINE_AA,
    )

    # Only line + current point.
    color = (80, 255, 80)  # BGR
    if len(pts) >= 2:
        cv2.polylines(frame_bgr, [pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
    px, py = int(pts[-1, 0, 0]), int(pts[-1, 0, 1])
    cv2.circle(frame_bgr, (px, py), radius=2, color=color, thickness=-1, lineType=cv2.LINE_AA)


def draw_time_overlay(
    frame_bgr: np.ndarray,
    minutes_value: float,
    clock_time: str,
    frame_number: int,
    total_frames: int,
    margin: int = 16,
) -> None:
    """Draw 3-row time/frame overlay in the bottom-left corner."""
    lines = [
        f"{int(round(minutes_value))} min",
        clock_time,
        f"{frame_number}/{total_frames}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Match the axis tick label sizing used in the CH4 overlay.
    scale = 0.35
    thickness = 1

    # Place three lines so the last line sits at the bottom margin.
    line_gap = 12
    x = margin
    y_start = frame_bgr.shape[0] - margin - line_gap * (len(lines) - 1)

    for i, text in enumerate(lines):
        y = y_start + i * line_gap
        # Black outline for readability over bright frames.
        cv2.putText(
            frame_bgr,
            text,
            (x, y),
            font,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            text,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def open_video_writer(
    out_path: Path,
    fps: float,
    width: int,
    height: int,
 ) -> cv2.VideoWriter:
    """Open an MP4 video writer with common codecs.

    Different systems support different codecs; we try a few.
    """
    codec_candidates = ["mp4v", "avc1", "H264"]

    for codec in codec_candidates:
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
            True,  # color (BGR)
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(
        "Could not open MP4 writer with codecs: " + ", ".join(codec_candidates)
    )


def build_frame_indices(n_frames: int, start: int, stop: int | None, step: int) -> range:
    """Build frame index range with validation."""
    if step <= 0:
        raise ValueError("--step must be >= 1")
    if start < 0:
        raise ValueError("--start must be >= 0")

    if stop is None:
        stop = n_frames

    if start >= n_frames:
        raise ValueError(f"--start={start} out of range for n_frames={n_frames}")
    if stop <= start:
        raise ValueError("--stop must be > --start")

    stop = min(stop, n_frames)
    return range(start, stop, step)


def read_frame(dataset: h5py.Dataset, idx: int) -> np.ndarray:
    """Read one frame from dataset and convert to NumPy array."""
    try:
        if dataset.ndim == 2:
            # Single-image dataset: ignore idx and return the same image.
            return np.asarray(dataset[()])
        return np.asarray(dataset[idx])
    except OSError as exc:
        msg = str(exc)
        if "can't dlopen" in msg or "libhdf5.so" in msg:
            raise OSError(
                f"{msg}\n\n"
                "Likely HDF5 plugin/library mismatch. Try:\n"
                "1) pip install hdf5plugin\n"
                "2) unset HDF5_PLUGIN_PATH\n"
                "3) load a cluster HDF5 module compatible with your Python env\n"
            ) from exc
        raise


def stream_to_mp4(
    h5_path: Path,
    out_path: Path,
    dataset_path: str | None,
    crop_size: int,
    resize_to: int | None,
    start: int,
    stop: int | None,
    step: int,
    fps: float,
    scale_mode: str,
    overlay_ch4: bool,
    ch4_dataset_path: str | None,
    overlay_window: int,
    overlay_width: int,
    overlay_height: int,
    overlay_margin: int,
    overlay_time: bool,
    time_dataset_path: str | None,
) -> None:
    """Stream frames from HDF5 to MP4 with center crop."""
    with h5py.File(h5_path, "r") as f:
        ds = find_image_dataset(f, dataset_path)

        if ds.ndim < 2:
            raise ValueError(f"Image dataset ndim={ds.ndim}; expected >=2")

        # Determine frame count and sample frame shape.
        n_frames = ds.shape[0] if ds.ndim >= 3 else 1
        frame_indices = build_frame_indices(n_frames=n_frames, start=start, stop=stop, step=step)

        ch4_ds_name = None
        ch4_values: np.ndarray | None = None
        if overlay_ch4:
            ch4_ds = find_ch4_dataset(f, ch4_dataset_path)
            ch4_values = np.asarray(ch4_ds[:], dtype=np.float32).ravel()
            if ch4_values.size == 0:
                raise ValueError("CH4 dataset is empty, cannot draw overlay.")
            ch4_ds_name = ch4_ds.name

        time_ds_name = None
        time_values: np.ndarray | None = None
        epoch_ds_name = None
        epoch_values: np.ndarray | None = None
        if overlay_time:
            time_ds = find_time_dataset(f, time_dataset_path)
            time_values = np.asarray(time_ds[:], dtype=np.float64).ravel()
            if time_values.size == 0:
                raise ValueError("Time dataset is empty, cannot draw overlay.")
            time_ds_name = time_ds.name

            # Pick epoch series for clock-time row if available.
            time_name_lower = time_ds_name.lower()
            if "epoch" in time_name_lower or (time_values.size > 0 and time_values[0] > 1e8):
                epoch_values = time_values
                epoch_ds_name = time_ds_name
            else:
                epoch_ds = find_optional_epoch_dataset(f)
                if epoch_ds is not None:
                    epoch_values = np.asarray(epoch_ds[:], dtype=np.float64).ravel()
                    if epoch_values.size > 0:
                        epoch_ds_name = epoch_ds.name
                    else:
                        epoch_values = None

        # Read one frame to infer crop and video shape.
        first_idx = next(iter(frame_indices))
        sample_frame = read_frame(ds, first_idx)
        processed = center_crop_2d(sample_frame, crop_size=crop_size)
        processed = maybe_resize_2d(processed, resize_to=resize_to)
        h, w = processed.shape

        writer = open_video_writer(out_path=out_path, fps=fps, width=w, height=h)
        try:
            written = 0
            for idx in frame_indices:
                frame = read_frame(ds, idx)
                processed = center_crop_2d(frame, crop_size=crop_size)
                processed = maybe_resize_2d(processed, resize_to=resize_to)
                gray8 = to_uint8(processed, mode=scale_mode)

                # OpenCV expects BGR color for video frames.
                bgr = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
                if overlay_ch4 and ch4_values is not None:
                    ch4_idx = map_frame_to_series_index(
                        frame_idx=idx, n_frames=n_frames, n_series=ch4_values.size
                    )
                    draw_ch4_overlay(
                        frame_bgr=bgr,
                        series=ch4_values,
                        current_idx=ch4_idx,
                        window=overlay_window,
                        width=overlay_width,
                        height=overlay_height,
                        margin=overlay_margin,
                    )
                if overlay_time and time_values is not None:
                    time_idx = map_frame_to_series_index(
                        frame_idx=idx, n_frames=n_frames, n_series=time_values.size
                    )
                    # Always show elapsed minutes from start, regardless of absolute epoch base.
                    elapsed_minutes = (time_values[time_idx] - time_values[0]) / 60.0
                    if epoch_values is not None:
                        epoch_idx = map_frame_to_series_index(
                            frame_idx=idx, n_frames=n_frames, n_series=epoch_values.size
                        )
                        actual_clock = datetime.fromtimestamp(
                            float(epoch_values[epoch_idx])
                        ).strftime("%H:%M:%S")
                    else:
                        actual_clock = "--:--:--"
                    draw_time_overlay(
                        bgr,
                        minutes_value=elapsed_minutes,
                        clock_time=actual_clock,
                        frame_number=idx + 1,
                        total_frames=n_frames,
                        margin=16,
                    )
                writer.write(bgr)
                written += 1

                # Light progress print every 200 frames.
                if written % 200 == 0:
                    print(f"Processed {written} frames...")
        finally:
            writer.release()

        print(f"File: {h5_path}")
        print(f"Dataset: {ds.name}")
        print(f"Original dataset shape: {ds.shape}, dtype: {ds.dtype}")
        print(f"Crop size: {crop_size}x{crop_size}")
        if resize_to is not None:
            print(f"Resize: {resize_to}x{resize_to}")
        if overlay_ch4:
            print(f"CH4 overlay: on ({ch4_ds_name}, window={overlay_window})")
        else:
            print("CH4 overlay: off")
        if overlay_time:
            if epoch_ds_name is not None:
                print(f"Time overlay: on (elapsed={time_ds_name}, epoch={epoch_ds_name})")
            else:
                print(f"Time overlay: on (elapsed={time_ds_name}, epoch=not found)")
        else:
            print("Time overlay: off")
        print(f"Frames written: {written}")
        print(f"Saved MP4: {out_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stream HDF5 frames one-by-one, center-crop, and save MP4."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to .h5 file, or a directory containing exactly one .h5 file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("center_crop_640.mp4"),
        help="Output mp4 path (default: center_crop_640.mp4).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional explicit image dataset path. Default: auto-detect.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=640,
        help="Center crop size in pixels (default: 640).",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Optional final square size after crop, e.g. 320.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start frame index (default: 0).",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stop frame index, exclusive (default: all frames).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame step for subsampling (default: 1).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Output video FPS (default: 20).",
    )
    parser.add_argument(
        "--scale",
        type=str,
        choices=["per-frame", "fixed-4095", "none"],
        default="per-frame",
        help=(
            "8-bit conversion mode: per-frame (default), "
            "fixed-4095 (0..4095 -> 0..255), or none (uint8 only)."
        ),
    )
    parser.add_argument(
        "--overlay-ch4",
        action="store_true",
        help="Enable CH4 trailing line overlay in bottom-right corner.",
    )
    parser.add_argument(
        "--ch4-dataset",
        type=str,
        default=None,
        help="Optional explicit CH4 dataset path. Default: auto-detect.",
    )
    parser.add_argument(
        "--overlay-window",
        type=int,
        default=1000,
        help="Number of trailing CH4 points in overlay (default: 1000).",
    )
    parser.add_argument(
        "--overlay-width",
        type=int,
        default=240,
        help="Overlay plot width in pixels (default: 240).",
    )
    parser.add_argument(
        "--overlay-height",
        type=int,
        default=120,
        help="Overlay plot height in pixels (default: 120).",
    )
    parser.add_argument(
        "--overlay-margin",
        type=int,
        default=16,
        help="Margin from frame border in pixels (default: 16).",
    )
    parser.add_argument(
        "--overlay-time",
        action="store_true",
        help="Enable bottom-left elapsed-time text overlay in minutes.",
    )
    parser.add_argument(
        "--time-dataset",
        type=str,
        default=None,
        help="Optional explicit time dataset path. Default: auto-detect elapsed_time/epoch.",
    )
    args = parser.parse_args()

    if args.crop <= 0:
        raise ValueError("--crop must be >= 1")
    if args.resize is not None and args.resize <= 0:
        raise ValueError("--resize must be >= 1")
    if args.overlay_window <= 0:
        raise ValueError("--overlay-window must be >= 1")
    if args.overlay_width <= 0 or args.overlay_height <= 0:
        raise ValueError("--overlay-width and --overlay-height must be >= 1")
    if args.overlay_margin < 0:
        raise ValueError("--overlay-margin must be >= 0")

    h5_file = resolve_h5_path(args.path)
    out_path = args.out if args.out.is_absolute() else Path.cwd() / args.out

    stream_to_mp4(
        h5_path=h5_file,
        out_path=out_path,
        dataset_path=args.dataset,
        crop_size=args.crop,
        resize_to=args.resize,
        start=args.start,
        stop=args.stop,
        step=args.step,
        fps=args.fps,
        scale_mode=args.scale,
        overlay_ch4=args.overlay_ch4,
        ch4_dataset_path=args.ch4_dataset,
        overlay_window=args.overlay_window,
        overlay_width=args.overlay_width,
        overlay_height=args.overlay_height,
        overlay_margin=args.overlay_margin,
        overlay_time=args.overlay_time,
        time_dataset_path=args.time_dataset,
    )


if __name__ == "__main__":
    main()
