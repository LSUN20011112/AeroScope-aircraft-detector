"""Tiled YOLOv5 inference and global non-maximum suppression utilities."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
import pathlib
from pathlib import Path
import sys
import time
import types
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


FINAL_COLUMNS = [
    "detection_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "box_width",
    "box_height",
    "tile_name",
]


@dataclass
class DetectionOutput:
    raw_detections: pd.DataFrame
    final_detections: pd.DataFrame
    annotated_image: Image.Image
    tile_count: int
    elapsed_seconds: float


def _channel_to_uint8(channel: np.ndarray) -> np.ndarray:
    """Convert an integer/float image channel to display-ready uint8."""
    channel = np.asarray(channel)
    if channel.dtype == np.uint8:
        return channel
    if channel.dtype == np.bool_:
        return channel.astype(np.uint8) * 255

    values = channel.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)

    valid = values[finite]
    low, high = np.percentile(valid, (2.0, 98.0))
    if high <= low:
        fill = float(np.clip(valid[0], 0.0, 255.0))
        return np.full(values.shape, round(fill), dtype=np.uint8)

    scaled = (values - low) * (255.0 / (high - low))
    scaled[~finite] = 0.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def decode_image_bytes(image_bytes: bytes, filename: str = "uploaded image") -> Image.Image:
    """Decode JPG/PNG/TIFF bytes and return a three-channel RGB image.

    High-bit-depth grayscale and RGB TIFF channels are contrast-stretched from
    their 2nd to 98th percentiles. The first frame/page is used for multi-page
    TIFF files. Raster georeferencing metadata is not retained.
    """
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if getattr(source, "n_frames", 1) > 1:
                source.seek(0)
            source.load()
            array = np.asarray(source)

            if array.dtype == np.uint8 and source.mode in {"RGB", "RGBA"}:
                return source.convert("RGB").copy()
            if array.dtype == np.uint8 and source.mode not in {"I", "F", "I;16"}:
                return source.convert("RGB").copy()
    except Exception as exc:
        raise ValueError(f"Unable to decode {filename}: {exc}") from exc

    if array.ndim == 2:
        gray = _channel_to_uint8(array)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
    elif array.ndim == 3:
        # Pillow normally returns HWC. The second branch also accommodates
        # uncommon channel-first arrays produced by some TIFF encoders.
        if array.shape[-1] in {1, 2, 3, 4}:
            channels = array[..., : min(3, array.shape[-1])]
        elif array.shape[0] in {1, 2, 3, 4}:
            channels = np.moveaxis(array[: min(3, array.shape[0])], 0, -1)
        else:
            raise ValueError(
                f"Unable to decode {filename}: unsupported TIFF band layout {array.shape}."
            )
        if channels.shape[-1] == 1:
            channels = np.repeat(channels, 3, axis=-1)
        elif channels.shape[-1] == 2:
            channels = np.repeat(channels[..., :1], 3, axis=-1)
        rgb = np.stack(
            [_channel_to_uint8(channels[..., index]) for index in range(3)],
            axis=-1,
        )
    else:
        raise ValueError(
            f"Unable to decode {filename}: unsupported image dimensions {array.shape}."
        )

    return Image.fromarray(rgb)


def axis_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """Return tile origins that cover an axis and always include its far edge."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    final_position = length - tile_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def generate_tiles(
    image: Image.Image,
    tile_size: int,
    overlap: float,
) -> list[dict]:
    """Create overlapping PIL tiles and retain full-image coordinate metadata."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    image = image.convert("RGB")
    width, height = image.size
    stride = max(1, round(tile_size * (1 - overlap)))
    x_positions = axis_positions(width, tile_size, stride)
    y_positions = axis_positions(height, tile_size, stride)

    tiles = []
    for row_index, y0 in enumerate(y_positions):
        for column_index, x0 in enumerate(x_positions):
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            tiles.append(
                {
                    "name": f"tile_r{row_index:03d}_c{column_index:03d}",
                    "image": image.crop((x0, y0, x1, y1)),
                    "x0": x0,
                    "y0": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                }
            )
    return tiles


def numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Pure NumPy NMS so global de-duplication is independent of torchvision ops."""
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= iou_threshold]
    return keep


def apply_global_nms(detections: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    """Apply class-aware NMS after mapping tile boxes to full-image coordinates."""
    if detections.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)
    kept_frames = []
    for _, frame in detections.groupby("class_id", sort=True):
        keep = numpy_nms(
            frame[["x1", "y1", "x2", "y2"]].to_numpy(),
            frame["confidence"].to_numpy(),
            iou_threshold,
        )
        kept_frames.append(frame.iloc[keep])
    final = pd.concat(kept_frames, ignore_index=True)
    final = final.sort_values("confidence", ascending=False).reset_index(drop=True)
    final.insert(0, "detection_id", np.arange(1, len(final) + 1))
    final["box_width"] = final["x2"] - final["x1"]
    final["box_height"] = final["y2"] - final["y1"]
    return final[FINAL_COLUMNS]


def load_yolov5_model(weights_path: Path, yolov5_root: Path, device: str = "cpu"):
    """Load a local YOLOv5 custom checkpoint through the official hub entry point."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")
    if not (yolov5_root / "hubconf.py").is_file():
        raise FileNotFoundError(
            f"YOLOv5 source not found: {yolov5_root}. Run setup_project.py first."
        )
    local_config_root = yolov5_root.parent.parent / ".cache"
    yolov5_config = local_config_root / "yolov5"
    ultralytics_config = local_config_root / "ultralytics"
    yolov5_config.mkdir(parents=True, exist_ok=True)
    ultralytics_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLOV5_CONFIG_DIR", str(yolov5_config))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_config))

    import torch

    root_string = str(yolov5_root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    # The checkpoint was produced by Python 3.13 in Linux/Colab, where concrete
    # pathlib classes are serialized from ``pathlib._local``. Python 3.12 has
    # no such submodule, while Windows cannot instantiate PosixPath. Install a
    # temporary compatibility module/class mapping only during unpickling.
    original_posix_path = pathlib.PosixPath
    original_pure_posix_path = pathlib.PurePosixPath
    existing_local_module = sys.modules.get("pathlib._local")
    created_local_shim = existing_local_module is None
    had_local_attribute = hasattr(pathlib, "_local")

    if created_local_shim:
        pathlib_local = types.ModuleType("pathlib._local")
        pathlib_local.Path = pathlib.Path
        pathlib_local.PurePath = pathlib.PurePath
        pathlib_local.WindowsPath = pathlib.WindowsPath
        pathlib_local.PureWindowsPath = pathlib.PureWindowsPath
        pathlib_local.PosixPath = (
            pathlib.WindowsPath if os.name == "nt" else pathlib.PosixPath
        )
        pathlib_local.PurePosixPath = (
            pathlib.PureWindowsPath if os.name == "nt" else pathlib.PurePosixPath
        )
        sys.modules["pathlib._local"] = pathlib_local
        setattr(pathlib, "_local", pathlib_local)
    else:
        pathlib_local = existing_local_module

    original_local_posix_path = getattr(pathlib_local, "PosixPath", None)
    original_local_pure_posix_path = getattr(pathlib_local, "PurePosixPath", None)
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
        pathlib.PurePosixPath = pathlib.PureWindowsPath
        pathlib_local.PosixPath = pathlib.WindowsPath
        pathlib_local.PurePosixPath = pathlib.PureWindowsPath
    try:
        model = torch.hub.load(
            root_string,
            "custom",
            path=str(weights_path.resolve()),
            source="local",
            device=device,
            force_reload=False,
            verbose=False,
        )
    finally:
        pathlib.PosixPath = original_posix_path
        pathlib.PurePosixPath = original_pure_posix_path
        if created_local_shim:
            if sys.modules.get("pathlib._local") is pathlib_local:
                del sys.modules["pathlib._local"]
            if not had_local_attribute and getattr(pathlib, "_local", None) is pathlib_local:
                delattr(pathlib, "_local")
        else:
            pathlib_local.PosixPath = original_local_posix_path
            pathlib_local.PurePosixPath = original_local_pure_posix_path
    model.eval()
    return model


def _batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def detect_aircraft(
    image: Image.Image,
    model,
    *,
    confidence_threshold: float = 0.25,
    detection_iou_threshold: float = 0.45,
    global_nms_iou_threshold: float = 0.40,
    tile_size: int = 640,
    overlap: float = 0.25,
    batch_size: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DetectionOutput:
    """Run overlapping-tile inference and return full-image HBB detections."""
    started = time.perf_counter()
    image = image.convert("RGB")
    full_width, full_height = image.size
    tiles = generate_tiles(image, tile_size, overlap)

    model.conf = float(confidence_threshold)
    model.iou = float(detection_iou_threshold)
    model.max_det = 2000

    raw_rows: list[dict] = []
    processed = 0
    class_names = model.names if hasattr(model, "names") else {0: "aircraft"}

    for batch in _batched(tiles, max(1, int(batch_size))):
        batch_images = [item["image"] for item in batch]
        results = model(batch_images, size=tile_size)
        predictions = results.xyxy

        for tile, prediction in zip(batch, predictions):
            values = prediction.detach().cpu().numpy()
            for x1, y1, x2, y2, confidence, class_id in values:
                class_index = int(class_id)
                if isinstance(class_names, dict):
                    class_name = str(class_names.get(class_index, class_index))
                else:
                    class_name = str(class_names[class_index])
                raw_rows.append(
                    {
                        "class_id": class_index,
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "x1": max(0.0, float(x1) + tile["x0"]),
                        "y1": max(0.0, float(y1) + tile["y0"]),
                        "x2": min(float(full_width), float(x2) + tile["x0"]),
                        "y2": min(float(full_height), float(y2) + tile["y0"]),
                        "tile_name": tile["name"],
                    }
                )
        processed += len(batch)
        if progress_callback:
            progress_callback(processed, len(tiles))

    raw_columns = [
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "tile_name",
    ]
    raw = pd.DataFrame(raw_rows, columns=raw_columns)
    final = apply_global_nms(raw, global_nms_iou_threshold)
    annotated = annotate_image(image, final)
    return DetectionOutput(
        raw_detections=raw,
        final_detections=final,
        annotated_image=annotated,
        tile_count=len(tiles),
        elapsed_seconds=time.perf_counter() - started,
    )


def annotate_image(image: Image.Image, detections: pd.DataFrame) -> Image.Image:
    """Draw red HBB boxes with confidence labels on a copy of the image."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, min(annotated.size) // 800)
    label_height = max(15, min(annotated.size) // 120)

    for _, detection in detections.iterrows():
        box = tuple(float(detection[key]) for key in ("x1", "y1", "x2", "y2"))
        draw.rectangle(box, outline=(220, 38, 38), width=line_width)
        label = f"#{int(detection['detection_id'])}  {detection['confidence']:.2f}"
        text_x = int(box[0])
        text_y = max(0, int(box[1]) - label_height)
        text_box = draw.textbbox((text_x, text_y), label)
        draw.rectangle(text_box, fill=(220, 38, 38))
        draw.text((text_x, text_y), label, fill="white")
    return annotated


def image_to_jpeg_bytes(image: Image.Image, quality: int = 95) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")

