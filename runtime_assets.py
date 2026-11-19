"""Download and verify runtime assets for local and Streamlit Cloud execution."""

from __future__ import annotations

import hashlib
from functools import lru_cache
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import requests


MODEL_FILENAME = "stage2_cors_add_hbb_best.pt"
DEFAULT_MODEL_SHA256 = "30f995545854aa13496adc67a9731b560c29ac57acc10f865e50ee5098b188fe"
YOLOV5_COMMIT = "cb2f17fd5d790cfdd0c07c25351904df9affbe6e"
YOLOV5_ARCHIVE_URL = f"https://github.com/ultralytics/yolov5/archive/{YOLOV5_COMMIT}.zip"


def _secret(name: str, default: str = "") -> str:
    """Read an environment variable first, then an optional Streamlit secret."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st

        value = str(st.secrets.get(name, default)).strip()
    except Exception:
        value = default
    return value


def get_model_url() -> str:
    return _secret("MODEL_URL")


def get_expected_model_sha256() -> str:
    return _secret("MODEL_SHA256", DEFAULT_MODEL_SHA256).lower()


def cache_root() -> Path:
    configured = os.getenv("AEROSCOPE_CACHE_DIR", "").strip()
    root = Path(configured) if configured else Path(tempfile.gettempdir()) / "aeroscope_runtime"
    root.mkdir(parents=True, exist_ok=True)
    return root


@lru_cache(maxsize=16)
def _sha256_cached(path_string: str, size: int, modified_ns: int) -> str:
    digest = hashlib.sha256()
    path = Path(path_string)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _valid_model(path: Path) -> bool:
    return path.is_file() and sha256_file(path).lower() == get_expected_model_sha256()


def runtime_status(app_root: Path) -> dict[str, bool]:
    local_model = app_root / "models" / MODEL_FILENAME
    local_yolov5 = app_root / "vendor" / "yolov5"
    cached_model = cache_root() / "models" / MODEL_FILENAME
    cached_yolov5 = cache_root() / "yolov5"
    return {
        "local_model": _valid_model(local_model),
        "local_yolov5": (local_yolov5 / "hubconf.py").is_file(),
        "cached_model": _valid_model(cached_model),
        "cached_yolov5": (cached_yolov5 / "hubconf.py").is_file(),
        "model_url_configured": bool(get_model_url()),
    }


def ensure_model(app_root: Path) -> Path:
    """Return a verified local checkpoint, downloading it when necessary."""
    bundled = app_root / "models" / MODEL_FILENAME
    if _valid_model(bundled):
        return bundled

    target = cache_root() / "models" / MODEL_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if _valid_model(target):
        return target

    model_url = get_model_url()
    if not model_url:
        raise RuntimeError(
            "MODEL_URL is not configured. Add the GitHub Release or Hugging Face "
            "download URL to Streamlit secrets."
        )

    partial = target.with_suffix(f".part-{os.getpid()}")
    try:
        with requests.get(
            model_url,
            stream=True,
            timeout=(30, 600),
            allow_redirects=True,
            headers={"User-Agent": "AeroScope/1.0"},
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as destination:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        destination.write(chunk)
        actual = sha256_file(partial).lower()
        expected = get_expected_model_sha256()
        if actual != expected:
            raise RuntimeError(
                f"Downloaded checkpoint failed SHA-256 verification: {actual}"
            )
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()
    return target


def ensure_yolov5_source(app_root: Path) -> Path:
    """Return the pinned YOLOv5 source, downloading the GitHub archive if absent."""
    bundled = app_root / "vendor" / "yolov5"
    if (bundled / "hubconf.py").is_file():
        return bundled

    target = cache_root() / "yolov5"
    marker = target / ".source_commit"
    if (target / "hubconf.py").is_file() and marker.is_file():
        if marker.read_text(encoding="utf-8").strip() == YOLOV5_COMMIT:
            return target

    if target.exists():
        shutil.rmtree(target)
    with tempfile.TemporaryDirectory(prefix="aeroscope_yolov5_", dir=cache_root()) as temp_string:
        temp = Path(temp_string)
        archive_path = temp / "yolov5.zip"
        with requests.get(
            YOLOV5_ARCHIVE_URL,
            stream=True,
            timeout=(30, 300),
            allow_redirects=True,
            headers={"User-Agent": "AeroScope/1.0"},
        ) as response:
            response.raise_for_status()
            with archive_path.open("wb") as destination:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        destination.write(chunk)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp)
        extracted = next(
            path for path in temp.iterdir()
            if path.is_dir() and path.name.startswith("yolov5-")
        )
        shutil.move(str(extracted), str(target))
    marker.write_text(YOLOV5_COMMIT + "\n", encoding="utf-8")
    return target


def ensure_runtime_assets(app_root: Path) -> tuple[Path, Path]:
    """Prepare the verified checkpoint and pinned YOLOv5 source."""
    return ensure_model(app_root), ensure_yolov5_source(app_root)
