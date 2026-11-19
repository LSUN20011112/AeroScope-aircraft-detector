"""AeroScope bilingual Streamlit application."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from i18n import tr
from inference import (
    dataframe_to_csv_bytes,
    decode_image_bytes,
    detect_aircraft,
    image_to_jpeg_bytes,
    load_yolov5_model,
)
from runtime_assets import ensure_runtime_assets, runtime_status


APP_ROOT = Path(__file__).resolve().parent
DEMO_IMAGE = APP_ROOT / "examples" / "Houston_Hobby_USDA_NAIP_2022-09-14.png"
DEMO_RESULT = APP_ROOT / "examples" / "Houston_Hobby_USDA_NAIP_2022-09-14_detections.jpg"
MODEL_INFO_PATH = APP_ROOT / "metadata" / "model_info.json"


st.set_page_config(
    page_title="AeroScope",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink:#132238; --muted:#5d6b7d; --blue:#0b5cab; --sky:#eaf4ff; --line:#dce6f0; }
      .stApp { background:#ffffff; color:var(--ink); }
      html, body, [class*="css"] { font-size:16px; }
      h1, h2, h3 { color:var(--ink); letter-spacing:-0.02em; }
      h1 { font-size:2.35rem !important; margin-bottom:0.2rem !important; }
      .hero { padding:1.3rem 1.5rem; border:1px solid var(--line); border-radius:16px;
              background:linear-gradient(135deg,#f8fbff 0%,#eef6ff 100%); margin-bottom:1.2rem; }
      .hero-kicker { color:var(--blue); font-weight:700; text-transform:uppercase; letter-spacing:.08em; font-size:.82rem; }
      .hero p { color:var(--muted); max-width:900px; margin-bottom:0; }
      .status-ready { padding:.72rem 1rem; border-radius:10px; background:#ecf8f0; border:1px solid #b8dfc5; color:#155c31; }
      .status-warn { padding:.72rem 1rem; border-radius:10px; background:#fff7e6; border:1px solid #ecd29c; color:#754c00; }
      .flow { display:flex; flex-wrap:wrap; align-items:center; gap:.55rem; margin:1rem 0 1.4rem 0; }
      .flow-step { border:1px solid #c8d9ea; border-radius:11px; background:#f7fbff; padding:.72rem .85rem; font-weight:650; color:#183a5d; }
      .flow-arrow { color:#66829e; font-weight:700; }
      .small-note { color:var(--muted); font-size:.92rem; }
      [data-testid="stMetricValue"] { color:var(--blue); }
      [data-testid="stSidebar"] { background:#f7f9fc; border-right:1px solid var(--line); }
      .stDownloadButton button, .stButton button { min-height:2.65rem; border-radius:9px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_language() -> str:
    current = st.session_state.get("language_code", "zh")
    selected = st.sidebar.selectbox(
        "语言 / Language",
        ["中文", "English"],
        index=0 if current == "zh" else 1,
        key="language_selector",
    )
    language = "zh" if selected == "中文" else "en"
    st.session_state.language_code = language
    return language


@st.cache_resource(show_spinner=False)
def get_model(weights_path: str, yolov5_root: str, device: str):
    return load_yolov5_model(Path(weights_path), Path(yolov5_root), device)


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but PyTorch cannot access a CUDA GPU.")
        return "cuda:0" if torch.cuda.is_available() and requested in {"auto", "cuda"} else "cpu"
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. Run python setup_project.py first.") from exc


def hero(language: str, guide: bool = False) -> None:
    title = tr(language, "about_title" if guide else "app_title")
    subtitle = tr(language, "about_intro" if guide else "app_subtitle")
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">AeroScope · YOLOv5x · HBB</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_selected_image(language: str):
    st.subheader(tr(language, "input_heading"))
    use_demo = st.checkbox(tr(language, "use_demo"), value=True)
    uploaded = st.file_uploader(
        tr(language, "upload"),
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        filename = uploaded.name
    elif use_demo and DEMO_IMAGE.is_file():
        image_bytes = DEMO_IMAGE.read_bytes()
        filename = DEMO_IMAGE.name
    else:
        return None, None, None

    image = decode_image_bytes(image_bytes, filename)
    fingerprint = hashlib.sha256(image_bytes).hexdigest()
    st.caption(tr(language, "image_info", width=image.width, height=image.height))
    return image, filename, fingerprint


def detection_page(language: str) -> None:
    hero(language)
    status = runtime_status(APP_ROOT)
    runtime_ready = (
        (status["local_model"] or status["cached_model"])
        and (status["local_yolov5"] or status["cached_yolov5"])
    )
    runtime_configured = (
        status["local_model"]
        or status["cached_model"]
        or status["model_url_configured"]
    )
    if runtime_ready:
        st.markdown(f'<div class="status-ready">● {tr(language, "model_ready")}</div>', unsafe_allow_html=True)
    elif runtime_configured:
        st.markdown(f'<div class="status-warn">● {tr(language, "runtime_download")}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="status-warn">● {tr(language, "model_missing")}<br>{tr(language, "setup_hint")}</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    image, filename, fingerprint = load_selected_image(language)
    if image is None:
        st.info(tr(language, "no_image"))
        return

    if st.session_state.get("input_fingerprint") != fingerprint:
        st.session_state.input_fingerprint = fingerprint
        st.session_state.pop("detection_output", None)
        st.session_state.pop("detection_filename", None)

    with st.expander(tr(language, "settings"), expanded=True):
        first, second, third = st.columns(3)
        with first:
            confidence = st.slider(tr(language, "confidence"), 0.05, 0.90, 0.25, 0.05)
            detection_iou = st.slider(tr(language, "det_iou"), 0.10, 0.90, 0.45, 0.05)
        with second:
            global_iou = st.slider(tr(language, "global_iou"), 0.10, 0.90, 0.40, 0.05)
            tile_size = st.selectbox(tr(language, "tile_size"), [512, 640, 768, 1024], index=1)
        with third:
            overlap = st.slider(tr(language, "overlap"), 0.00, 0.50, 0.25, 0.05)
            batch_size = st.selectbox(tr(language, "batch_size"), [1, 2, 4, 8], index=2)
            device = st.selectbox(tr(language, "device"), ["auto", "cpu", "cuda"], index=0)

    preview_left, preview_right = st.columns([3, 2])
    with preview_left:
        st.image(image, caption=tr(language, "original"), width="stretch")
    with preview_right:
        if filename == DEMO_IMAGE.name and DEMO_RESULT.is_file():
            st.image(DEMO_RESULT, caption="Houston reference result · 63 detections", width="stretch")

    run_clicked = st.button(
        tr(language, "run"),
        type="primary",
        width="stretch",
        disabled=not runtime_configured,
    )

    if run_clicked:
        progress = st.progress(0.0, text=tr(language, "running"))

        def update_progress(current: int, total: int) -> None:
            progress.progress(
                min(1.0, current / max(1, total)),
                text=tr(language, "progress", current=current, total=total),
            )

        try:
            selected_device = resolve_device(device)
            weights_path, yolov5_root = ensure_runtime_assets(APP_ROOT)
            model = get_model(str(weights_path), str(yolov5_root), selected_device)
            output = detect_aircraft(
                image,
                model,
                confidence_threshold=confidence,
                detection_iou_threshold=detection_iou,
                global_nms_iou_threshold=global_iou,
                tile_size=tile_size,
                overlap=overlap,
                batch_size=batch_size,
                progress_callback=update_progress,
            )
            st.session_state.detection_output = output
            st.session_state.detection_filename = Path(filename).stem
            progress.empty()
        except Exception as exc:  # Display a useful local-app diagnostic without hiding details.
            progress.empty()
            st.error(f"{tr(language, 'error')}: {exc}")
            with st.expander("Technical details"):
                st.exception(exc)

    output = st.session_state.get("detection_output")
    if output is None:
        return

    st.divider()
    st.subheader(tr(language, "result_heading"))
    metrics = st.columns(4)
    metrics[0].metric(tr(language, "raw_count"), len(output.raw_detections))
    metrics[1].metric(tr(language, "final_count"), len(output.final_detections))
    metrics[2].metric(tr(language, "tile_count"), output.tile_count)
    metrics[3].metric(tr(language, "runtime"), tr(language, "seconds", value=output.elapsed_seconds))

    if output.final_detections.empty:
        st.warning(tr(language, "no_detections"))
    tabs = st.tabs([tr(language, "annotated"), tr(language, "original"), tr(language, "table")])
    with tabs[0]:
        st.image(output.annotated_image, width="stretch")
    with tabs[1]:
        st.image(image, width="stretch")
    with tabs[2]:
        display_columns = [
            "detection_id", "confidence", "x1", "y1", "x2", "y2", "box_width", "box_height"
        ]
        st.dataframe(output.final_detections[display_columns], width="stretch", hide_index=True)

    stem = st.session_state.get("detection_filename", "aircraft_detection")
    downloads = st.columns(3)
    downloads[0].download_button(
        tr(language, "download_image"),
        image_to_jpeg_bytes(output.annotated_image),
        file_name=f"{stem}_detections.jpg",
        mime="image/jpeg",
        width="stretch",
    )
    downloads[1].download_button(
        tr(language, "download_csv"),
        dataframe_to_csv_bytes(output.final_detections),
        file_name=f"{stem}_final_detections.csv",
        mime="text/csv",
        width="stretch",
    )
    downloads[2].download_button(
        tr(language, "download_raw_csv"),
        dataframe_to_csv_bytes(output.raw_detections),
        file_name=f"{stem}_raw_tile_detections.csv",
        mime="text/csv",
        width="stretch",
    )
    st.caption(tr(language, "privacy"))


def workflow_html(language: str) -> str:
    keys = [
        "step_coco", "step_hr", "step_hr_test", "step_cors", "step_cors_test", "step_houston", "step_app"
    ]
    parts = []
    for index, key in enumerate(keys):
        parts.append(f'<div class="flow-step">{tr(language, key)}</div>')
        if index < len(keys) - 1:
            parts.append('<div class="flow-arrow">↓</div>')
    return '<div class="flow">' + "".join(parts) + "</div>"


def guide_page(language: str) -> None:
    hero(language, guide=True)
    tabs = st.tabs(
        [
            tr(language, "workflow_tab"),
            tr(language, "usage_tab"),
            tr(language, "inference_tab"),
            tr(language, "model_tab"),
        ]
    )

    with tabs[0]:
        st.markdown(workflow_html(language), unsafe_allow_html=True)
        st.write(tr(language, "workflow_text"))
        st.info(
            "HRPlanesv2: 1,490 train / 420 validation / 210 test\n\n"
            "CORS-ADD-HBB: 3,199 train / 565 internal validation / 1,722 official held-out test"
        )

    with tabs[1]:
        for index, step in enumerate(tr(language, "usage_steps"), start=1):
            st.markdown(f"**{index}.** {step}")
        st.markdown("---")
        st.caption(tr(language, "privacy"))

    with tabs[2]:
        inference_keys = [
            ("zh", "上传影像", "Uploaded image"),
            ("tile", "重叠切片", "Overlapping tiling"),
            ("model", "YOLOv5x Tile 推理", "YOLOv5x tile inference"),
            ("map", "映射回原图坐标", "Map to global coordinates"),
            ("nms", "Global NMS 去重", "Global NMS"),
            ("out", "最终 HBB 检测结果", "Final HBB detections"),
        ]
        labels = [zh if language == "zh" else en for _, zh, en in inference_keys]
        st.markdown(
            '<div class="flow">'
            + '<div class="flow-arrow">↓</div>'.join(
                f'<div class="flow-step">{label}</div>' for label in labels
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        st.write(tr(language, "inference_text"))
        if DEMO_RESULT.is_file():
            st.image(DEMO_RESULT, caption="Houston Hobby Airport · 63 final detections", width="stretch")

    with tabs[3]:
        st.write(tr(language, "metrics_note"))
        metrics = pd.DataFrame(
            [
                ["HRPlanesv2 test", 210, 0.988, 0.984, 0.991, 0.796],
                ["CORS-ADD-HBB held-out test", 1722, 0.946, 0.905, 0.937, 0.534],
            ],
            columns=["Evaluation split", "Images", "Precision", "Recall", "mAP50", "mAP50-95"],
        )
        st.dataframe(metrics, width="stretch", hide_index=True)
        if MODEL_INFO_PATH.is_file():
            with st.expander("Model metadata"):
                st.json(json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8")))
        st.subheader(tr(language, "limitations"))
        st.write(tr(language, "limitations_text"))


language = get_language()
st.sidebar.markdown("### AeroScope")
page_options = ["detect", "guide"]
page = st.sidebar.radio(
    tr(language, "page"),
    page_options,
    key="page_selector",
    format_func=lambda value: tr(language, "detect_page" if value == "detect" else "guide_page"),
)
st.sidebar.caption("YOLOv5x · HRPlanesv2 → CORS-ADD-HBB · HBB")

if page == "detect":
    detection_page(language)
else:
    guide_page(language)

