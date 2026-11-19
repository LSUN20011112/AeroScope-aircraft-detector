# AeroScope Remote-Sensing Aircraft Detection Web App

A bilingual Streamlit application that uses the final Stage-2 YOLOv5x checkpoint from the sequential transfer-learning project to detect aircraft in user-uploaded remote-sensing imagery.

**Live application:** deployment pending

## Features

- Switch the complete interface between Chinese and English.
- Upload JPG, JPEG, or PNG remote-sensing images.
- Use the included Houston Hobby Airport image as a demonstration.
- Divide large images into overlapping tiles for inference.
- Map tile detections back to full-image coordinates.
- Apply Global NMS to remove duplicate detections across overlapping tiles.
- Adjust confidence, within-tile IoU, Global NMS IoU, tile size, overlap, batch size, and device.
- Compare the original and annotated images.
- Download the annotated JPEG, final detection CSV, and raw tile-detection CSV.
- Read the project workflow, operation guide, inference explanation, metrics, and limitations on a separate page.

## Project structure

```text
WEBAPP_GITHUB_READY/
├── app.py
├── inference.py
├── i18n.py
├── runtime_assets.py
├── examples/
│   ├── Houston_Hobby_USDA_NAIP_2022-09-14.png
│   └── Houston_Hobby_USDA_NAIP_2022-09-14_detections.jpg
├── metadata/
│   ├── model_info.json
│   └── training_environment.json
├── models/
│   └── README.md
├── tests/
│   └── test_inference_utils.py
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── requirements.txt
├── packages.txt
├── .python-version
├── .gitignore
├── MODEL_CARD.md
├── DEPLOYMENT_GUIDE.md
└── THIRD_PARTY_NOTICES.md
```

## Model workflow

```text
COCO-pretrained YOLOv5x
→ HRPlanesv2 fine-tuning
→ HRPlanesv2 validation checkpoint selection
→ HRPlanesv2 held-out test
→ CORS-ADD-HBB transfer learning
→ CORS internal-validation checkpoint selection
→ official CORS held-out final test
→ Houston external qualitative inference
→ AeroScope Web App
```

Validation selects the best epoch-specific checkpoint during each training stage. The held-out test sets are used only for final evaluation. The Web App uses the final Stage-2 checkpoint.

## Final performance

| Evaluation split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| HRPlanesv2 test | 0.988 | 0.984 | 0.991 | 0.796 |
| CORS-ADD-HBB held-out test | 0.946 | 0.905 | 0.937 | 0.534 |

The included 3,000 × 2,500 Houston demonstration produces 97 raw tile detections and 63 final detections after Global NMS. Because that image does not have complete manual ground truth, it is reported as qualitative external inference rather than a source of Precision, Recall, F1, or mAP.

## Model checkpoint

The final checkpoint is approximately 173 MB and therefore is not stored as an ordinary Git repository file. At deployment time, `runtime_assets.py` downloads it from the `MODEL_URL` configured in Streamlit Secrets and verifies its SHA-256 checksum.

```text
Filename: stage2_cors_add_hbb_best.pt
SHA-256: 30f995545854aa13496adc67a9731b560c29ac57acc10f865e50ee5098b188fe
```

Recommended hosting options:

- GitHub Release asset
- Hugging Face model repository
- another public HTTPS object-storage URL

## Run locally

Python 3.12 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Copy the final checkpoint into:

```text
models/stage2_cors_add_hbb_best.pt
```

Then launch:

```powershell
python -m streamlit run app.py
```

Streamlit normally opens `http://localhost:8501`.

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this folder.
2. Upload the final checkpoint as a GitHub Release asset or to Hugging Face.
3. Sign in at https://share.streamlit.io with GitHub.
4. Create an app from this repository with `app.py` as the entrypoint.
5. Add these values to Streamlit Secrets:

```toml
MODEL_URL = "DIRECT_MODEL_DOWNLOAD_URL"
MODEL_SHA256 = "30f995545854aa13496adc67a9731b560c29ac57acc10f865e50ee5098b188fe"
```

6. Deploy the app.

See `DEPLOYMENT_GUIDE.md` for the complete DIP-style workflow.

## Default inference configuration

- Architecture: YOLOv5x
- Detection class: aircraft
- Bounding-box type: HBB
- Tile size: 640 × 640
- Tile overlap: 25%
- Confidence threshold: 0.25
- Within-tile NMS IoU: 0.45
- Global NMS IoU: 0.40

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Important limitations

This is an academic demonstration, not a certified aviation, mapping, or surveillance system. Predictions may include false positives or false negatives, especially for imagery outside the training distribution. Review `MODEL_CARD.md` and `THIRD_PARTY_NOTICES.md` before public or commercial deployment.
