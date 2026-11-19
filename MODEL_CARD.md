# AeroScope Model Card

## Model

- Architecture: YOLOv5x
- Task: single-class aircraft detection in remote-sensing imagery
- Output: horizontal bounding boxes (HBB)
- Final checkpoint: `models/stage2_cors_add_hbb_best.pt`

## Training lineage

```text
COCO-pretrained YOLOv5x
→ HRPlanesv2 fine-tuning
→ HRPlanesv2 validation checkpoint selection
→ HRPlanesv2 held-out test
→ CORS-ADD-HBB fine-tuning
→ CORS internal validation checkpoint selection
→ official CORS held-out evaluation
```

The Web App uses the final Stage-2 checkpoint. HRPlanesv2 and CORS validation splits were used for epoch/checkpoint selection, while the held-out test sets were used only for final evaluation.

## Held-out performance

| Dataset | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| HRPlanesv2 test | 0.988 | 0.984 | 0.991 | 0.796 |
| CORS-ADD-HBB held-out test | 0.946 | 0.905 | 0.937 | 0.534 |

## Product inference

The application divides large images into overlapping tiles, runs YOLOv5x on every tile, maps local detections back to full-image coordinates, and applies class-aware Global NMS. It returns an annotated JPEG plus raw and de-duplicated CSV tables.

## Limitations

The model produces HBB detections only. Performance can vary for sensors, locations, spatial resolutions, aircraft types, or weather conditions outside the training distribution. The Houston image is a qualitative external demonstration because it has no complete manually verified ground truth.
