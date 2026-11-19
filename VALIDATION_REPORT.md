# Deployment Package Validation

Validated on 2026-11-18.

## Passed checks

- Six Python source/test files parse successfully.
- Seven utility and runtime-asset tests pass.
- No GitHub upload file exceeds 25 MiB.
- No `.pt` checkpoint, virtual environment, YOLOv5 source clone, or cache is included.
- Runtime asset manager recognizes cached/deployed assets.
- Stage-2 checkpoint SHA-256 verification succeeds.
- Linux/Colab checkpoint loads successfully on Windows.
- Model class metadata is `{0: 'aircraft'}`.
- Complete deployment-style inference reproduces:
  - 30 Houston tiles
  - 97 raw tile detections
  - 63 final detections after Global NMS

## Test command

```powershell
python -m unittest discover -s tests -v
```

The public deployment still requires a real `MODEL_URL` in Streamlit Secrets. Follow `DEPLOYMENT_GUIDE.md` after creating the GitHub repository and model release.
