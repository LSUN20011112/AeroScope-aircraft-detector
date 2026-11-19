# Model checkpoint

The 173 MB Stage-2 checkpoint is intentionally excluded from the Git repository.

For local execution, copy:

```text
stage2_cors_add_hbb_best.pt
```

into this folder.

For Streamlit Community Cloud, upload the checkpoint to a GitHub Release or Hugging Face and configure its direct download URL as `MODEL_URL` in Streamlit Secrets. `runtime_assets.py` downloads the file on first use and verifies this SHA-256 value:

```text
30f995545854aa13496adc67a9731b560c29ac57acc10f865e50ee5098b188fe
```
