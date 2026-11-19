# Upload Checklist

Upload every item inside this folder to the root of the dedicated AeroScope GitHub repository.

Before committing, confirm that the repository root contains:

- `app.py`
- `runtime_assets.py`
- `requirements.txt`
- `packages.txt`
- `README.md`
- `.streamlit/config.toml`

Confirm that it does **not** contain:

- `.venv/`
- any `.pt` file
- `vendor/yolov5/`
- `.cache/` or `__pycache__/`
- `.streamlit/secrets.toml`

After the source files are uploaded:

1. Create the GitHub Release and upload the 173 MB checkpoint from the complete local Web App folder.
2. Copy the release asset's direct download URL.
3. Create the Streamlit Community Cloud app with `app.py` as the entrypoint.
4. Add `MODEL_URL` and `MODEL_SHA256` in Streamlit Secrets.
5. Deploy and test the Houston example.
