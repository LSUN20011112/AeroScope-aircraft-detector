# GitHub and Streamlit Community Cloud Deployment

This package follows the same deployment pattern as the DIP Road Damage Detection Web App. The only additional step is external model hosting because the AeroScope checkpoint is 173 MB.

## 1. Create the GitHub repository

Create a new repository, for example:

```text
aeroscope-aircraft-detector
```

Upload **the contents inside `WEBAPP_GITHUB_READY`** to the repository root. Do not upload the outer folder as an extra nesting level.

The repository should show `app.py`, `requirements.txt`, `packages.txt`, and `README.md` at its top level.

## 2. Publish the model as a GitHub Release

1. In the repository, open **Releases** and select **Draft a new release**.
2. Create tag `v1.0-model` and title it `AeroScope Stage-2 Model`.
3. Upload this local file:

```text
D:\RS_PRO\WEBAPP_READY\models\stage2_cors_add_hbb_best.pt
```

4. Publish the release.
5. Copy the release asset's direct download URL. It normally follows this pattern:

```text
https://github.com/USERNAME/REPOSITORY/releases/download/v1.0-model/stage2_cors_add_hbb_best.pt
```

## 3. Create the Streamlit Cloud app

1. Sign in to https://share.streamlit.io with GitHub.
2. Select **Create app**.
3. Choose the AeroScope repository and its default branch.
4. Set the entrypoint to `app.py`.
5. Choose Python 3.12 in Advanced settings when the option is available.

Before deploying, add the following to **App settings → Secrets**:

```toml
MODEL_URL = "PASTE_THE_DIRECT_RELEASE_ASSET_URL_HERE"
MODEL_SHA256 = "30f995545854aa13496adc67a9731b560c29ac57acc10f865e50ee5098b188fe"
```

Then deploy the app.

## 4. First-run behavior

On the first detection request, the application:

1. Downloads the 173 MB checkpoint from `MODEL_URL`.
2. Verifies its SHA-256 checksum.
3. Downloads the pinned official YOLOv5 source revision.
4. Loads the model and runs tiled inference.

The model and source remain cached for the life of the Streamlit server process. A sleeping/restarted Community Cloud instance may download them again.

## 5. Local testing of the GitHub package

Create a local environment and install dependencies:

```powershell
cd D:\RS_PRO\WEBAPP_GITHUB_READY
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Copy the model from the complete local package:

```powershell
Copy-Item D:\RS_PRO\WEBAPP_READY\models\stage2_cors_add_hbb_best.pt .\models\stage2_cors_add_hbb_best.pt
```

Run:

```powershell
python -m streamlit run app.py
```

The copied checkpoint remains ignored by Git because `models/*.pt` is listed in `.gitignore`.

## 6. Files that must not be committed

```text
.venv/
models/*.pt
vendor/
.cache/
__pycache__/
.streamlit/secrets.toml
```

Never commit private URLs, tokens, passwords, or Streamlit secrets.
