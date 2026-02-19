# Local CorEx Demo (MNIST)

This repo is designed for students to explore MNIST clustering, CorEx training, and ablation experiments through a Streamlit app.

## Quickstart (recommended: uv)

**Prereqs:** Python 3.10–3.12 and Git.

1. Clone the repo.
2. Install dependencies and CPU PyTorch builds with uv.
3. Start the Streamlit app.

```bash
git clone <YOUR_REPO_URL>
cd Local_CorEx_Demo

# install dependencies (uses pyproject.toml + uv.lock)
uv sync

# run the app
uv run streamlit run streamlit_app.py
```

## Alternative (pip + venv)

```bash
git clone <YOUR_REPO_URL>
cd Local_CorEx_Demo

python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# install local-corex from the included repo and app deps
pip install -e ./local-corex-repo
pip install -r streamlit_requirements.txt

# run the app
streamlit run streamlit_app.py
```

## Dataset download

MNIST is downloaded automatically on the first run (to the MNIST/ folder). An internet connection is required the first time.

## Notes

- This repo includes local-corex in local-corex-repo/ so students can run everything without cloning an extra package.
- If PyTorch install issues occur, use CPU wheels:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## What to open

Start with the Streamlit app (streamlit_app.py). The notebooks in paper_mnist/ are available for deeper dives.
