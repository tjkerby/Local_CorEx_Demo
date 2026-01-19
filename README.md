# Local CorEx Experiments

This repository houses paper reproduction materials, exploratory notebooks, and ablation scripts that depend on the core `local-corex` package. Keep the package repository focused on reusable library code and place any experiment-specific assets here.

## Structure

```
local-corex-experiments/
├── requirements.txt         # Pins local-corex version plus experiment deps
├── pyproject.toml           # Optional hatch/uv configuration (placeholder)
├── data/                    # Archived experiment artifacts (small only)
└── experiments/
    ├── paper_crime/
    ├── paper_mnist/
    ├── paper_bikes/
    ├── ablation_study/
    └── misc/
```

Each subdirectory should contain a README that explains how to reproduce the corresponding study, including dataset sources and expected outputs. Large raw datasets should live outside of git or be referenced via download scripts.

## Using a Pinned Package Version

To ensure reproducibility, install the core package via:

```bash
pip install local-corex==0.1.0
```

or, when developing both repos side-by-side, you can point to a local wheel built from the main package repository. Keep the pinned version in `requirements.txt` synchronized with the experiments that live here.
