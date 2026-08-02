# Environment for this reproduction run

A Python 3.12 virtualenv is available at `.venv` (symlinked to the shared base env).
Activate: `source .venv/bin/activate`  — or run scripts with `.venv/bin/python`.

Pre-installed (CPU-only):
  numpy, scipy, sympy, torch (CPU build), scikit-learn, lightning, pandas

You MAY use all of the above. `pip install` is allowed if you need something else.

## Datasets
Datasets are NOT pre-bundled. Downloading them is PART OF THE TASK — do not mark a
claim `inconclusive` merely because a dataset is needed. Fetch into `./data/` via
`wget`/`curl`, `huggingface-cli`, `torchvision.datasets`, or the paper's official
script. Only refuse for data that is genuinely inaccessible (paywalled / restricted
approval you cannot obtain / private). Keep it small + CPU-only to fit the budget.

Do NOT request GPU. If the paper's method needs GPU-only scale, reproduce a small
CPU-feasible proxy and report it as such.
