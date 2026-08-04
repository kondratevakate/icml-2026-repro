## Reproducing

```bash
cd <this dir>
python3 -m venv .venv && .venv/bin/pip install numpy scipy pandas pyarrow
.venv/bin/python verify_claim1.py    # ~15 s
.venv/bin/python verify_claim2.py    # ~25 s
.venv/bin/python verify_claim3.py    # ~19 min
.venv/bin/python verify_claim4.py    # ~15 min
.venv/bin/python verify_claim5.py    # ~9 min
.venv/bin/python verify_claim6.py    # ~5 s
```
Data is already in `data/`; re-download with
`curl -sL -o data/unscreened.parquet https://huggingface.co/datasets/Mabyduck/CLIC2024-test-human-eval/resolve/main/data/unscreened-00000-of-00001.parquet` (and likewise `screened`).

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Reproducing"}\n-->
