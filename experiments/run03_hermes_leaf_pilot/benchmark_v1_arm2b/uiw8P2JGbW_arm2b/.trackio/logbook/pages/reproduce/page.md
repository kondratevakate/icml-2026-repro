## Reproduce
```bash
python3 -m venv venv && ./venv/bin/pip install numpy scipy sympy
for n in 1 2 3 4 5 6; do ./venv/bin/python verify_claim$n.py; done   # ~15 min total, CPU only
```

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Reproduce"}\n-->
