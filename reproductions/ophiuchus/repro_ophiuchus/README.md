# Ophiuchus release audit

Run:

```bash
python audit_artifacts.py
python -m pytest -q
```

The runtime smoke test stubs only unavailable `gym` and `ray` imports. It
executes the released worker's local zoom, parser, termination, and reward
logic without calling external segmentation servers or a model.
