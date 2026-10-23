"""Rerun the verdict_audit analyses and check their outputs against the committed results.

Each step runs one analysis script in its own folder, from the committed coding sheets and the downloaded
inputs (see README.md, "Data that are not in this repository"). Before a step, the committed versions of the
files it may write are set aside; after it, every new JSON output is compared with the committed one (parsed,
so formatting does not matter), the new files are copied into logs/<run>/ when they differ, and the committed
files are always put back. The working tree is unchanged afterwards. Language-model coding is not rerun.

Usage: python reproduce.py [--list] [step ...]      (default: every step)
Writes logs/reproduce_<UTC time>.json (environment, git commit, per-step status and duration).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable

# name, folder, commands, compared outputs, other files the step rewrites (restored, not compared),
# and documented keys left out of the comparison with the reason (reported in the log)
STEPS = [
    ("verdicts", ".", [["analyze.py"]], ["results/results.json"], ["results/"], {}),
    ("verdicts-confirmation", ".", [["confirm.py"]], ["results_confirmation/results.json"], ["results_confirmation/"], {}),
    ("healthcare", "healthcare", [["analyze.py"]], ["results.json"], [], {}),
    ("e1", "e1_adjusted", [["analyze_e1.py", "validate"], ["analyze_e1.py", "models"]],
     ["validation_e1.json", "results_e1.json"], ["models_run.log"], {}),
    ("e1-validated", "e1_adjusted", [["e1_validated.py"]], ["results_e1_validated.json"], [], {}),
    ("falsifiability", "falsifiability", [["analyze6.py", "results"]], ["results6.json"], ["results_run.log"], {}),
    ("voters", "voters", [["analyze_voters.py"]], ["results/results.json"], ["results/"], {}),
    ("voters-q3", "voters", [["q3_score.py"]], ["results/q3.json"], [], {}),
    ("voters-exposure", "voters", [["analyze_exposure.py"]], ["results/exposure.json"], ["results/exposure_manifest.json"], {}),
    ("voters-posthoc", "voters", [["posthoc_blog_logbooks.py"]], ["results/posthoc_blog_logbooks.json"], [], {}),
    ("voters-review", "voters", [["review_checks.py"]], ["results/review_checks.json"], [],
     {".local_trace_indexes.dirs_skipped_unreadable": "counts local folders the trace scan could not read; it depends on the checkout (it rose after the 2026-10-19 reorganisation added long-path clones)"}),
    ("readiness-labels", "readiness", [["score_labels.py"]], ["results/labelling.json"], [],
     {".coder_model": "coder provenance added by hand in c058d2e from the subagent transcripts; not computed by score_labels.py"}),
    ("readiness", "readiness", [["analyze_readiness.py"]], ["results/readiness.json"], ["results/readiness_manifest.json"], {}),
    ("p8", "readiness", [["analyze_p8.py", "run"]], ["results/p8_coding.json", "results/p8.json"], ["results/p8_manifest.json"], {}),
    ("p8-posthoc", "readiness", [["posthoc_p8.py"]], ["results/p8_posthoc.json"], [], {}),
    ("p9", "readiness", [["analyze_p9.py", "run"]], ["results/p9_coding.json", "results/p9.json"], ["results/p9_manifest.json"], {}),
    ("p9-posthoc", "readiness", [["posthoc_p9.py"]], ["results/p9_posthoc.json"], [], {}),
    ("p9-areas", "readiness", [["posthoc_p9_areas.py"]], ["results/p9_areas.json"], [], {}),
    ("p9-uptake", "readiness", [["posthoc_p9_uptake.py"]], ["results/p9_uptake.json"], [], {}),
    ("p9-adversarial", "readiness", [["posthoc_p9_adversarial.py"]], ["results/p9_adversarial.json"], [], {}),
]


def diff_paths(a, b, path="", out=None, limit=12):
    """Key paths where two parsed JSON values differ (at most `limit`)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k} (only in {'new' if k in a else 'committed'})")
            else:
                diff_paths(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path or '.'}: committed {str(b)[:60]} -> new {str(a)[:60]}")
    return out


def environment() -> dict:
    pkgs = {}
    for p in ("numpy", "pandas", "scipy", "statsmodels", "pypdf", "huggingface_hub", "httpx", "requests"):
        try:
            pkgs[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pkgs[p] = None
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return dict(python=platform.python_version(), platform=platform.platform(), packages=pkgs,
                git_commit=git("rev-parse", "HEAD"), git_dirty_verdict_audit=bool(git("status", "--porcelain", "--", "verdict_audit")))


def committed_copy(path: Path, saved: dict) -> Path | None:
    """Where the committed version of `path` was set aside (directly, or inside a set-aside folder)."""
    for orig, moved in saved.items():
        if path == orig:
            return moved
        if path.is_relative_to(orig):
            return moved / path.relative_to(orig)
    return None


def run_step(name, folder, commands, compared, rewritten, ignore, run_dir: Path) -> dict:
    cwd = (HERE / folder).resolve()
    # A trailing "/" names a whole output folder: some scripts create it with exist_ok=False and run only once.
    dirs = [cwd / p.rstrip("/") for p in rewritten if p.endswith("/")]
    touched = dirs + [cwd / p for p in compared + rewritten if not p.endswith("/") and not any((cwd / p).is_relative_to(d) for d in dirs)]
    saved = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, p in enumerate(touched):  # set committed files aside (some scripts refuse to overwrite)
            if p.exists():
                saved[p] = Path(tmp) / f"{i}_{p.name}"
                shutil.move(str(p), str(saved[p]))
        rec = dict(step=name, folder=folder, commands=[" ".join(c) for c in commands])
        started = time.monotonic()
        try:
            for c in commands:
                proc = subprocess.run([PY, *c], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if proc.returncode:
                    rec.update(status="error", returncode=proc.returncode, stderr_tail=proc.stderr[-1500:])
                    break
            else:
                diffs, missing = {}, []
                for rel in compared:
                    new, old = cwd / rel, committed_copy(cwd / rel, saved)
                    if not new.exists():
                        missing.append(rel)
                        continue
                    if old is None:
                        diffs[rel] = ["no committed file to compare with"]
                        continue
                    d = diff_paths(json.loads(new.read_text(encoding="utf-8")), json.loads(old.read_text(encoding="utf-8")))
                    skipped = [x for x in d if any(x.startswith(k) for k in ignore)]
                    if skipped:
                        rec.setdefault("ignored", {})[rel] = dict(differences=skipped, reasons=ignore)
                    d = [x for x in d if x not in skipped]
                    if d:
                        diffs[rel] = d
                        keep = run_dir / name / rel
                        keep.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(new, keep)
                rec.update(status="match" if not diffs and not missing else "differs", differences=diffs, missing=missing,
                           new_sha256={rel: hashlib.sha256((cwd / rel).read_bytes()).hexdigest()
                                       for rel in compared if (cwd / rel).exists()})
        finally:
            for p in touched:  # always put the committed state back
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
                if p in saved:
                    shutil.move(str(saved[p]), str(p))
        rec["seconds"] = round(time.monotonic() - started, 1)
    return rec


def main() -> None:
    args = sys.argv[1:]
    if "--list" in args:
        for s in STEPS:
            print(f"{s[0]:24} {s[1]:16} {' ; '.join(' '.join(c) for c in s[2])}")
        return
    chosen = [s for s in STEPS if not args or s[0] in args]
    unknown = set(args) - {s[0] for s in STEPS}
    if unknown:
        raise SystemExit(f"unknown steps: {sorted(unknown)}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = HERE / "logs"
    run_dir = logs / stamp
    report = dict(started=stamp, environment=environment(), steps=[])
    for s in chosen:
        print(f"[{s[0]}] running ...", flush=True)
        rec = run_step(*s, run_dir)
        report["steps"].append(rec)
        print(f"[{s[0]}] {rec['status']} in {rec['seconds']} s", flush=True)
    report["summary"] = {k: sum(1 for r in report["steps"] if r["status"] == k) for k in ("match", "differs", "error")}
    logs.mkdir(exist_ok=True)
    (logs / f"reproduce_{stamp}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("summary:", report["summary"], "| log:", f"logs/reproduce_{stamp}.json")


if __name__ == "__main__":
    main()
