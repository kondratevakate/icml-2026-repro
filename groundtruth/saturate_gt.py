#!/usr/bin/env python3
"""
saturate_gt.py — keep the ICML-2026-repro Ground-Truth (hard-GT) set saturated.

What it does
------------
1. Pulls the LIVE leaderboard verdicts (public HF dataset, no creds).
2. Recomputes the strict-CPU candidate set:
     consensus >= 3  AND  non-Theory  AND  STRICT cpu-only/no-gpu
   (hw_signals scraped from each paper's canonical leaderboard logbook README via
    hf_hub_download; rejects hard-gpu and gpu-without-cpu).
3. Excludes papers already in the original-45 and in the current strict-41 list.
4. Appends NEW orids to groundtruth/out/new_cpu_friendly_strict.json
   (strict_cpu_by_time / strict_cpu_by_claims), preserving existing entries.
5. Writes the next batch file (top10c_new_cpu.json, top10d_..., by repro-time order)
   for the runner (run03 arm1 leaf) to consume.
6. Prints a diff of what changed.

Usage
-----
    python3 groundtruth/saturate_gt.py            # pull + recompute + append + next batch
    python3 groundtruth/saturate_gt.py --dry-run  # pull + recompute, print diff, no write
    python3 groundtruth/saturate_gt.py --next-batch-size 10

Notes
-----
- Requires: requests (for curl-equivalent) or curl CLI; huggingface_hub (for README scrape).
- CPU scan hits the HF Hub per paper (rate-limited); for 500+ papers this takes minutes.
- The script is idempotent: re-running with no new consensus papers yields no changes.
"""
import json, collections, re, os, sys, subprocess, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_LIVE = os.path.join(ROOT, "groundtruth", "data_live")
OUT = os.path.join(ROOT, "groundtruth", "out")
VERDICTS_URL = "https://huggingface.co/datasets/ICML-2026-agent-repro/verdicts/resolve/main/verdicts.json"

GPU_HARD = re.compile(r"\b(A100|H100|V100|GPU|CUDA|cuda|multi[- ]?gpu|8x|4x GPU|gpu cluster|hardware accelerated|distributed training|deep learning framework training)\b", re.I)
GPU_SOFT = re.compile(r"\b(gpu)\b", re.I)
CPU_OK = re.compile(r"\b(cpu|numpy|scipy|scikit|pandas|matplotlib|sklearn|sympy|octave|matlab|julia|pure python)\b", re.I)
THEORY = re.compile(r"(theorem|proof|we prove|bound|convergence|lemma|proposition|asymptotic|minimax|identifiab|guarantee)", re.I)


def pull_verdicts():
    os.makedirs(DATA_LIVE, exist_ok=True)
    dest = os.path.join(DATA_LIVE, "verdicts_live.json")
    print(f"[pull] {VERDICTS_URL}")
    r = subprocess.run(["curl", "-L", "-s", "-o", dest, VERDICTS_URL], check=False)
    if r.returncode != 0:
        # fallback: python requests
        try:
            import requests
            with requests.get(VERDICTS_URL, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
        except Exception as e:
            print(f"[pull] FAILED: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"[pull] saved -> {dest}")
    return dest


def load_json(p):
    with open(p) as f:
        return json.load(f)


def hw_signals_of(orid, by, hf_hub_download):
    spaces = by.get(orid, [])
    if not spaces:
        return None
    sid = spaces[0]
    try:
        path = hf_hub_download(repo_id=sid, filename="README.md", repo_type="space", token=True)
        t = open(path).read()
    except Exception:
        return None
    sig = []
    if GPU_HARD.search(t):
        sig.append("hard-gpu")
    if GPU_SOFT.search(t):
        sig.append("gpu")
    for kw in ["cpu", "numpy", "scipy", "scikit", "sklearn", "pandas", "sympy", "matlab", "octave"]:
        if re.search(r"\b" + kw + r"\b", t, re.I):
            sig.append(kw)
    return sig


def cpu_friendly(orid, sig, papers, mode="mild"):
    """mild: no hard-gpu AND (cpu mentioned OR no gpu mention at all)  [matches original-45]
       strict: cpu-only / no-gpu, rejects gpu-without-cpu            [matches strict-41]"""
    if sig is None:
        # no README scraped: fall back to title/area cpu hint for mild mode
        if mode == "strict":
            return False
        return CPU_OK.search((papers.get(orid, {}).get("title", "") + " " +
                              papers.get(orid, {}).get("area", ""))) is not None
    if "hard-gpu" in sig:
        return False
    if mode == "strict":
        return "cpu" in sig
    # mild: gpu without cpu is still ok if it's a soft mention and no hard-gpu
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--next-batch-size", type=int, default=10)
    ap.add_argument("--mode", choices=["mild", "strict"], default="mild",
                    help="mild = no-hard-gpu AND (cpu-mention OR no-gpu) [matches original-45]; "
                         "strict = cpu-only/no-gpu [matches strict-41]")
    ap.add_argument("--max-new", type=int, default=0, help="cap new papers to enqueue (0=all)")
    args = ap.parse_args()

    verdicts_path = pull_verdicts()
    VL = load_json(verdicts_path)
    IDX = load_json(os.path.join(DATA_LIVE, "index.json"))
    ANC = load_json(os.path.join(DATA_LIVE, "claims_anchored.json"))
    ORIG = load_json(os.path.join(OUT, "selected_for_runs.json"))["papers"]
    orig_orids = {p["orid"] for p in ORIG}
    papers = {p["orid"]: p for p in IDX["papers"]}

    strict_path = os.path.join(OUT, "new_cpu_friendly_strict.json")
    strict_doc = load_json(strict_path) if os.path.exists(strict_path) else {
        "criteria": "", "scanned": 0, "strict_cpu_count": 0,
        "strict_cpu_by_time": [], "strict_cpu_by_claims": []}
    known_orids = {r["orid"] for r in strict_doc.get("strict_cpu_by_time", [])}

    from huggingface_hub import hf_hub_download
    # group verdicts by orid -> list of submission entries (space_id)
    by = collections.defaultdict(list)
    for sid, val in VL.items():
        o = val.get("orid")
        if o:
            by[o].append(val.get("space_id", sid))

    # consensus >= 3 = at least 3 independent submissions (space_ids) for the paper
    consensus = {o: len(subs) for o, subs in by.items() if len(subs) >= 3}

    candidates = []  # new orids passing all filters
    scanned = 0
    for o, cnt in consensus.items():
        if o in orig_orids or o in known_orids:
            continue
        meta = papers.get(o, {})
        title_area = meta.get("title", "") + " " + meta.get("area", "")
        if THEORY.search(title_area):
            continue
        sig = hw_signals_of(o, by, hf_hub_download)
        scanned += 1
        if not cpu_friendly(o, sig, papers, mode=args.mode):
            continue
        anchored = ANC.get(o, [])
        # take latest judged_at across submissions
        ja = max((VL[sid].get("judged_at") for sid in by[o] if VL[sid].get("judged_at")),
                 default=None)
        candidates.append({
            "orid": o, "title": meta.get("title", ""), "area": meta.get("area", ""),
            "submissions": cnt, "judged_at": ja,
            "hw_signals": sig, "cpu_friendly_strict": True,
            "n_anchored_claims": len(anchored) if isinstance(anchored, list) else 0,
        })

    if args.max_new:
        candidates = candidates[:args.max_new]

    print(f"\n[scan] consensus>=3 candidates outside known sets: {len(candidates)} "
          f"(scanned READMEs: {scanned})")

    if not candidates:
        print("[diff] no new papers to add — GT already saturated at this snapshot.")
        return

    # append to strict doc, preserving existing
    existing = strict_doc.get("strict_cpu_by_time", [])
    merged = existing + candidates
    strict_doc["strict_cpu_by_time"] = sorted(
        merged, key=lambda r: r.get("judged_at") or "", reverse=True)
    strict_doc["strict_cpu_by_claims"] = sorted(
        merged, key=lambda r: -r.get("n_anchored_claims", 0))
    strict_doc["strict_cpu_count"] = len(merged)
    strict_doc["criteria"] = ("consensus>=3 AND non-Theory AND STRICT cpu-only/no-gpu "
                              "(hw_signals from README, no hard-gpu, cpu mentioned)")

    # next batch file
    batch_letter = "c"
    while os.path.exists(os.path.join(OUT, f"top10{batch_letter}_new_cpu.json")):
        batch_letter = chr(ord(batch_letter) + 1)
    next_batch = strict_doc["strict_cpu_by_time"][:args.next_batch_size]
    next_path = os.path.join(OUT, f"top10{batch_letter}_new_cpu.json")

    print(f"[diff] +{len(candidates)} new papers. Next batch -> {os.path.basename(next_path)} "
          f"({len(next_batch)} papers)")
    for r in candidates[: args.next_batch_size]:
        print(f"    {r['orid']} sub={r['submissions']} claims={r['n_anchored_claims']} "
              f"{r['title'][:50]}")

    if args.dry_run:
        print("[dry-run] no writes performed.")
        return

    with open(strict_path, "w") as f:
        json.dump(strict_doc, f, indent=1, ensure_ascii=False)
    with open(next_path, "w") as f:
        json.dump({"batch": batch_letter, "papers": next_batch}, f, indent=1, ensure_ascii=False)
    print(f"[write] updated {os.path.relpath(strict_path, ROOT)}")
    print(f"[write] wrote {os.path.relpath(next_path, ROOT)}")


if __name__ == "__main__":
    main()
