#!/usr/bin/env python3
"""
adapt_claims_to_arc.py — convert ICML-repro input_bundle.json → ARC-compatible idea.

ARC's `arc_bench/scripts/make_manifests.py` expects an `idea.json` with:
- "Name", "Title", "Abstract" (optional), "Experiment" (detailed prompt)

Usage:
  python3 adapt_claims_to_arc.py <orid> --out ideas/<orid>.json
"""
import json, os, sys, argparse, re, pathlib

HERE = pathlib.Path(__file__).parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orid")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bundle = json.load(open(os.path.join(HERE, args.orid, "input_bundle.json")))
    claims = bundle.get("anchored_claims", [])
    title = bundle.get("title", args.orid)
    arxiv = bundle.get("arxiv", "")

    # Build experiment prompt
    lines = [
        f"Reproducibility task: verify the anchored claims of '{title}' (ICML 2026, arXiv {arxiv}) on CPU.",
        "",
        "For each claim below, write a self-contained verify_claim<N>.py using only numpy/scipy/sympy (CPU),",
        "run it, save outputs to results/claim<N>.json, and report a verdict (verified/falsified/toy/inconclusive).",
        "Honest boundary: if data/GPU required, report inconclusive — do NOT fabricate toy data.",
        "",
    ]
    for i, c in enumerate(claims, 1):
        if isinstance(c, dict):
            stmt = c.get("statement", c.get("claim", ""))
            loc = c.get("source_location", c.get("location", ""))
        else:
            stmt = str(c)
            m = re.search(r"(theorem\s+\d+[\.\d]*|section\s+\d+[\.\d]*|eq\.?\s*\d+[\.\d]*)", stmt, re.I)
            loc = m.group(1) if m else ""
        lines.append(f"Claim {i} ({loc}): {stmt}")

    experiment = "\n".join(lines)

    idea = {
        "Name": f"icml-repro-{args.orid}-arc",
        "Title": title,
        "Abstract": f"Reproduction of anchored claims from '{title}' (ICML 2026).",
        "Experiment": experiment,
        "_repro_orid": args.orid,
        "_repro_model": "tencent/hy3:free",
        "_repro_type": "reproduction",
    }

    out = args.out or os.path.join(HERE, "arcs", f"{args.orid}.json")
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    json.dump(idea, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[ARC adapt] wrote {out}")

if __name__ == "__main__":
    main()