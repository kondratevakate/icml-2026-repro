#!/usr/bin/env python3
"""
adapt_claims_to_sakana.py — convert an ICML-repro input_bundle.json (anchored claims)
into a Sakana-compatible `idea.json` so arm 3 (AI-Scientist-v2) can attempt the
reproduction task using its own agent loop.

Sakana's `launch_scientist_bfts.py` reads `ideas/<name>.json` with fields like
`Name`, `Title`, `Title (ar5iv)`, `Abstract`, `Experiment` (a detailed natural-language
description of what experiments to run). We map each anchored claim to the Experiment
prompt + embed the paper context.

Usage:
  python3 adapt_claims_to_sakana.py <orid> [--out ideas/<orid>.json]
"""
import json, os, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orid")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bundle = json.load(open(os.path.join(HERE, args.orid, "input_bundle.json")))
    claims = bundle.get("anchored_claims", [])
    title = bundle.get("title", args.orid)
    arxiv = bundle.get("arxiv", "")
    openreview = bundle.get("openreview", "")

    # Build a single Sakana "Experiment" prompt that drives reproduction of all claims.
    exp_lines = []
    for i, c in enumerate(claims, start=1):
        cid = i
        if isinstance(c, dict):
            stmt = c.get("statement", c.get("claim", ""))
            loc = c.get("source_location", c.get("location", ""))
        else:  # claim is a plain string (most bundles)
            stmt = str(c)
            # try to pull a theorem/section reference from the text
            import re as _re
            m = _re.search(r"(theorem\s+\d+[\.\d]*|lemma\s+\d+[\.\d]*|section\s+\d+[\.\d]*|eq\.?\s*\d+[\.\d]*)", stmt, _re.I)
            loc = m.group(1) if m else ""
        exp_lines.append(
            f"- Claim {cid} (source: {loc}): reproduce and verify the statement:\n"
            f'  "{stmt}"\n'
            f"  Write a self-contained verify_claim{cid}.py (numpy/scipy/sympy, CPU-only) that "
            f"produces real numbers; include a mutation test for any verified claim."
        )
    experiment = (
        f"Reproduce the anchored claims of the ICML 2026 paper '{title}' "
        f"(arXiv {arxiv}, OpenReview {openreview}) on CPU.\n"
        "For EACH claim below, write a verify_claim<N>.py that actually runs and produces numbers, "
        "save raw output to results/claim<N>.json, and state a verdict "
        "(verified / falsified / toy / inconclusive) with the exact source location. "
        "If a claim needs data/GPU you lack, report inconclusive with the reason — do NOT fabricate a toy substitute.\n"
        "At the end write logbook.md (per-claim verdict + Evidence boundary) and check_reproducibility.py.\n\n"
        + "\n".join(exp_lines)
    )

    idea = {
        "Name": f"icml-repro-{args.orid}",
        "Title": title,
        "Title (ar5iv)": f"https://ar5iv.org/abs/{arxiv}" if arxiv else "",
        "Abstract": f"Reproduction of anchored claims from '{title}' (ICML 2026).",
        "Experiment": experiment,
        # metadata for our harness (not used by Sakana, but harmless)
        "_repro_orid": args.orid,
        "_repro_model": "tencent/hy3:free",
    }

    out = args.out or os.path.join(HERE, "ideas", f"{args.orid}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(idea, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[adapt] wrote Sakana idea -> {out}")
    print(f"        title: {title}")
    print(f"        claims: {len(claims)}")

if __name__ == "__main__":
    main()
