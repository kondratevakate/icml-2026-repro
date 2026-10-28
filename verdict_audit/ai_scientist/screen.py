"""Screen all paper sets for AI-scientist / autonomous-research papers with the frozen rule in PROTOCOL.md.

Reads the git-ignored inputs from the main checkout (MAIN_CHECKOUT, default the icml-repro-working folder).

Usage: python screen.py   -> results/papers.csv, results/summary.json
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = Path(os.environ.get("MAIN_CHECKOUT", r"C:\Projects\02_academia\studies\icml\icml-repro-working"))
sys.path.insert(0, str(MAIN / "verdict_audit" / "readiness"))
import p8_common as P8  # noqa: E402

PATTERNS = {
    "ai_scientist": r"\bai[- ]scientists?\b|\bai co-?scientists?\b|\bautomated scientists?\b|\bautonomous scientists?\b|"
                    r"\bscientist agents?\b",
    "research_agent": r"\bresearch agents?\b",
    "automated_research": r"\bautomated research\b|\bautonomous research\b|\bauto-?research\b|\bresearch automation\b|"
                          r"\bautomat(?:e|ing|ion of) (?:scientific )?research\b|\bagentic research\b|"
                          r"\bend-to-end research\b",
    "automated_discovery": r"\bautomated scientific discovery\b|\bautonomous scientific discovery\b|"
                           r"\bai-driven scientific discovery\b",
    "automated_experimentation": r"\bself-driving lab|\bautonomous lab|\bautonomous experimentation\b",
    "paper_reproduction": r"\breproduc(?:e|ing|tion) (?:of )?(?:ml |research |scientific )?papers\b|\bpaper reproduction\b|"
                          r"\breplicat(?:e|ing|ion) (?:of )?(?:ml |research )?papers\b",
    "named_benchmark": r"\bpaperbench\b|\bmle-bench\b|\bre-bench\b|\bscienceagentbench\b|\bcore-bench\b|"
                       r"\bmlagentbench\b|\bresearchbench\b|\bdiscoverybench\b",
}
RE = {k: re.compile(v, re.IGNORECASE) for k, v in PATTERNS.items()}


def main() -> dict:
    units = json.loads((P8.DATA / "papers.json").read_text(encoding="utf-8"))
    cv = P8.claim_verdicts()
    rows = []
    for u in units:
        text = f"{u.get('title') or ''} {u.get('abstract') or ''}"
        hits = [k for k, r in RE.items() if r.search(text)]
        if not hits:
            continue
        row = dict(stratum=u["stratum"], paper_id=u["forum_id"], title=" ".join((u.get("title") or "").split()),
                   patterns=";".join(hits), attempted="", decided="")
        if u["stratum"] == "icml-2026":
            cl = cv.get(u["forum_id"])
            row["attempted"] = "yes" if cl is not None else "no"
            row["decided"] = ("yes" if any(c["verdict"] in P8.DECISIVE for c in cl) else "no") if cl is not None else ""
        rows.append(row)
    rows.sort(key=lambda r: (r["stratum"], r["paper_id"]))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with (out / "papers.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["stratum", "paper_id", "title", "patterns", "attempted", "decided"])
        w.writeheader()
        w.writerows(rows)
    total = Counter(u["stratum"] for u in units)
    found = Counter(r["stratum"] for r in rows)
    icml = [u for u in units if u["stratum"] == "icml-2026"]
    att_all = sum(u["forum_id"] in cv for u in icml)
    dec_all = sum(any(c["verdict"] in P8.DECISIVE for c in cv[u["forum_id"]]) for u in icml if u["forum_id"] in cv)
    sel = [r for r in rows if r["stratum"] == "icml-2026"]
    summary = dict(
        papers_screened=len(units), included=len(rows),
        by_venue_year={s: dict(papers=total[s], included=found[s], share=found[s] / total[s]) for s in sorted(total)},
        by_pattern=dict(Counter(p for r in rows for p in r["patterns"].split(";"))),
        icml_2026=dict(included=len(sel), attempted=sum(r["attempted"] == "yes" for r in sel),
                       decided=sum(r["decided"] == "yes" for r in sel),
                       all_papers=len(icml), all_attempted=att_all, all_decided=dec_all),
        verdict_snapshot_sha256=P8.VERDICTS_SHA)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
