"""Protocol 9 release table: one row per frame paper, identifiers and codes only, safe to publish.

The conference listing files and full texts behind protocol 9 may not be redistributed (registry entries
ml-venue-listings-2024-2026-v1 and ml-venue-fulltext-2025-2026-v1). This table carries no abstract, no full text and
no listing field beyond the paper's identifier and title, so a reader can re-identify every paper and recompute every
protocol 9 prevalence from it.

Columns: item (blinded coding id), venue, year, forum_id, openreview_url (empty for non-OpenReview ids, which start
with "src:"), title, group (life_science | other), topic_label (protocol 7 label, life-science rows only),
fulltext_status and fulltext_source and arxiv_id (from the full-text manifest; uncoded papers have no codes),
then the twelve protocol 9 fields (final codes: coder 1, with adjudication for variance and compute) and the three
reference fields (challenge papers only).

Usage: python build_p9_release.py   -> results/p9_papers.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from analyze_p9 import REF_FIELDS, final_codes, reference_codes
from build_p9_sheets import FIELDS, KEYS
from fetch_p9 import MANIFEST

HERE = Path(__file__).resolve().parent
FRAME = HERE / "results" / "p9_frame.json"
OUT = HERE / "results" / "p9_papers.csv"
COLUMNS = ("item", "venue", "year", "forum_id", "openreview_url", "title", "group", "topic_label",
           "fulltext_status", "fulltext_source", "arxiv_id", *FIELDS, *REF_FIELDS)


def fulltext() -> dict:
    """Last manifest row per paper key (later rows are retries)."""
    out = {}
    with MANIFEST.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["key"]] = r
    return out


def rows() -> list[dict]:
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    labels = {(u["stratum"], u["forum_id"]): u.get("label") for u in json.loads(FRAME.read_text(encoding="utf-8"))}
    final, ref, ft = final_codes(), reference_codes(), fulltext()
    out = []
    for item in sorted(keys):
        k = keys[item]
        venue, year = k["stratum"].rsplit("-", 1)
        m = ft.get(k["key"], {})
        fid = k["forum_id"]
        row = dict(item=item, venue=venue, year=year, forum_id=fid,
                   openreview_url="" if fid.startswith("src:") else f"https://openreview.net/forum?id={fid}",
                   title=k["title"], group=k["group"], topic_label=labels.get((k["stratum"], fid)) or "",
                   fulltext_status=m.get("status", "not_fetched"), fulltext_source=m.get("source") or "",
                   arxiv_id=m.get("arxiv_id") or "")
        row.update({f: final.get(item, {}).get(f, "") for f in FIELDS})
        row.update({f: ref.get(item, {}).get(f, "") for f in REF_FIELDS})
        out.append(row)
    return out


def main() -> None:
    data = rows()
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(data)
    coded = sum(bool(r["code"]) for r in data)
    print(f"{OUT.name}: {len(data)} papers, {coded} coded", file=sys.stderr)


if __name__ == "__main__":
    main()
