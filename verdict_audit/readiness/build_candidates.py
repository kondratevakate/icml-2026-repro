"""Build the Protocol 7 paper table, candidate list, coder samples and blind coding sheets.

Writes data/papers.json (all units), data/sheets/*.csv (title + abstract only) and data/keys.json;
results/candidates_summary.json holds counts only.
"""
from __future__ import annotations

import ast
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
VENUES = [("iclr", 2025), ("iclr", 2026), ("neurips", 2024), ("neurips", 2025), ("neurips", 2026), ("icml", 2025),
          ("icml", 2026), ("iccv", 2025), ("cvpr", 2024), ("cvpr", 2025), ("eccv", 2024), ("eccv", 2026)]
PAGE_ABSTRACTS = ("iclr-2026", "neurips-2026", "eccv-2026")  # Amendments 1-2
TOPICS = {
    "Applications->Health", "Applications->Health / Medicine", "Applications->Genetics, Cell Biology, Health, etc",
    "Applications->Genetics, Cell Biology, etc", "Applications->Neuroscience, Cognitive Science",
    "Neuroscience, Cognitive Science", "Computational Biology", "Applications->Chemistry and Drug Discovery",
}
MEDICAL = re.compile(
    r"\b(patients?|clinical\w*|clinicians?|hospitals?|EHRs?|electronic health records?|medical|medicine|diseases?|"
    r"diagnos\w*|prognos\w*|radiolog\w*|patholog\w*|histopatholog\w*|MRI|fMRI|CT scans?|X-rays?|ultrasound|ECG|EEG|"
    r"surg\w*|cancers?|tumou?rs?|oncolog\w*|cardi\w*|alzheimer\w*|dementia|ICU|MIMIC\w*|sepsis|drugs?|pharmac\w*|"
    r"therap\w*|health ?care|public health|epidemi\w*|biomedic\w*|proteins?|genes?|genom\w*|transcriptom\w*|"
    r"single-cell|scRNA\w*|DNA|RNA|molecul\w*|enzymes?|antibod\w*|peptides?|biolog\w*|neurons?|neuroscien\w*|brain)\b",
    re.IGNORECASE)
SEED = 20261018
PAPERS_JSON = HERE.parents[1] / "papers.json"


def forum_id(item: dict) -> str:
    m = re.search(r"[?&]id=([^&]+)", str(item.get("paper_url") or ""))
    return m.group(1) if m else f"src:{item.get('sourceid') or item.get('id')}"


def eventmedia(item: dict) -> list[dict]:
    em = item.get("eventmedia") or []
    if isinstance(em, str):
        try:
            em = ast.literal_eval(em)
        except (ValueError, SyntaxError):
            return []
    return [m for m in em if isinstance(m, dict)]


def extra_abstracts() -> dict[str, dict[str, str]]:
    """Amendment 1: abstracts for the 2026 strata, keyed by forum id (ICML) or virtual-site path (ICLR)."""
    papers = json.loads(PAPERS_JSON.read_text(encoding="utf-8-sig"))
    out = {"icml-2026": {k: " ".join(str(v).split()) for k, v in papers["abstracts"].items() if v}}
    for stratum in PAGE_ABSTRACTS:
        pages = json.loads((DATA / f"{stratum}_abstracts.json").read_text(encoding="utf-8"))
        out[stratum] = {k: v["abstract"] for k, v in pages.items() if v.get("status") == "ok" and v.get("abstract")}
    return out


def load_units() -> tuple[list[dict], Counter]:
    units, counts = [], Counter()
    extra = extra_abstracts()
    for venue, year in VENUES:
        raw = json.loads((DATA / f"{venue}-{year}.json").read_text(encoding="utf-8"))
        items = raw["results"] if isinstance(raw, dict) else raw
        seen = set()
        for it in items:
            if str(it.get("eventtype", "")).lower() not in ("poster", "oral", "spotlight"):
                counts["not_poster_or_oral"] += 1
                continue
            fid = forum_id(it)
            stratum = f"{venue}-{year}"
            if (stratum, fid) in seen:
                counts["duplicate"] += 1
                continue
            seen.add((stratum, fid))
            abstract = " ".join(str(it.get("abstract") or "").split())
            if stratum == "icml-2026" and (not abstract or abstract == "None"):
                abstract = extra["icml-2026"].get(fid, "")
            if stratum in PAGE_ABSTRACTS and (not abstract or abstract == "None"):
                abstract = extra[stratum].get(str(it.get("virtualsite_url")), "")
            if not abstract or abstract == "None":
                counts["no_abstract"] += 1
                continue
            title = " ".join(str(it.get("name") or "").split())
            units.append(dict(stratum=stratum, forum_id=fid, title=title, abstract=abstract, topic=it.get("topic") or "",
                              url=str(it.get("url") or ""),
                              media_urls=[str(m.get("uri")) for m in eventmedia(it) if m.get("type") == "URL" and m.get("uri")],
                              by_topic=(it.get("topic") or "") in TOPICS,
                              by_regex=bool(MEDICAL.search(title + " " + abstract))))
    for u in units:
        u["candidate"] = u["by_topic"] or u["by_regex"]
    return units, counts


def write_sheet(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["item_id", "title", "abstract"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    units, counts = load_units()
    rng = random.Random(SEED)
    cand = [u for u in units if u["candidate"]]
    non = [u for u in units if not u["candidate"]]
    overlap = set(rng.sample(range(len(cand)), round(0.25 * len(cand))))
    non_sample = rng.sample(range(len(non)), 200)
    reclaim = json.loads((DATA / "reclaim_arxiv_abstracts.json").read_text(encoding="utf-8"))

    items = []  # (key, title, abstract, in coder1, in coder2)
    for i, u in enumerate(cand):
        items.append((f"cand:{u['stratum']}:{u['forum_id']}", u["title"], u["abstract"], True, i in overlap))
    for j in non_sample:
        u = non[j]
        items.append((f"non:{u['stratum']}:{u['forum_id']}", u["title"], u["abstract"], True, True))
    for aid, v in sorted(reclaim.items()):
        items.append((f"reclaim:{aid}", v["title"], v["abstract"], True, True))
    rng.shuffle(items)
    keys = {}
    sheet1, sheet2 = [], []
    for n, (key, title, abstract, c1, c2) in enumerate(items):
        item_id = f"P{n:05d}"
        keys[item_id] = key
        row = dict(item_id=item_id, title=title, abstract=abstract)
        if c1:
            sheet1.append(row)
        if c2:
            sheet2.append(row)
    sheets = DATA / "sheets"
    sheets.mkdir(exist_ok=True)
    write_sheet(sheets / "coder1_all.csv", sheet1)
    write_sheet(sheets / "coder2_all.csv", sheet2)
    (DATA / "keys.json").write_text(json.dumps(keys), encoding="utf-8")
    (DATA / "papers.json").write_text(json.dumps(units), encoding="utf-8")
    import hashlib
    summary = dict(papers_json_sha256=hashlib.sha256(PAPERS_JSON.read_bytes()).hexdigest(),
                   page_abstracts_sha256={s: hashlib.sha256((DATA / f"{s}_abstracts.json").read_bytes()).hexdigest()
                                          for s in PAGE_ABSTRACTS},
                   units=len(units), excluded=dict(counts), by_stratum=dict(Counter(u["stratum"] for u in units)),
                   candidates=len(cand), candidates_by_stratum=dict(Counter(u["stratum"] for u in cand)),
                   candidates_by_topic_only=sum(u["by_topic"] and not u["by_regex"] for u in cand),
                   candidates_by_regex_only=sum(u["by_regex"] and not u["by_topic"] for u in cand),
                   overlap_candidates=len(overlap), non_candidate_sample=len(non_sample), reclaim=len(reclaim),
                   coder1_items=len(sheet1), coder2_items=len(sheet2))
    (HERE / "results" / "candidates_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
