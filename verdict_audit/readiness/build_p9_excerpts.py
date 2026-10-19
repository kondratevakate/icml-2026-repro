"""Protocol 9 excerpts: up to 15,000 characters per paper from its full text, in the frozen order:
(1) the last 3,000 characters of the Introduction; (2) sections whose heading matches the data, code,
reproducibility, compute and limitation pattern; (3) NeurIPS checklist items 4-8 and 13 when present; (4) other
paragraphs matching the protocol 8 regex. A truncation flag records whether the cap cut anything.

Reads the raw full texts under DATA_ROOT (fetch_p9.py) and writes the excerpts there too, under derived/, never
into git. Writes aggregates only to results/p9_excerpts_summary.json. Resumable per run (rebuilds all).

Usage: python build_p9_excerpts.py [--limit N]
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF, for PDF text blocks
from bs4 import BeautifulSoup

from fetch_p8_fulltext import PATTERN as P8_PATTERN
from fetch_p9 import DS, MANIFEST

HERE = Path(__file__).resolve().parent
OUT = DS / "derived" / "excerpts-v1"
CAP, INTRO_TAIL, SECTION_CAP = 15000, 3000, 2000
# Note 9-1: order of matched sections (release and access statements first, then compute and scale, then
# setup, then data descriptions, then limitations); each contributes at most SECTION_CAP characters.
PRIORITY = ("availab", "reproducib", "ethic", "compute", "resources", "implementation", "training details",
            "experimental setup", "data", "limitation")
SECTION_RE = re.compile(r"availab|reproducib|ethic|data|experimental setup|implementation|training details|compute|"
                        r"resources|limitation", re.IGNORECASE)
INTRO_RE = re.compile(r"\bintroduction\b", re.IGNORECASE)
SKIP_RE = re.compile(r"\breferences\b|\bbibliography\b|instructions for reporting errors", re.IGNORECASE)
CHECKLIST_RE = re.compile(r"checklist", re.IGNORECASE)
CHECKLIST_ITEMS = {4, 5, 6, 7, 8, 13}
PDF_HEADING_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[A-H](?:\.\d+)*)\.?\s+[A-Z][^\n]{1,90}|abstract|introduction|acknowledge?ments?|references|"
    r"(?:data|code)(?: and code| and data)? availability[^\n]{0,40}|reproducibility statement|ethics statement|"
    r"limitations?|broader impacts?[^\n]{0,30}|impact statement|neurips paper checklist)$", re.IGNORECASE)


def blocks_html(raw: bytes) -> list[tuple[str, str, str]]:
    """(section heading, nearest heading, paragraph) in document order, from arXiv LaTeXML HTML."""
    soup = BeautifulSoup(raw.decode("utf-8", "replace"), "lxml")
    for m in soup.find_all("math"):
        m.replace_with(" " + (m.get("alttext") or "") + " ")
    for t in soup.find_all(["script", "style", "nav", "header", "footer"]):
        t.decompose()
    out, section, nearest = [], "abstract", "abstract"
    for el in soup.find_all(["h2", "h3", "h4", "h5", "h6", "p"]):
        text = " ".join(el.get_text(" ").split())
        if el.name == "p":
            if len(text) > 40 or text.startswith("Answer:"):  # checklist answers are short paragraphs
                out.append((section, nearest, text))
        elif "ltx_title" in (el.get("class") or []):
            if el.name == "h2":
                section = text
            nearest = text
    return out


def blocks_pdf(raw: bytes) -> list[tuple[str, str, str]]:
    """The same from a PDF: text blocks, with short blocks that look like headings taken as headings."""
    out, section, nearest = [], "abstract", "abstract"
    with fitz.open(stream=raw, filetype="pdf") as doc:
        for page in doc:
            for b in page.get_text("blocks"):
                text = " ".join(b[4].split())
                if not text:
                    continue
                if len(text) <= 100 and PDF_HEADING_RE.match(text):
                    nearest = text
                    if not re.match(r"^(?:\d+|[A-H])\.\d", text):  # top level: "3 Method", "A Proofs", unnumbered
                        section = text
                elif len(text) > 40:
                    out.append((section, nearest, text))
    return out


def checklist_items(paras: list[str]) -> list[str]:
    """Question, answer and justification of NeurIPS checklist items 4-8 and 13; items are counted by their
    'Question:' paragraphs, and the checklist's own guideline text is left out."""
    keep, item = [], 0
    for p in paras:
        if p.startswith("Question:"):
            item += 1
        if item in CHECKLIST_ITEMS and p.startswith(("Question:", "Answer:", "Justification:")):
            keep.append(f"(item {item}) {p}" if p.startswith("Question:") else p)
    return keep


def build(blocks: list[tuple[str, str, str]]) -> dict:
    pieces, used = [], set()

    def add(label: str, paras: list[str]):
        new = [p for p in paras if p not in used]
        used.update(new)
        if new:
            pieces.append(f"[{label}]\n" + "\n".join(new))

    intro = [p for s, _, p in blocks if INTRO_RE.search(s)]
    intro_text = "\n".join(intro)[-INTRO_TAIL:] if intro else ""
    if intro_text:
        pieces.append("[Introduction, last part]\n" + intro_text)
        used.update(intro)
    headings = []
    for s, h, p in blocks:
        if (SECTION_RE.search(h) or SECTION_RE.search(s)) and not SKIP_RE.search(h) and not CHECKLIST_RE.search(s) \
                and not INTRO_RE.search(s) and h not in headings:
            headings.append(h)
    def rank(h: str) -> int:
        low = h.lower()
        return min((i for i, k in enumerate(PRIORITY) if k in low), default=len(PRIORITY))
    for h in sorted(headings, key=rank):  # stable: document order within a priority level
        text = "\n".join(p for s, hh, p in blocks if hh == h and p not in used)
        if text:
            used.update(p for s, hh, p in blocks if hh == h)
            pieces.append(f"[{h}]\n" + (text if len(text) <= SECTION_CAP else text[:SECTION_CAP] + " [...]"))
    checklist = [p for s, _, p in blocks if CHECKLIST_RE.search(s)]
    items = checklist_items(checklist)
    if items:
        add("NeurIPS checklist, items 4-8 and 13", items)
    regex_paras = [p for s, h, p in blocks if P8_PATTERN.search(p) and not SKIP_RE.search(h) and p not in used]
    add("Other paragraphs on data, code and releases", regex_paras)
    full = "\n\n".join(pieces)
    return dict(excerpt=full[:CAP], chars=min(len(full), CAP), truncated=len(full) > CAP, intro_found=bool(intro_text),
                matched_headings=headings[:40], checklist_found=bool(checklist), checklist_items_found=bool(items),
                n_regex_paragraphs=len(regex_paras))


def main(limit: int | None) -> dict:
    recs = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        recs[r["key"]] = r
    ok = [r for r in recs.values() if r["status"] == "ok"][:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    stats, rows = Counter(), []
    for r in ok:
        raw = gzip.decompress((DS / r["path"]).read_bytes())
        try:
            blocks = blocks_html(raw) if r["source"] == "arxiv_html" else blocks_pdf(raw)
            ex = build(blocks)
            status = "ok" if ex["excerpt"] else "empty"
        except Exception as exc:  # counted and reported, the paper is excluded from coding
            ex, status = dict(excerpt="", error=f"{type(exc).__name__}: {str(exc)[:120]}"), "error"
        rows.append(dict(key=r["key"], stratum=r["stratum"], group=r["group"], source=r["source"], status=status, **ex))
        stats[(r["source"], status)] += 1
    with (OUT / "excerpts.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    good = [x for x in rows if x["status"] == "ok"]
    summary = dict(papers_with_fulltext=len(ok), by_source_and_status={f"{a}|{b}": n for (a, b), n in sorted(stats.items())},
                   truncated=sum(x["truncated"] for x in good), intro_found=sum(x["intro_found"] for x in good),
                   checklist_items_found=sum(x["checklist_items_found"] for x in good),
                   median_chars=sorted(x["chars"] for x in good)[len(good) // 2] if good else None,
                   with_matched_section=sum(bool(x["matched_headings"]) for x in good))
    if limit is None:
        (HERE / "results" / "p9_excerpts_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":
    lim = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--limit" else None
    print(json.dumps(main(lim), indent=1))
