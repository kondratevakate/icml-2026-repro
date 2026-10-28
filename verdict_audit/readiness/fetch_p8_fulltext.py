"""H8.5 full texts: arXiv HTML, else PDF via pypdf; data/code excerpts capped at 6,000 characters. Resumable.

Writes data/p8/fulltext/<orid>.json and, when done, the H8.5 coding sheets.
"""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import time
import urllib.error
import urllib.request

import p8_common as P

HEADERS = {"User-Agent": "Mozilla/5.0 (research audit of reproducibility; arXiv full text, 1 request per 3 s)"}
PATTERN = re.compile(
    r"dataset|data availab|publicly available|public|private|proprietary|in-house|institutional|IRB|ethic|consent|"
    r"synthetic|simulated|toy|sample data|demo|download|access|PhysioNet|credential|request|licen|weights|checkpoint|code",
    re.IGNORECASE)
CAP = 6000
OUT = P.P8 / "fulltext"


def get(url: str) -> bytes | None:
    time.sleep(3)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=90) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def paragraphs_html(raw: bytes) -> list[str]:
    body = raw.decode("utf-8", "replace")
    body = re.sub(r"(?is)<(script|style|math|nav|header|footer)[^>]*>.*?</\1>", " ", body)
    paras = re.findall(r"(?is)<p[^>]*>(.*?)</p>", body)
    return [" ".join(html.unescape(re.sub(r"<[^>]+>", " ", p)).split()) for p in paras]


def paragraphs_pdf(raw: bytes) -> list[str]:
    from pypdf import PdfReader
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
    return [" ".join(p.split()) for p in re.split(r"\n\s*\n|\.\n", text)]


def excerpt(paras: list[str]) -> str:
    out, n = [], 0
    for p in paras:
        if len(p) > 40 and PATTERN.search(p):
            out.append(p)
            n += len(p)
            if n >= CAP:
                break
    return "\n\n".join(out)[:CAP]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    papers = json.loads((P.P8 / "h85_papers.json").read_text(encoding="utf-8"))
    titles = {p["orid"]: p["title"] for p in json.loads((P.ROOT / "papers.json").read_text(encoding="utf-8-sig"))["papers"]}
    for i, p in enumerate(papers, 1):
        path = OUT / f"{p['orid']}.json"
        if path.exists():
            continue
        rec = dict(orid=p["orid"], arxiv=p["arxiv"], status="no_arxiv_id")
        if p["arxiv"]:
            try:
                raw = get(f"https://arxiv.org/html/{p['arxiv']}")
                paras, source = (paragraphs_html(raw), "html") if raw else ([], None)
                if not raw or len(" ".join(paras)) < 2000:
                    raw = get(f"https://arxiv.org/pdf/{p['arxiv']}")
                    paras, source = (paragraphs_pdf(raw), "pdf") if raw else ([], None)
                text = excerpt(paras)
                rec.update(status="ok" if text else "no_excerpt", source=source,
                           raw_sha256=hashlib.sha256(raw).hexdigest() if raw else None, excerpt=text)
            except Exception as exc:  # recorded and counted, retried on the next run
                rec.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:150]}")
        if rec["status"] != "error":
            path.write_text(json.dumps(rec), encoding="utf-8")
        if i % 20 == 0:
            print(f"{i}/{len(papers)}", flush=True)
    recs = [json.loads((OUT / f"{p['orid']}.json").read_text(encoding="utf-8")) for p in papers if (OUT / f"{p['orid']}.json").exists()]
    from collections import Counter
    print(Counter(r["status"] for r in recs), "missing files:", len(papers) - len(recs))
    if len(recs) < len(papers):
        return
    import random
    rng = random.Random(P.SEED_SAMPLE)
    ok = [r for r in recs if r["status"] == "ok"]
    rng.shuffle(ok)
    keys = json.loads((P.P8 / "keys.json").read_text(encoding="utf-8"))
    group = {p["orid"]: p["group"] for p in papers}
    rows = []
    for n, r in enumerate(ok):
        iid = f"F{n:04d}"
        keys[iid] = dict(task="h85", group=group[r["orid"]], orid=r["orid"])
        rows.append(dict(item_id=iid, title=titles.get(r["orid"], ""), excerpt=r["excerpt"]))
    (P.P8 / "keys.json").write_text(json.dumps(keys), encoding="utf-8")
    for role in ("coder1", "coder2"):
        for b in range(0, len(rows), 30):
            with (P.P8 / "sheets" / f"h85_{role}_batch_{b // 30:02d}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, ["item_id", "title", "excerpt"])
                w.writeheader()
                w.writerows(rows[b:b + 30])
    print("h85 sheets:", len(rows), "papers in", (len(rows) + 29) // 30, "batches per coder")


if __name__ == "__main__":
    main()
