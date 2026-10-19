"""Protocol 9 full texts: arXiv title search, arXiv HTML or PDF, then open proceedings PDFs (PMLR, CVF).

Raw responses are stored gzipped under the shared data root (never in git), one manifest line per paper.
Resumable: papers with a final status are skipped; errors are retried on the next run. At most one request
per 3 s per host family. Logs carry paper keys and counts only.

Usage: python fetch_p9.py index            -> metadata/venue_index.json (PMLR v267/v306, CVF CVPR/ICCV 2025)
       python fetch_p9.py run [--limit N]  -> raw/<source>/<key>.<ext>.gz, metadata/manifest.jsonl
"""
from __future__ import annotations

import datetime as dt
import difflib
import gzip
import hashlib
import html
import json
import os
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"C:\Projects\data"))
DS = DATA_ROOT / "datasets" / "literature" / "ml-venue-fulltext" / "2025-2026-v1"
RAW, META = DS / "raw", DS / "metadata"
MANIFEST = META / "manifest.jsonl"
HEADERS = {"User-Agent": "Mozilla/5.0 (research audit of reproducibility practice; 1 request per 3 s per host)"}
DELAY, SEED, MIN_TEXT, RATIO = 3.0, 20261019, 2000, 0.95
# arXiv answered 429 at one request per 3 s on 2026-10-19 after ~3,000 papers; it gets 6 s (still within the
# protocol's 'at most one request per 3 s per host').
DELAY_BY_FAMILY = {"arxiv": 6.0}
FINAL = ("ok", "no_match", "ambiguous", "no_fulltext")
VENUE_INDEX = {
    "icml-2025": ("pmlr", "https://proceedings.mlr.press/v267/"),
    "icml-2026": ("pmlr", "https://proceedings.mlr.press/v306/"),
    "cvpr-2025": ("cvf", "https://openaccess.thecvf.com/CVPR2025?day=all"),
    "iccv-2025": ("cvf", "https://openaccess.thecvf.com/ICCV2025?day=all"),
}
_last: dict[str, float] = {}


def family(url: str) -> str:
    host = urllib.parse.urlparse(url).hostname or ""
    return "arxiv" if host.endswith("arxiv.org") else host


def get(url: str) -> tuple[bytes, str] | None:
    """GET with per-host-family spacing; None on 404; retries 429/5xx three times, then raises."""
    for attempt in range(4):
        wait = DELAY_BY_FAMILY.get(family(url), DELAY) - (time.monotonic() - _last.get(family(url), 0.0))
        if wait > 0:
            time.sleep(wait)
        _last[family(url)] = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=90) as r:
                return r.read(), r.geturl()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(60 * 2 ** attempt)  # 60, 120, 240 s
                continue
            raise
    raise RuntimeError("unreachable")


def norm(title: str) -> str:
    t = unicodedata.normalize("NFKD", html.unescape(title)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def similar(a: str, b: str) -> float:
    return 1.0 if a == b else difflib.SequenceMatcher(None, a, b).ratio()


def parse_atom(feed: str) -> list[dict]:
    out = []
    for e in feed.split("<entry>")[1:]:
        m_id = re.search(r"<id>https?://arxiv\.org/abs/([^<]+)</id>", e)
        m_t = re.search(r"<title>(.*?)</title>", e, re.S)
        m_p = re.search(r"<published>(\d{4})-", e)
        if m_id and m_t and m_p:
            out.append(dict(id=re.sub(r"v\d+$", "", m_id.group(1)), title=" ".join(m_t.group(1).split()),
                            year=int(m_p.group(1))))
    return out


def pick(entries: list[dict], title: str, venue_year: int) -> tuple[str, str | None, float]:
    """Return (status, arxiv id, ratio): ok, ambiguous or no_match under the protocol 9 rule."""
    n = norm(title)
    ok = {}
    for e in entries:
        r = similar(n, norm(e["title"]))
        if r >= RATIO and e["year"] >= venue_year - 1:
            ok[e["id"]] = max(r, ok.get(e["id"], 0.0))
    if len(ok) > 1:
        return "ambiguous", None, max(ok.values())
    if ok:
        (i, r), = ok.items()
        return "ok", i, r
    return "no_match", None, 0.0


def search(title: str, venue_year: int) -> tuple[str, str | None, float]:
    words = norm(title).split()
    queries = ["ti:%22" + "+".join(words) + "%22"]
    long_words = sorted({w for w in words if len(w) > 3}, key=len, reverse=True)[:10]
    if long_words:
        queries.append("+AND+".join(f"ti:{w}" for w in long_words))
    for q in queries:
        got = get(f"https://export.arxiv.org/api/query?search_query={q}&max_results=10")
        status = pick(parse_atom(got[0].decode("utf-8", "replace")), title, venue_year) if got else ("no_match", None, 0.0)
        if status[0] != "no_match":
            return status
    return "no_match", None, 0.0


def html_text_len(raw: bytes) -> int:
    body = re.sub(r"(?is)<(script|style|math|nav|header|footer)[^>]*>.*?</\1>", " ", raw.decode("utf-8", "replace"))
    return sum(len(" ".join(html.unescape(re.sub(r"<[^>]+>", " ", p)).split()))
               for p in re.findall(r"(?is)<p[^>]*>(.*?)</p>", body))


def decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", "replace")


def parse_pmlr(page: str, base: str) -> dict:
    """forum id -> pdf url, from the OpenReview link listed with each paper."""
    out = {}
    for block in page.split('<div class="paper">')[1:]:
        pdf = re.search(r'href="([^"]+\.pdf)"', block)
        orl = re.search(r'openreview\.net/forum\?id=([A-Za-z0-9_-]+)', block)
        if pdf and orl:
            out[orl.group(1)] = urllib.parse.urljoin(base, pdf.group(1))
    return out


def parse_cvf(page: str, base: str) -> dict:
    """normalised title -> pdf url."""
    out = {}
    for href, title in re.findall(r'<dt class="ptitle">\s*(?:<br>)?\s*<a href="([^"]+)">(.*?)</a>', page, re.S):
        pdf = href.replace("/html/", "/papers/").replace(".html", ".pdf")
        out[norm(title)] = urllib.parse.urljoin(base, pdf)
    return out


def store(sub: str, key: str, ext: str, raw: bytes) -> tuple[str, str]:
    d = RAW / sub
    d.mkdir(parents=True, exist_ok=True)
    final = d / f"{key}.{ext}.gz"
    part = final.with_name(final.name + ".part")
    part.write_bytes(gzip.compress(raw, mtime=0))
    if gzip.decompress(part.read_bytes()) != raw:
        raise IOError(f"{key}: gzip round trip failed")
    part.replace(final)
    return str(final.relative_to(DS)).replace("\\", "/"), hashlib.sha256(raw).hexdigest()


def key_of(p: dict) -> str:
    return f"{p['stratum']}__{re.sub(r'[^A-Za-z0-9_-]', '-', p['forum_id'])}"


def index() -> None:
    META.mkdir(parents=True, exist_ok=True)
    out = {}
    for stratum, (kind, url) in VENUE_INDEX.items():
        got = get(url)
        if not got:
            raise SystemExit(f"index missing: {url}")
        raw, final_url = got
        path, sha = store("indexes", stratum, "html", raw)
        page = decode(raw)
        mapping = parse_pmlr(page, final_url) if kind == "pmlr" else parse_cvf(page, final_url)
        out[stratum] = dict(kind=kind, url=url, path=path, sha256=sha, n=len(mapping), map=mapping)
        print(stratum, kind, len(mapping), flush=True)
    (META / "venue_index.json").write_text(json.dumps(out), encoding="utf-8")


def done_keys() -> set[str]:
    if not MANIFEST.exists():
        return set()
    return {r["key"] for r in map(json.loads, MANIFEST.read_text(encoding="utf-8").splitlines()) if r["status"] in FINAL}


def order(frame: list[dict]) -> list[dict]:
    """ICML 2026 first (needed for Q9.4/Q9.5 and the pilot), the rest in seeded random order."""
    first = [p for p in frame if p["stratum"] == "icml-2026"]
    rest = [p for p in frame if p["stratum"] != "icml-2026"]
    random.Random(SEED).shuffle(rest)
    return first + rest


def fetch_one(p: dict, vidx: dict) -> dict:
    rec = dict(key=key_of(p), stratum=p["stratum"], forum_id=p["forum_id"], group=p["group"],
               retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    year = int(p["stratum"][-4:])
    status, aid, ratio = search(p["title"], year)
    rec.update(match=status, arxiv_id=aid, match_ratio=round(ratio, 4))
    if aid:
        got = get(f"https://arxiv.org/html/{aid}")
        if got and html_text_len(got[0]) >= MIN_TEXT:
            path, sha = store("arxiv_html", rec["key"], "html", got[0])
            return dict(rec, status="ok", source="arxiv_html", url=got[1], path=path, sha256=sha, bytes=len(got[0]))
        got = get(f"https://arxiv.org/pdf/{aid}")
        if got and got[0][:4] == b"%PDF":
            path, sha = store("arxiv_pdf", rec["key"], "pdf", got[0])
            return dict(rec, status="ok", source="arxiv_pdf", url=got[1], path=path, sha256=sha, bytes=len(got[0]))
    v = vidx.get(p["stratum"])
    url = None
    if v and v["kind"] == "pmlr":
        url = v["map"].get(p["forum_id"])
    elif v and v["kind"] == "cvf":
        n = norm(p["title"])
        hits = [u for t, u in v["map"].items() if similar(n, t) >= RATIO]
        url = hits[0] if len(hits) == 1 else None
    if url:
        got = get(url)
        if got and got[0][:4] == b"%PDF":
            path, sha = store("venue_pdf", rec["key"], "pdf", got[0])
            return dict(rec, status="ok", source=v["kind"], url=got[1], path=path, sha256=sha, bytes=len(got[0]))
    return dict(rec, status="no_fulltext" if aid or url else status)


def run(limit: int | None) -> None:
    META.mkdir(parents=True, exist_ok=True)
    frame = json.loads((HERE / "results" / "p9_frame.json").read_text(encoding="utf-8"))
    vidx = json.loads((META / "venue_index.json").read_text(encoding="utf-8"))
    done = done_keys()
    todo = [p for p in order(frame) if key_of(p) not in done][:limit]
    stats = Counter()
    for i, p in enumerate(todo, 1):
        try:
            rec = fetch_one(p, vidx)
        except Exception as exc:  # recorded, retried on the next run
            rec = dict(key=key_of(p), stratum=p["stratum"], forum_id=p["forum_id"], group=p["group"],
                       status="error", error=f"{type(exc).__name__}: {str(exc)[:150]}")
        with MANIFEST.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        stats[rec["status"] if rec["status"] != "ok" else rec["source"]] += 1
        if i % 25 == 0 or i == len(todo):
            print(f"{i}/{len(todo)} {dict(stats)}", flush=True)
    print("done:", len(done_keys()), "of", len(frame), "papers with a final status", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "index":
        index()
    else:
        run(int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--limit" else None)
