"""Fetch ICLR 2026 abstracts from per-paper virtual-site pages (Protocol 7, Amendment 1). Resumable."""
from __future__ import annotations

import html
import json
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = DATA / "iclr-2026_abstracts.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (research audit of public conference metadata)"}
ABSTRACT = re.compile(r'(?is)<div[^>]*id="abstract[^"]*"[^>]*>(.*?)</div>')
RATE = 2.0
_lock, _next = threading.Lock(), [0.0]


def throttle() -> None:
    with _lock:
        now = time.monotonic()
        wait = max(0.0, _next[0] - now)
        _next[0] = max(now, _next[0]) + 1 / RATE
    time.sleep(wait)


def fetch(path: str) -> dict:
    for attempt in range(3):
        try:
            throttle()
            with urllib.request.urlopen(urllib.request.Request("https://iclr.cc" + path, headers=HEADERS), timeout=60) as r:
                body = r.read().decode("utf-8", "replace")
            m = ABSTRACT.search(body)
            if not m:
                return dict(status="no_abstract_element")
            return dict(status="ok", abstract=" ".join(html.unescape(re.sub(r"<[^>]+>", " ", m.group(1))).split()))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = f"{type(exc).__name__}: {str(exc)[:120]}"
            time.sleep(10 * (attempt + 1))
    return dict(status="error", error=last)


def main() -> None:
    raw = json.loads((DATA / "iclr-2026.json").read_text(encoding="utf-8"))
    items = raw["results"] if isinstance(raw, dict) else raw
    paths = sorted({str(i["virtualsite_url"]) for i in items if i.get("virtualsite_url")})
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    todo = [p for p in paths if done.get(p, {}).get("status") != "ok"]
    print(f"pages {len(paths)}, to fetch {len(todo)}", flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(4) as pool:
        futures = {pool.submit(fetch, p): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            done[futures[fut]] = fut.result()
            if i % 500 == 0 or i == len(todo):
                OUT.write_text(json.dumps(done), encoding="utf-8")
                el = time.monotonic() - started
                print(f"{i}/{len(todo)} {el:.0f}s, est. total {el * len(todo) / i / 60:.0f} min", flush=True)
    OUT.write_text(json.dumps(done), encoding="utf-8")
    from collections import Counter
    print(Counter(v["status"] for v in done.values()))


if __name__ == "__main__":
    main()
