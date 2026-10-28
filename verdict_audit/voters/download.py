"""Download logbook pages of multi-rated logbooks at their judged revision (PROTOCOL.md, Download).

Resumable: Spaces with status ok in the manifest are skipped. Writes pages_manifest.json.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import hashlib

import threading
from urllib.parse import quote

import httpx
import requests
from huggingface_hub import HfApi, get_session
from huggingface_hub.utils import build_hf_headers
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError

import common as C

API = HfApi()
SESSION = get_session()
HEADERS = build_hf_headers()
MANIFEST = C.DATA / "pages_manifest.json"
API_PER_SECOND = 2.5  # 'api' bucket is 1000 calls / 300 s; stay at 750
_lock = threading.Lock()
_next_api = [0.0]


def throttle() -> None:
    with _lock:
        now = time.monotonic()
        wait = max(0.0, _next_api[0] - now)
        _next_api[0] = max(now, _next_api[0]) + 1 / API_PER_SECOND
    time.sleep(wait)


def wanted_path(path: str) -> bool:
    return path == "logbook.json" or (path.startswith("pages/") and path.endswith(".md"))


def get_file(space_id: str, sha: str, path: str) -> bytes:
    url = f"https://huggingface.co/spaces/{space_id}/resolve/{sha}/{quote(path)}"
    for _ in range(6):
        r = SESSION.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", "60")) + 1)
            continue
        r.raise_for_status()
        return r.content
    raise OSError(f"429 persisted for {path}")


def fetch(space_id: str, sha: str) -> dict:
    """Store each file as <dir>/<sha1(path)[:16]>.txt; Windows cannot hold the long page slugs."""
    target = C.local_dir(space_id)
    last = None
    for attempt in range(4):
        try:
            throttle()
            paths = sorted(p for p in API.list_repo_files(space_id, repo_type="space", revision=sha)
                           if wanted_path(p))
            target.mkdir(parents=True, exist_ok=True)
            files = {}
            for path in paths:
                data = get_file(space_id, sha, path)
                (target / C.file_key(path)).write_bytes(data)
                files[path] = hashlib.sha256(data).hexdigest()
            return dict(status="ok" if files else "empty", sha=sha, files=files)
        except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
            return dict(status="missing", sha=sha, error=type(exc).__name__)
        except (HfHubHTTPError, OSError, requests.RequestException, httpx.HTTPError) as exc:
            last = f"{type(exc).__name__}: {str(exc)[:200]}"
            if "429" in last:
                time.sleep(150)
            else:
                time.sleep(10 * (attempt + 1))
    return dict(status="error", sha=sha, error=last)


def main() -> None:
    C.check_inputs()
    logbooks, _, _, anchored, _ = C.load()
    multi = C.multi_rated_claims(logbooks, anchored)
    wanted = sorted(set().union(*[set(v) for v in multi.values()]))
    sha = {r["space_id"]: r["sha"] for r in logbooks}
    C.DATA.mkdir(exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    todo = [s for s in wanted if manifest.get(s, {}).get("status") not in ("ok", "empty", "missing")]
    print(f"wanted {len(wanted)}, to fetch {len(todo)}", flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(8) as pool:
        futures = {pool.submit(fetch, s, sha[s]): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            manifest[futures[fut]] = fut.result()
            if i % 200 == 0 or i == len(todo):
                MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
                el = time.monotonic() - started
                print(f"{i}/{len(todo)} {el:.0f}s, est. total {el * len(todo) / i / 60:.0f} min", flush=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    from collections import Counter
    print(Counter(m["status"] for m in manifest.values()))


if __name__ == "__main__":
    main()
