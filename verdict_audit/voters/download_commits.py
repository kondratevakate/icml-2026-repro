"""Download commit ids and times per Space (PROTOCOL.md, Protocol 2). Resumable; stores no authors or messages."""
from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

import common as C
from download import API, throttle

MANIFEST = C.DATA / "commits_manifest.json"


def fetch(space_id: str) -> dict:
    last = None
    for attempt in range(4):
        try:
            throttle()
            commits = API.list_repo_commits(space_id, repo_type="space")
            return dict(status="ok", commits=[[c.commit_id, c.created_at.isoformat()] for c in commits])
        except RepositoryNotFoundError as exc:
            return dict(status="missing", error=type(exc).__name__)
        except (HfHubHTTPError, OSError, httpx.HTTPError) as exc:
            last = f"{type(exc).__name__}: {str(exc)[:200]}"
            time.sleep(150 if "429" in last else 10 * (attempt + 1))
    return dict(status="error", error=last)


def main() -> None:
    C.check_inputs()
    logbooks, _, _, anchored, _ = C.load()
    multi = C.multi_rated_claims(logbooks, anchored)
    wanted = sorted(set().union(*[set(v) for v in multi.values()]))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    todo = [s for s in wanted if manifest.get(s, {}).get("status") not in ("ok", "missing")]
    print(f"wanted {len(wanted)}, to fetch {len(todo)}", flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(4) as pool:
        futures = {pool.submit(fetch, s): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            manifest[futures[fut]] = fut.result()
            if i % 250 == 0 or i == len(todo):
                MANIFEST.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
                el = time.monotonic() - started
                print(f"{i}/{len(todo)} {el:.0f}s, est. total {el * len(todo) / i / 60:.0f} min", flush=True)
    MANIFEST.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    print(Counter(m["status"] for m in manifest.values()))


if __name__ == "__main__":
    main()
