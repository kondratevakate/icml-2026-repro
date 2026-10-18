"""Download file trees (path, blob id, size; no content) at both revisions of new-revision events (Protocol 3)."""
from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError

import common as C
import resubmission_events as R
from download import API, throttle

MANIFEST = C.DATA / "trees_manifest.json"


def key(space: str, sha: str) -> str:
    return f"{space}@{sha}"


def fetch(space: str, sha: str) -> dict:
    last = None
    for attempt in range(4):
        try:
            throttle()
            files = {}
            for item in API.list_repo_tree(space, repo_type="space", revision=sha, recursive=True):
                if hasattr(item, "blob_id"):
                    files[item.path] = [item.blob_id, item.size]
            return dict(status="ok", files=files)
        except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
            return dict(status="missing", error=type(exc).__name__)
        except (HfHubHTTPError, OSError, httpx.HTTPError) as exc:
            last = f"{type(exc).__name__}: {str(exc)[:200]}"
            time.sleep(150 if "429" in last else 10 * (attempt + 1))
    return dict(status="error", error=last)


def main() -> None:
    recs = R.load_records()
    wanted = sorted({(e["space"], e[r]["sha"]) for e in R.events(recs) if e["kind"] == "new_sha" for r in ("r1", "r2")})
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    todo = [w for w in wanted if manifest.get(key(*w), {}).get("status") not in ("ok", "missing")]
    print(f"wanted {len(wanted)}, to fetch {len(todo)}", flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(4) as pool:
        futures = {pool.submit(fetch, *w): w for w in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            manifest[key(*futures[fut])] = fut.result()
            if i % 250 == 0 or i == len(todo):
                MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")
                el = time.monotonic() - started
                print(f"{i}/{len(todo)} {el:.0f}s, est. total {el * len(todo) / i / 60:.0f} min", flush=True)
    MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")
    print(Counter(m["status"] for m in manifest.values()))


if __name__ == "__main__":
    main()
