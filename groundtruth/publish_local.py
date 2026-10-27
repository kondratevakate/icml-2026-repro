#!/usr/bin/env python3
"""publish_local.py — publish a local reproduction folder's .trackio bundle to HF
Spaces, but ONLY if the local score is strictly better than what is already live.

Compares local _score.json (pts/10) against the currently-published Space's
_score.json (fetched from the Space repo). If local is better (or Space missing),
upload. Otherwise skip.

Usage: python3 publish_local.py <dir> [--dry-run]
"""
import json, os, sys, argparse
from huggingface_hub import HfApi

OWNER = "kondratevakate"
SPACE_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "space_map.json")


def resolve_space_id(api, orid, fallback_space_id):
    """Never create a duplicate Space for a paper that already has one.

    1) Consult space_map.json (hard orid->space_id mapping, maintained manually/by script).
    2) Best-effort: scan all owner Spaces' README for `paper-<orid>` tag.
    3) Fallback: generated slug (only if truly new paper).
    Returns (space_id, existed_bool)."""
    # 1) hard map
    try:
        mp = json.load(open(SPACE_MAP))
        if orid in mp:
            return mp[orid], True
    except Exception:
        pass
    # 2) tag scan
    try:
        for s in api.list_spaces(author=OWNER):
            sid = s.id
            try:
                readme = api.hf_hub_download(repo_id=sid, repo_type="space",
                                            filename="README.md", local_files_only=False)
                if f"paper-{orid}" in open(readme).read():
                    return sid, True
            except Exception:
                continue
    except Exception as e:
        print(f"  (tag-scan skipped: {type(e).__name__})")
    return fallback_space_id, False


def register_space(orid, space_id):
    """Record orid->space_id in space_map.json so future runs EDIT, never duplicate."""
    try:
        mp = json.load(open(SPACE_MAP))
    except Exception:
        mp = {}
    if orid not in mp:
        mp[orid] = space_id
        json.dump(mp, open(SPACE_MAP, "w"), indent=2)
        print(f"  registered {orid} -> {space_id} in space_map.json")


def pts_of(score_path):
    if not os.path.exists(score_path):
        return None
    d = json.load(open(score_path))
    pv = d.get("per_claim_verdict") or {}
    if not isinstance(pv, dict):
        return None
    n = len(pv)
    pts = 0
    for v in pv.values():
        vl = (v or "").lower()
        if "verified" in vl or "falsified" in vl:
            pts += 2
        elif "toy" in vl:
            pts += 1
    return round(pts * 10 / (2 * n), 1) if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="publish even if local_pts <= remote_pts (e.g. refreshed logbook, same score)")
    a = ap.parse_args()
    d = os.path.abspath(a.dir)
    trackio = os.path.join(d, ".trackio")
    meta_p = os.path.join(trackio, "metadata.json")
    if not os.path.exists(meta_p):
        raise SystemExit(f"no .trackio/metadata.json in {d}; run local_to_trackio.py first")
    meta = json.load(open(meta_p))
    space_id = meta.get("space_id") or f"{OWNER}/repro-{os.path.basename(d)}"
    orid = next((t[6:] for t in meta.get("tags", []) if t.startswith("paper-")), "?")
    api = HfApi()
    # resolve by paper-<orid> tag across ALL owner spaces -> EDIT existing, never duplicate
    resolved, existed = resolve_space_id(api, orid, space_id)
    if existed and resolved != space_id:
        print(f"  NOTE: editing EXISTING space {resolved} (tag paper-{orid}) instead of {space_id}")
        space_id = resolved
    local_pts = pts_of(os.path.join(d, "_score.json"))
    print(f"local  {orid}: pts={local_pts}  space={space_id}")
    remote_pts = None
    try:
        remote_score = api.hf_hub_download(repo_id=space_id, repo_type="space",
                                           filename="logbook/_score.json",
                                           local_files_only=False)
        remote_pts = pts_of(remote_score)
        print(f"remote {orid}: pts={remote_pts}")
    except Exception as e:
        print(f"remote {orid}: not found ({type(e).__name__})")

    if remote_pts is not None and local_pts is not None and local_pts <= remote_pts:
        if a.force:
            print(f"FORCE: local {local_pts} <= remote {remote_pts} but --force set, publishing")
        else:
            print(f"SKIP: local {local_pts} <= remote {remote_pts} (no improvement)")
            return
    if a.dry_run:
        print("DRY-RUN: would publish (local better or remote missing)")
        return

    api.create_repo(space_id, repo_type="space", space_sdk="static", exist_ok=True,
                    private=False)
    # The full trackio bundle (index.html, logbook.js/css, *.png, workspace.json,
    # pages/<slug>/page.md) lives in .trackio/logbook/. Upload it to the Space ROOT
    # so HF static renderer shows it (root index.html). Remove any stale nested
    # logbook/ left by earlier broken publishes.
    lb_root = os.path.join(trackio, "logbook")
    if not os.path.isdir(lb_root):
        raise SystemExit(f"no .trackio/logbook in {d}; run local_to_trackio.py first")
    api.upload_folder(folder_path=lb_root, repo_id=space_id, repo_type="space",
                      path_in_repo=None, delete_patterns=["logbook/*"],
                      commit_message=f"Reproduction logbook for {orid} (local pts={local_pts})")
    # tags
    tags = meta.get("tags") or ["icml2026-repro", f"paper-{orid}"]
    for t in ("trackio", "trackio-logbook", "open-experiment"):
        if t not in tags:
            tags.insert(0, t)
    card = {"title": meta.get("title", orid), "emoji": meta.get("emoji", "🎯"),
            "colorFrom": "blue", "colorTo": "green", "sdk": "static",
            "pinned": False, "tags": tags}
    readme = "---\n" + "\n".join(f"{k}: {json.dumps(v)}" for k, v in card.items()) + "\n---\n"
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=space_id, repo_type="space",
                    commit_message="Add challenge tags")
    register_space(orid, space_id)
    print(f"PUBLISHED {space_id}  url=https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
