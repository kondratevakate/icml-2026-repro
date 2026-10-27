#!/usr/bin/env python3
"""Publish the quota-blocked Codex logbooks from the codex branch to HF Spaces.

Each logbook lives at <dir>/.trackio/logbook/ in origin/codex/medical-reproducibility-map.
We export that tree to a temp dir, verify it locally, and upload it as a static Space
tagged `icml2026-repro` + `paper-<orid>` so the challenge judge picks it up.

Usage: python3 publish_logbooks.py [--dry-run] [dir ...]
"""
import json, os, re, shutil, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRANCH = "origin/codex/medical-reproducibility-map"
OWNER = "kondratevakate"
DIRS = ["globalhealthatlas", "supgcl", "glean", "caml", "dpsurv",
        "lvcg", "bayes_causal_meta", "sprout", "medcrp_cl"]


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True)


def export(d, dest):
    """Extract <d>/.trackio/logbook/** from the branch into dest/; return written paths."""
    written = []
    for f in git("ls-tree", "-r", "--name-only", BRANCH, "--",
                 f"{d}/.trackio/logbook").stdout.splitlines():
        if not f.strip():
            continue
        rel = f.split(".trackio/logbook/", 1)[1]
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "wb").write(subprocess.run(["git", "-C", REPO, "show", f"{BRANCH}:{f}"],
                                             capture_output=True).stdout)
        written.append(rel)
    return written


def read_meta(d):
    r = git("show", f"{BRANCH}:{d}/.trackio/metadata.json")
    return json.loads(r.stdout) if r.returncode == 0 else {}


def orid_of(dest, meta):
    """Recover the paper orid from the `paper-<orid>` tag Codex already wrote."""
    for t in meta.get("tags", []):
        if t.startswith("paper-") and len(t) == 16:
            return t[6:]
    lb = os.path.join(dest, "logbook.json")
    if os.path.exists(lb):
        m = re.search(r"forum\?id=([A-Za-z0-9]{10})", json.dumps(json.load(open(lb))))
        if m:
            return m.group(1)
    return None


def main():
    dry = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    targets = args or DIRS

    from huggingface_hub import HfApi
    api = HfApi()
    existing = {s.id for s in api.list_spaces(author=OWNER)}
    results = []

    for d in targets:
        tmp = tempfile.mkdtemp(prefix=f"logbook-{d}-")
        try:
            files = export(d, tmp)
            meta = read_meta(d)
            orid = orid_of(tmp, meta)
            has_index = os.path.exists(os.path.join(tmp, "index.html"))
            pages = len([f for f in files if f.startswith("pages/") and f.endswith("page.md")])
            slug = meta.get("space_id") or f"{OWNER}/repro-{d.replace('_','-')}"
            row = {"dir": d, "orid": orid, "files": len(files), "pages": pages,
                   "index_html": has_index, "space": slug,
                   "already_exists": slug in existing}
            if not (files and orid and has_index and pages):
                row["status"] = "SKIP: incomplete bundle"
            elif dry:
                row["status"] = ("DRY-RUN would repair" if row["already_exists"]
                                 else "DRY-RUN ok")
            else:
                api.create_repo(slug, repo_type="space", space_sdk="static",
                                exist_ok=True, private=False)
                api.upload_folder(folder_path=tmp, repo_id=slug, repo_type="space",
                                  commit_message=f"Publish reproduction logbook for {orid}")
                tags = meta.get("tags") or ["icml2026-repro", f"paper-{orid}"]
                for t in ("trackio", "trackio-logbook", "open-experiment"):
                    if t not in tags:
                        tags.insert(0, t)
                card = {"title": meta.get("title") or f"Repro: {d}",
                        "emoji": meta.get("emoji", "🎯"),
                        "colorFrom": "blue", "colorTo": "green",
                        "sdk": "static", "pinned": False, "tags": tags}
                readme = ("---\n" + "\n".join(
                    f"{k}: {json.dumps(v)}" for k, v in card.items()) + "\n---\n")
                api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                                repo_id=slug, repo_type="space",
                                commit_message="Add challenge tags")
                row["status"] = "REPAIRED" if row["already_exists"] else "PUBLISHED"
                row["url"] = f"https://huggingface.co/spaces/{slug}"
            results.append(row)
            print(f"{row['status']:28} {d:20} orid={orid} files={len(files)} pages={pages}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"\npublished={sum(r['status']=='PUBLISHED' for r in results)} "
          f"repaired={sum(r['status']=='REPAIRED' for r in results)} "
          f"skipped={sum(r['status'].startswith('SKIP') for r in results)}")
    json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "out", "publish_log.json"), "w"), indent=1)
    return results


if __name__ == "__main__":
    main()
