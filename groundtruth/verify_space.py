"""verify_space.py — post-publish structural check against the reference Space.
Usage: python3 verify_space.py <space_id>
Checks (read-only on HF):
  - index.html exists, is NOT the HF stub, references logbook.js + logbook.css
  - logbook.json: schema_version==2 and root.children non-empty
  - at least 1 pages/<slug>/page.md present (real claim content)
  - workspace.json + at least one .png in root
Exits 0 if all pass, 1 otherwise. No suite — invoked per published Space.
"""
import sys, json, os
from huggingface_hub import HfApi, hf_hub_download

REF_CHILDREN = 8  # reference Space has 8 root.children (exec-summary + 6 claims + conclusion)

def main():
    sid = sys.argv[1]
    api = HfApi()
    try:
        fs = list(api.list_repo_files(repo_id=sid, repo_type="space"))
    except Exception as e:
        print(f"[FAIL] cannot list {sid}: {e}"); sys.exit(1)
    root = [f for f in fs if not f.startswith("logbook/")]
    problems = []

    # index.html
    try:
        ih = open(hf_hub_download(repo_id=sid, repo_type="space", filename="index.html")).read()
        if "Welcome to your static Space" in ih:
            problems.append("index.html is HF stub")
        if "logbook.js" not in ih or "logbook.css" not in ih:
            problems.append("index.html missing logbook.js/css refs")
    except Exception as e:
        problems.append(f"index.html missing: {e}")

    # assets
    if not any(f.endswith(".png") for f in root):
        problems.append("no .png asset in root")
    if "workspace.json" not in root:
        problems.append("workspace.json missing")

    # logbook.json
    try:
        lj = json.load(open(hf_hub_download(repo_id=sid, repo_type="space", filename="logbook.json")))
        if lj.get("schema_version") != 2:
            problems.append(f"schema_version={lj.get('schema_version')} != 2")
        kids = lj.get("root", {}).get("children", [])
        if not kids:
            problems.append("root.children empty")
    except Exception as e:
        problems.append(f"logbook.json missing/bad: {e}")
        kids = []

    # claim pages
    claim_pages = sum(1 for f in fs if f.startswith("pages/claim-") and f.endswith("/page.md"))
    if claim_pages == 0:
        problems.append("no pages/<slug>/page.md claim content")

    ok = len(problems) == 0
    n = len(kids)
    print(f"{sid}: root={len(root)} png={'Y' if any(f.endswith('.png') for f in root) else 'N'} "
          f"ws={'Y' if 'workspace.json' in root else 'N'} claim_pages={claim_pages} children={n}")
    if ok:
        print("  RESULT: OK — renders like reference")
    else:
        print("  RESULT: FAIL")
        for p in problems:
            print("   -", p)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
