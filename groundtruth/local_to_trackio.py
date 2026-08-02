#!/usr/bin/env python3
"""local_to_trackio.py — adapt a local reproduction run folder into a .trackio
logbook bundle ready for HF Spaces publish.

Input:  <dir>/  containing logbook.md, _score.json, input_bundle.json, _run_meta.json
Output: <dir>/.trackio/{metadata.json, logbook/{index.html,logbook.js,logbook.css,
        logbook.json, README.md, pages/*.md}}

The trackio static files (index.html/js/css) are copied from a reference space
that already publishes correctly (default: ccd). This avoids hand-writing the
trackio runtime.

Usage: python3 local_to_trackio.py <dir> [--ref <reference_dir_with_.trackio>]
"""
import json, os, re, shutil, sys, argparse

DEFAULT_REF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ccd")


def slugify(orid, title):
    s = re.sub(r"[^a-z0-9]+", "-", (title or orid).lower()).strip("-")
    return s[:60] or orid


def split_pages(logbook_md):
    """Split logbook.md into trackio pages. A page per '## Claim N ...' and
    preserve '## Summary' / '## Conclusion' / '## Executive' sections."""
    lines = logbook_md.splitlines()
    pages = []  # (title, slug, body_lines)
    cur = None
    buf = []
    for ln in lines:
        m = re.match(r"^##\s+(.*)$", ln)
        if m:
            if cur is not None:
                pages.append((cur[0], cur[1], buf))
            title = m.group(1).strip()
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80] or "section"
            cur = (title, slug)
            buf = [ln]
        else:
            if cur is not None:
                buf.append(ln)
    if cur is not None:
        pages.append((cur[0], cur[1], buf))
    if not pages:
        pages = [("logbook", "logbook", lines)]
    return pages


def build(dir_path, ref_dir):
    dir_path = os.path.abspath(dir_path)
    lb = os.path.join(dir_path, "logbook.md")
    if not os.path.exists(lb):
        raise SystemExit(f"no logbook.md in {dir_path}")
    orid = os.path.basename(dir_path).replace("_arm1","").replace("_arm2","").replace("_arm3","").replace("_arm4","")
    # try input_bundle / _run_meta for title+arxiv
    title, arxiv = orid, ""
    ib = os.path.join(dir_path, "input_bundle.json")
    if os.path.exists(ib):
        d = json.load(open(ib))
        title = d.get("title") or title
        arxiv = d.get("arxiv_id") or ""
    rm = os.path.join(dir_path, "_run_meta.json")
    if os.path.exists(rm):
        d = json.load(open(rm))
        title = d.get("paper_title") or title
        arxiv = d.get("arxiv_id") or arxiv
        orid = d.get("openreview_id") or orid
    slug = slugify(orid, title)
    space_id = f"kondratevakate/repro-{slug}"
    out = os.path.join(dir_path, ".trackio")
    lbdir = os.path.join(out, "logbook")
    os.makedirs(os.path.join(lbdir, "pages"), exist_ok=True)

    # 1) copy trackio static from ref
    ref_lb = os.path.join(ref_dir, ".trackio", "logbook")
    for f in ("index.html", "logbook.js", "logbook.css", "logbook.json", "README.md",
              "bucket-icon.svg"):
        src = os.path.join(ref_lb, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(lbdir, f))

    # 2) split logbook into pages
    pages = split_pages(open(lb).read())
    index_lines = ["# Logbook pages", ""]
    for title, pslug, body in pages:
        ppath = os.path.join(lbdir, "pages", f"{pslug}.md")
        open(ppath, "w").write("\n".join(body))
        index_lines.append(f"- [{title}](pages/{pslug}.md)")
    open(os.path.join(lbdir, "pages", "index.md"), "w").write("\n".join(index_lines) + "\n")

    # 3) metadata.json
    meta = {
        "space_id": space_id,
        "emoji": "🎯",
        "created_at": "",
        "last_page": slug,
        "tags": ["icml2026-repro", f"paper-{orid}"],
        "paper": {"arxiv_id": arxiv},
        "private": False,
        "autosync": True,
    }
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=2)

    # 4) README card
    card = {"title": title, "emoji": "🎯", "colorFrom": "blue", "colorTo": "green",
            "sdk": "static", "pinned": False, "tags": meta["tags"]}
    readme = "---\n" + "\n".join(f"{k}: {json.dumps(v)}" for k, v in card.items()) + "\n---\n"
    open(os.path.join(lbdir, "README.md"), "w").write(readme)
    print(f"built {out}")
    print(f"  space_id={space_id}")
    print(f"  orid={orid} arxiv={arxiv} pages={len(pages)}")
    return space_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", default=DEFAULT_REF)
    a = ap.parse_args()
    build(a.dir, a.ref)
