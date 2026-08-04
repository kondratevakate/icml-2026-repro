#!/usr/bin/env python3
"""local_to_trackio.py — adapt a local reproduction run folder into a .trackio
logbook bundle ready for HF Spaces publish.

Input:  <dir>/  containing logbook.md, _score.json, input_bundle.json, _run_meta.json
Output: <dir>/.trackio/{metadata.json, logbook/{index.html,logbook.js,logbook.css,
        logbook.json, workspace.json, README.md, trackio-*.png, pages/<slug>/page.md}}

The trackio static files (index.html/js/css, *.png, workspace.json) are copied
RECURSIVELY from a reference space that already publishes correctly (default: a
full bundle downloaded from a known-good Space). This avoids hand-writing the
trackio runtime and guarantees logbook.js finds every asset it requires.

Pages are emitted as `pages/<slug>/page.md` with trackio-cell wrappers, and
logbook.json uses the schema_version=2 tree (root.children[]) that the runtime
expects.

Usage: python3 local_to_trackio.py <dir> [--ref <reference_dir_with_.trackio/logbook>]
"""
import json, os, re, shutil, sys, argparse

# Full working trackio bundle (assets + runtime) downloaded from a known-good Space.
DEFAULT_REF = "/tmp/trackio_ref/logbook"


def slugify(orid, title):
    s = re.sub(r"[^a-z0-9]+", "-", (title or orid).lower()).strip("-")
    return s[:60] or orid


def norm_orid(name):
    """Strip _arm1/_arm2b/... suffixes to recover the paper orid from a run dir name."""
    o = name
    for suf in ("_arm1", "_arm2b", "_arm2", "_arm3b", "_arm3", "_arm4b", "_arm4"):
        o = o.replace(suf, "")
    return o


def split_pages(logbook_md):
    """Split logbook.md into (title, slug, body_lines) per top-level '## Heading'.

    Each page = the heading line + every line until the next '## Heading'.
    The heading line itself is kept in the body (so the claim title shows);
    cell_wrap adds an H1 only when the body does not already start with one.
    """
    lines = logbook_md.splitlines()
    pages = []
    cur_title, cur_slug, cur_body = None, None, []
    for ln in lines:
        m = re.match(r"^##\s+(.*)$", ln)
        if m:
            if cur_title is not None:
                pages.append((cur_title, cur_slug, cur_body))
            cur_title = m.group(1).strip()
            cur_slug = re.sub(r"[^a-z0-9]+", "-", cur_title.lower()).strip("-")[:80] or "section"
            cur_body = [ln]
        else:
            if cur_title is not None:
                cur_body.append(ln)
    if cur_title is not None:
        pages.append((cur_title, cur_slug, cur_body))
    if not pages:
        pages = [("logbook", "logbook", lines)]
    return pages


def cell_wrap(title, body_lines):
    """Wrap markdown body in a trackio-cell marked block (matches working Spaces)."""
    body = "\n".join(body_lines).strip("\n")
    # If body already starts with a heading, don't prepend another # title.
    if not body.startswith("#"):
        body = f"# {title}\n\n" + body
    return f"{body}\n\n---\n<!-- trackio-cell\n{{\"type\": \"markdown\", \"id\": \"cell_{hash}\", \"created_at\": \"2026-01-01T00:00:00+00:00\", \"title\": {json.dumps(title)}}}\\n-->\n" \
        .replace("{hash}", str(abs(hash(title)) % 10**12))


def build(dir_path, ref_dir):
    dir_path = os.path.abspath(dir_path)
    lb = os.path.join(dir_path, "logbook.md")
    if not os.path.exists(lb):
        raise SystemExit(f"no logbook.md in {dir_path}")
    # Normalize v1 logbook into trackio-friendly format (one '## Claim N' page per
    # claim) WITHOUT changing any verdict or evidence. Content-preserving only.
    try:
        import rewrite_logbook_format as rlf
        src = open(lb).read()
        new_md, changed = rlf.rewrite(src)
        if changed:
            open(lb, "w").write(new_md)
            print(f"  [normalize] rewrote logbook.md into per-claim '## Claim N' sections (changed={changed})")
    except Exception as e:
        print(f"  [normalize] skipped: {e}")
    orid = norm_orid(os.path.basename(dir_path))
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
    # resolve space_id from space_map.json (authoritative orid->Space mapping) if present;
    # otherwise fall back to repro-<orid>. This guarantees we EDIT the correct existing
    # challenge Space (e.g. repro-globalhealthatlas-release-audit) rather than creating a
    # near-duplicate slug.
    space_id = None
    sm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "groundtruth", "space_map.json")
    try:
        sm = json.load(open(sm_path))
        if orid in sm:
            space_id = sm[orid]
    except Exception:
        pass
    if not space_id:
        slug = orid
        if len(slug) > 60:
            slug = slug[:60]
        space_id = f"kondratevakate/repro-{slug}"
    # derive slug from the final space_id so it is always defined
    slug = space_id.split("/")[-1].replace("repro-", "", 1) if space_id else orid
    out = os.path.join(dir_path, ".trackio")
    lbdir = os.path.join(out, "logbook")
    # clean previous build so stale/foreign pages (from an old ref copytree) cannot leak
    if os.path.isdir(lbdir):
        shutil.rmtree(lbdir)
    os.makedirs(os.path.join(lbdir, "pages"), exist_ok=True)

    # 1) copy trackio STATIC assets from ref (runtime + png + workspace.json + README).
    #    Do NOT copy ref/pages/ or ref/logbook.json — we generate those from the
    #    local logbook.md below (otherwise stale/foreign pages leak in).
    STATIC = ("index.html", "logbook.js", "logbook.css", "workspace.json",
              "README.md", "bucket-icon.svg", "style.css",
              "trackio-logo.png", "trackio-logo-light.png", "trackio-wordmark-dark.png")
    if os.path.isdir(ref_dir):
        for f in STATIC:
            src = os.path.join(ref_dir, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(lbdir, f))
    else:
        raise SystemExit(f"ref logbook dir missing: {ref_dir}")

    # 2) split logbook.md into pages/<slug>/page.md (trackio-cell format)
    pages = split_pages(open(lb).read())
    children = []
    index_lines = ["# Logbook pages", ""]
    for title_p, pslug, body in pages:
        pdir = os.path.join(lbdir, "pages", pslug)
        os.makedirs(pdir, exist_ok=True)
        open(os.path.join(pdir, "page.md"), "w").write(cell_wrap(title_p, body))
        index_lines.append(f"- [{title_p}](pages/{pslug}/page.md)")
        children.append({"slug": pslug, "title": title_p,
                         "file": f"pages/{pslug}/page.md", "children": []})
    open(os.path.join(lbdir, "pages", "index.md"), "w").write("\n".join(index_lines) + "\n")

    # 3) logbook.json (schema_version 2 tree the runtime expects)
    meta_t = {
        "schema_version": 2,
        "title": f"Reproduction: {title}",
        "emoji": "🎯",
        "space_id": space_id,
        "paper": {"arxiv_id": arxiv},
        "tags": ["icml2026-repro", f"paper-{orid}"],
        "updated_at": "2026-01-01T00:00:00+00:00",
        "root": {"slug": "index", "title": f"Reproduction: {title}",
                 "file": "pages/index.md", "children": children},
        "traces": [],
        "workspace": {"file": "workspace.json", "file_count": 0, "total_size": 0, "bucket_id": None},
        "agent_view_tokens": 0, "trace_view_tokens": 0, "workspace_view_tokens": 0,
        "revision": "local",
    }
    json.dump(meta_t, open(os.path.join(lbdir, "logbook.json"), "w"), indent=2)

    # 4) workspace.json (empty, as in working Spaces)
    json.dump({"schema_version": 1, "file_count": 0, "total_size": 0,
               "files": [], "hub_refs": [], "reference_only": True},
              open(os.path.join(lbdir, "workspace.json"), "w"), indent=2)

    # 5) metadata.json (used by publish_local.py)
    meta = {
        "space_id": space_id,
        "title": title,
        "emoji": "🎯",
        "created_at": "",
        "last_page": slug,
        "tags": ["icml2026-repro", f"paper-{orid}"],
        "paper": {"arxiv_id": arxiv},
        "private": False,
        "autosync": False,
    }
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=2)

    # 6) README card (root)
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
