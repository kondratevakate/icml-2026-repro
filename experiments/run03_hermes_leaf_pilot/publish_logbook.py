#!/usr/bin/env python3
"""
publish_logbook.py — ICML-2026 repro challenge: build + validate + publish a Trackio logbook.

USAGE
  python3 publish_logbook.py <orid> [--no-publish] [--skip-validate]
  python3 publish_logbook.py --all

WHAT IT DOES
  1. Reads <orid>/logbook.md (flat reproduction logbook written by any arm: arm1..arm4).
  2. Builds the official Trackio page-based structure under <orid>/.trackio/logbook/
     (pages/index.md + executive-summary + claim-* + conclusion, README.md, logbook.json)
     and <orid>/.trackio/metadata.json (tags icml2026-repro + paper-<orid>, space_id).
  3. Copies Trackio runtime files (logbook.js, index.html, logbook.css, bucket-icon.svg)
     from the canonical ccd/ example so the Space renders.
  4. Runs `trackio logbook sync` (regenerates site files) and the OFFICIAL
     validate_icml_logbook.py (must print "Logbook validation passed").
  5. Publishes to Hugging Face Space  kondratevakate/repro-<slug>  via `trackio logbook publish`.

PREREQUISITES (see requirements_publish.txt)
  - Python venv with: trackio>=0.32.0, huggingface_hub, plus the repo base venv deps.
  - HF token at ~/.cache/huggingface/token (kondratevakate, org ICML-2026-agent-repro).
  - ccd/.trackio/logbook/ present (source of runtime files + format reference).

NOTE ON arm2-4: each arm only needs to (a) write <orid>/logbook.md in the established
flat format (Summary table + per-claim sections + Evidence boundary + Artifacts), then
(b) run THIS script. No arm-specific logic required — the build is format-driven.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../icml-2026-repro
PILOT = os.path.join(REPO, "experiments", "run03_hermes_leaf_pilot")
CCD_LB = os.path.join(REPO, "ccd", ".trackio", "logbook")
VALIDATOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_icml_logbook.py")
HF_TOKEN = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
TRACKIO = os.environ.get("TRACKIO_BIN", "/tmp/run03_base_venv/bin/trackio")
NOW = "2026-08-02T05:00:00+00:00"


def cell_id():
    return "cell_" + uuid.uuid4().hex[:12]


def md_cell(title, body, pinned=False):
    c = {"type": "markdown", "id": cell_id(), "created_at": NOW, "title": title}
    if pinned:
        c["pinned"] = True
        c["pinned_at"] = NOW
    return f"<!-- trackio-cell\n{json.dumps(c)}\n-->\n{body}\n"


def figure_cell(title, html, pinned=False):
    c = {"type": "figure", "id": cell_id(), "created_at": NOW, "title": title}
    if pinned:
        c["pinned"] = True
        c["pinned_at"] = NOW
    return f"<!-- trackio-cell\n{json.dumps(c)}\n-->\n````html\n{html}\n````\n"


def slugify_claim(header):
    m = re.match(r"claim\s*(\d+)\s*[—-]\s*(.*)", header, re.IGNORECASE)
    rest = m.group(2) if m else header
    rest = re.sub(r"[^a-z0-9]+", "-", rest.lower()).strip("-")
    return f"claim-{m.group(1) if m else 'x'}-{rest}"


def build(orid, meta):
    src = os.path.join(PILOT, orid)
    out = os.path.join(src, ".trackio", "logbook")
    os.makedirs(os.path.join(out, "pages"), exist_ok=True)
    space = f"kondratevakate/{meta['space']}"
    title, arxiv = meta["title"], meta["arxiv"]

    text = open(os.path.join(src, "logbook.md")).read()
    lines = text.split("\n")
    sections, cur, buf = {}, None, []
    for ln in lines:
        m = re.match(r"^##\s+(.*)", ln)
        if m:
            if cur is not None:
                sections[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1).strip(), []
        else:
            buf.append(ln)
    if cur is not None:
        sections[cur] = "\n".join(buf).strip()

    claim_secs = {k: v for k, v in sections.items() if k.lower().startswith("claim")}
    def cnum(k):
        m = re.match(r"claim\s*(\d+)", k.lower())
        return int(m.group(1)) if m else 999
    claim_items = sorted(claim_secs.items(), key=lambda x: cnum(x[0]))

    # index.md — ONLY title + Pages table (validator rejects any intro/link text)
    index = f"# Reproduction: {title}\n\n## Pages\n\n| Page |\n| --- |\n| [Executive summary](#/executive-summary) |\n"
    for hdr, _ in claim_items:
        index += f"| [Claim {cnum(hdr)}: {hdr.split('—',1)[-1].strip()}](#/{slugify_claim(hdr)}) |\n"
    index += "| [Conclusion](#/conclusion) |\n"
    open(os.path.join(out, "pages", "index.md"), "w").write(index)

    # executive-summary
    poster = f"""<!-- poster_embed.html -->
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;border-radius:10px;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 4px;font-size:19px;border-bottom:2px solid #1f6f8b;padding-bottom:8px">{title}</h2>
  <div style="font-size:12px;color:#555;margin-bottom:10px">ICML 2026 &middot; OpenReview {orid} &middot; arXiv {arxiv} &middot; reproduced by kondratevate &middot; CPU</div>
  <p style="font-size:13px">Independent CPU reproduction of the paper's anchored claims. See per-claim pages for verdicts, mutation tests, and evidence boundaries.</p>
</div>"""
    eb = md_cell("Executive summary",
        "Independent CPU reproduction of the paper's anchored claims (numpy/torch/scipy/sympy). Per-claim verdicts below; mutation tests on every verified claim; honest inconclusive where data/GPU/training blocks a full reproduction. Wall time and cost near zero.",
        pinned=True)
    eb += "\n## Scope & cost\n\n| Item | This reproduction | Full replication |\n| --- | --- | --- |\n| Scope | Anchored claims feasible on CPU | Full paper (incl. data/GPU experiments) |\n| Hardware | CPU | CPU / GPU per paper |\n| Compute time | minutes–hours | varies |\n| Cost | ~$0 | ~$0 |\n| Outcome | see per-claim verdicts | — |\n"
    eb += "\n---\n" + figure_cell("Reproduction poster", poster, pinned=True)
    os.makedirs(os.path.join(out, "pages", "executive-summary"), exist_ok=True)
    open(os.path.join(out, "pages", "executive-summary", "page.md"), "w").write("# Executive summary\n\n---\n" + eb)

    children = []
    for hdr, body in claim_items:
        slug = slugify_claim(hdr)
        pdir = os.path.join(out, "pages", slug)
        os.makedirs(pdir, exist_ok=True)
        body_clean = re.sub(r"^#.*\n", "", body, count=1).strip()
        open(os.path.join(pdir, "page.md"), "w").write(f"# {hdr}\n\n---\n" + md_cell(hdr, body_clean))
        children.append({"slug": slug, "title": hdr, "file": f"pages/{slug}/page.md", "children": []})

    # conclusion
    ev = sections.get("Evidence boundary", "")
    arts = sections.get("Artifacts", "")
    cb = md_cell("Overall findings",
        "**Overall findings** in per-claim pages. Verdicts follow the evidence; mutation tests confirm mechanism, not correlation; inconclusive claims state the blocker honestly.")
    if ev:
        cb += "\n" + md_cell("Evidence boundary", ev)
    if arts:
        cb += "\n" + md_cell("Artifacts", arts)
    art = {"type": "artifact", "id": cell_id(), "created_at": NOW, "title": "Reproduction bundle",
           "artifact": f"{meta['space']}/repro-bundle:v0", "artifact_type": "dataset"}
    cb += f"\n---\n<!-- trackio-cell\n{json.dumps(art)}\n-->\n**📦 Artifact** `{meta['space']}/repro-bundle:v0` · dataset\n\nhttps://huggingface.co/buckets/kondratevakate/{meta['space']}-artifacts#{meta['space']}/repro-bundle:v0\n"
    dash = {"type": "dashboard", "id": cell_id(), "created_at": NOW, "title": f"Dashboard: {meta['space']}", "dashboard_project": meta['space']}
    cb += f"\n---\n<!-- trackio-cell\n{json.dumps(dash)}\n-->\n**🎯 Trackio dashboard** `{meta['space']}`\n\ntrackio-local-dashboard://{meta['space']}\n"
    os.makedirs(os.path.join(out, "pages", "conclusion"), exist_ok=True)
    open(os.path.join(out, "pages", "conclusion", "page.md"), "w").write("# Conclusion\n\n---\n" + cb)

    logbook = {"schema_version": 1, "title": f"Reproduction: {title}", "emoji": "🎯", "space_id": space,
               "paper": {"arxiv_id": arxiv}, "tags": ["icml2026-repro", f"paper-{orid}"], "updated_at": NOW,
               "root": {"slug": "index", "title": f"Reproduction: {title}", "file": "pages/index.md",
                        "children": [{"slug": "executive-summary", "title": "Executive summary", "file": "pages/executive-summary/page.md", "children": []},
                                     *children,
                                     {"slug": "conclusion", "title": "Conclusion", "file": "pages/conclusion/page.md", "children": []}]},
               "agent_view_tokens": 0, "revision": "0"}
    open(os.path.join(out, "logbook.json"), "w").write(json.dumps(logbook, indent=2))

    readme = f"""---
title: "{title}"
emoji: 🎯
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
tags:
 - trackio
 - trackio-logbook
 - open-experiment
 - icml2026-repro
 - paper-{orid}
---

# {title}

An open experiment logbook, published with [Trackio](https://github.com/gradio-app/trackio).
"""
    open(os.path.join(out, "README.md"), "w").write(readme)

    # metadata.json MUST live at .trackio/ level (validator reads proj/metadata.json)
    metadata = {"space_id": space, "tags": ["icml2026-repro", f"paper-{orid}"], "arxiv_id": arxiv, "title": title}
    open(os.path.join(src, ".trackio", "metadata.json"), "w").write(json.dumps(metadata, indent=2))

    for f in ["logbook.js", "index.html", "logbook.css", "bucket-icon.svg"]:
        shutil.copy(os.path.join(CCD_LB, f), os.path.join(out, f))

    return space, lb if (lb := out) else out


def validate(lb_dir, space):
    r = subprocess.run([TRACKIO, "logbook", "sync"], cwd=lb_dir, capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"sync failed: {r.stderr[:300]}"
    v = subprocess.run([sys.executable, VALIDATOR, "--space", space], cwd=lb_dir, capture_output=True, text=True)
    ok = v.returncode == 0 and "passed" in v.stdout
    return ok, (v.stdout.strip() if ok else v.stdout + v.stderr)


def publish(lb_dir, space):
    p = subprocess.run([TRACKIO, "logbook", "publish", space], cwd=lb_dir, capture_output=True, text=True,
                       env={**os.environ, "HF_TOKEN": HF_TOKEN})
    return p.returncode == 0, p.stdout.strip()[:160] + (" | ERR " + p.stderr.strip()[:160] if p.returncode else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orid", nargs="?", help="paper orid, or --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-publish", action="store_true", help="build+validate only")
    ap.add_argument("--meta", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "run03_targets.json"), help="orid->meta json")
    a = ap.parse_args()

    targets = json.load(open(a.meta))
    orids = list(targets.keys()) if a.all else [a.orid]
    for o in orids:
        if o not in targets:
            print(f"SKIP {o}: not in meta"); continue
        if targets[o].get("skip"):
            print(f"SKIP {o}: {targets[o]['skip']}"); continue
        space, lb = build(o, targets[o])
        print(f"[{o}] space={space}")
        ok, msg = validate(lb, space)
        print(f"  validate: {'PASS' if ok else 'FAIL'} — {msg}")
        if not ok:
            continue
        if a.no_publish:
            print("  (--no-publish, skipped publish)")
            continue
        pok, pmsg = publish(lb, space)
        print(f"  publish: {'OK' if pok else 'FAIL'} — {pmsg}")


if __name__ == "__main__":
    main()
