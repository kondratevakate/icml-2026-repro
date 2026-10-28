#!/usr/bin/env python3
"""Build the Trackio logbook from frozen SupGCL audit evidence."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "claims_audit.json"
SPACE_ID = "kondratevakate/repro-supervised-graph-contrastive-learning-grn"
TITLE = "Reproduction: Supervised Graph Contrastive Learning for Gene Regulatory Networks"
PAGES = {
    "C1": "Claim 1: Theorem 1 and LINCS supervision",
    "C2": "Claim 2: Corollary 1 temperature limit",
    "C3": "Claim 3: node-level TCGA performance",
    "C4": "Claim 4: graph-level TCGA performance",
    "C5": "Claim 5: superiority across 13 tasks",
    "C6": "Claim 6: embedding-space analysis",
}


def create_logbook() -> None:
    if PROJECT.exists():
        raise RuntimeError(f"Logbook already exists: {PROJECT}")
    original_cwd = Path.cwd()
    original_find = lb.find_project_dir
    try:
        os.chdir(CANDIDATE)
        lb.find_project_dir = lambda *args, **kwargs: None
        created = Path(lb.create_logbook(title=TITLE, space_id=SPACE_ID))
    finally:
        lb.find_project_dir = original_find
        os.chdir(original_cwd)
    if created.resolve() != PROJECT.resolve():
        raise RuntimeError(f"Unexpected logbook path: {created}")


def claim_text(claim: dict) -> str:
    cid = claim["id"]
    if cid == "C1":
        return f"""**Verdict - VERIFIED (2/2).**

Theorem 1 was reconstructed from independent finite probability tables.
Across `{claim["independent_probability_tables"]}` seeded tables, direct joint
KL and the chain-rule decomposition agreed to maximum absolute error
`{claim["max_chain_rule_absolute_error"]:.3g}`.

Pinned author source independently confirms that LINCS knockdown graphs and
the knockdown-gene metadata feed `TeacherSampler`. This verifies the stated
mathematical decomposition and data mechanism, not downstream performance.
"""
    if cid == "C2":
        return f"""**Verdict - VERIFIED (2/2).**

Across `{claim["independent_probability_tables"]}` independent similarity and
node-loss tables, increasing `tau_a` made both augmentation distributions
uniform with zero monotonicity failures. At `tau_a=100000`, the worst
objective distance to the uniformly averaged node loss was
`{claim["max_final_distance_to_node_limit"]:.3g}` and the worst augmentation
KL was `{claim["max_final_augmentation_kl"]:.3g}`.

The analytic softmax limit independently establishes the exact
`tau_a -> infinity` result stated by Corollary 1.
"""
    return f"""**Verdict - NOT EXECUTED (0/2).**

{claim["reason"]} Paper tables and author-reported values are not treated as
independent evidence.
"""


def copy_bundle() -> None:
    destination = PROJECT / "logbook" / "repro_supgcl"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        HERE,
        destination,
        ignore=shutil.ignore_patterns(
            "build_logbook.py", "__pycache__", "*.pyc", ".pytest_cache"
        ),
    )


def write_hub_readme() -> None:
    content = """---
title: "Reproduction: Supervised Graph Contrastive Learning for Gene Regulatory Networks"
emoji: "🧬"
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
tags:
 - trackio
 - trackio-logbook
 - open-experiment
 - icml2026-repro
 - paper-0Y7itV1kb7
 - arxiv:2505.17786
---

# SupGCL reproduction

Independent CPU audit of Theorem 1 and Corollary 1. Prepared score: **4/12**.

The runnable clean-room bundle and frozen evidence are included in
[`repro_supgcl/`](./repro_supgcl/README.md).
"""
    (PROJECT / "logbook" / "README.md").write_text(content, encoding="utf-8")


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": [
                "icml2026-repro",
                "paper-0Y7itV1kb7",
                "arxiv:2505.17786",
            ],
            "paper": {"arxiv_id": "2505.17786"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive_slug = lb.ensure_page(PROJECT, "Executive summary")
    claim_slugs = {
        cid: lb.ensure_page(PROJECT, title) for cid, title in PAGES.items()
    }
    conclusion_slug = lb.ensure_page(PROJECT, "Conclusion")

    executive = f"""**Outcome.** This independent CPU/source audit prepares
**{evidence["prepared_score"]}/{evidence["maximum_score"]}**. Theorem 1's KL
decomposition and Corollary 1's unsupervised-GCL limit are verified. The
TCGA/LINCS empirical claims remain unexecuted.

| Item | Value |
| --- | --- |
| Prepared score | 4/12 |
| GPU | None |
| Private medical data | None |
| Independent tests | 5 passing |
| Author commit | `{evidence["provenance"]["official_code_commit"]}` |

The full empirical path requires the official 4.3 GB graph archive, H100-class
pretraining, and repeated cross-validation. No leaderboard or peer
reproduction was inspected or used.

Primary sources: [arXiv](https://arxiv.org/abs/2505.17786),
[OpenReview](https://openreview.net/forum?id=0Y7itV1kb7),
[official code](https://github.com/shobioinfo/SupGCL), and
[official data](https://zenodo.org/records/15496012).
"""
    lb.add_markdown_cell(
        PROJECT, executive_slug, executive, title="Executive summary"
    )
    cell_id = lb.last_cell_id(PROJECT, page=executive_slug)
    lb.set_cell_pinned(PROJECT, cell_id, pinned=True, page=executive_slug)

    poster = """<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #c9d2d8;padding:20px;background:#fff;color:#17212b">
<h2 style="font-size:18px;margin:0 0 8px;border-bottom:3px solid #176b5b;padding-bottom:8px">SupGCL: independent theorem audit</h2>
<p style="font-size:12px">ICML 2026 | 0Y7itV1kb7 | arXiv 2505.17786 | Prepared 4/12</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#edf3f1"><th style="text-align:left;padding:6px">Evidence</th><th>Finding</th><th>Verdict</th></tr>
<tr><td style="padding:6px">Theorem 1</td><td>100 independent KL chain-rule probes</td><td style="color:#176b5b;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">Corollary 1</td><td>25 temperature-limit probes plus analytic limit</td><td style="color:#176b5b;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">Tables 2-3</td><td>4.3 GB data and long GPU runs required</td><td>NOT EXECUTED</td></tr>
<tr><td style="padding:6px">Embedding analysis</td><td>Pretrained checkpoints not released</td><td>NOT EXECUTED</td></tr>
</table></div>"""
    lb.add_figure_cell(
        PROJECT, executive_slug, html=poster, title="Reproduction poster"
    )
    cell_id = lb.last_cell_id(PROJECT, page=executive_slug)
    lb.set_cell_pinned(PROJECT, cell_id, pinned=True, page=executive_slug)

    for cid, slug in claim_slugs.items():
        claim = next(item for item in evidence["claims"] if item["id"] == cid)
        lb.add_markdown_cell(
            PROJECT, slug, claim_text(claim), title=PAGES[cid]
        )
        lb.add_code_cell(
            PROJECT,
            slug,
            output=json.dumps(claim, indent=2, sort_keys=True),
            title=f"{cid} machine-readable evidence",
            code_paths=["repro_supgcl/audit_claims.py"],
            language="python",
        )

    conclusion = """The two mathematical claims are independently verified
without relying on paper-rendered metrics. The pinned release also establishes
the claimed connection to LINCS knockdown supervision.

This result is deliberately limited: the released training code uses
subsampling and a stochastic objective, and this audit does not claim that it
is an exact unbiased estimator of the full loss. The four empirical claims
remain unexecuted until the official data and suitable GPU compute are
available.
"""
    lb.add_markdown_cell(
        PROJECT, conclusion_slug, conclusion, title="Conclusion"
    )

    index = [
        f"# {TITLE}",
        "",
        lb.TOC_HEADING,
        "",
        lb.TOC_HEADER,
        lb.TOC_SEP,
        f"| [Executive summary](#/{executive_slug}) |",
    ]
    index.extend(
        f"| [{PAGES[cid]}](#/{claim_slugs[cid]}) |" for cid in PAGES
    )
    index.extend([f"| [Conclusion](#/{conclusion_slug}) |", ""])
    (PROJECT / "logbook" / "pages" / "index.md").write_text(
        "\n".join(index), encoding="utf-8"
    )
    copy_bundle()
    write_hub_readme()
    lb.write_site_files(PROJECT)
    print(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
