#!/usr/bin/env python3
"""Build a Trackio logbook from frozen SP-Mind evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
SPACE_ID = "kondratevakate/repro-spmind-artifact-audit"
TITLE = "Reproduction: SP-Mind: An Autonomous Reasoning Agent for Spatial Proteomics Analysis"
PAGES = {
    "C1": "Claim 1: autonomous agent architecture",
    "C2": "Claim 2: SP-Bench execution accuracy",
    "C3": "Claim 3: SP-Bench construction",
    "C4": "Claim 4: CRC-CODEX quantification",
    "C5": "Claim 5: cell-annotation similarity",
}


def create_logbook() -> None:
    if PROJECT.exists():
        for page in lb.list_pages(PROJECT):
            page_path = PROJECT / "logbook" / page["file"]
            cells = lb._parse_cells_from_text(
                page_path.read_text(encoding="utf-8"),
                page["slug"],
                page["title"],
            )
            for cell in cells:
                lb.remove_cell(PROJECT, cell["id"], page=page["slug"])
        return
    old_cwd = Path.cwd()
    old_find = lb.find_project_dir
    try:
        os.chdir(CANDIDATE)
        lb.find_project_dir = lambda *args, **kwargs: None
        created = Path(lb.create_logbook(title=TITLE, space_id=SPACE_ID))
    finally:
        lb.find_project_dir = old_find
        os.chdir(old_cwd)
    if created.resolve() != PROJECT.resolve():
        raise RuntimeError(f"Unexpected logbook path: {created}")


def claim_body(cid: str, audit: dict, annotation: dict) -> str:
    benchmark = audit["benchmark"]
    tools = audit["tools"]
    if cid == "C1":
        return f"""**Verdict - PARTIALLY VERIFIED FOR THE RELEASED ARCHITECTURE.**

An AST-based audit finds `{tools["registered_tool_count"]}` unique registered
tools, `{tools["skill_document_count"]}` skill documents, and implementations
for all eight benchmark stages. Every registered description resolves to a
top-level implementation. The agent invokes the Claude Agent SDK streaming
loop with a system prompt requiring stepwise data inspection and tool
execution.

This verifies the released wiring, not the full autonomous end-to-end claim:
the actual analysis pipeline was not executed, the iterative loop is delegated
to Claude Agent SDK, and the configured timeout is not enforced. The paper's
priority word "first" is not evaluated.
"""
    if cid == "C2":
        return """**Verdict - INCONCLUSIVE, NOT EXECUTED.**

Reproducing 68.9% overall accuracy requires three executions of 102 tasks with
`claude-sonnet-4-20250514`. That model is retired, the 306 original traces and
predictions are not released, and the repository provides no automatic
implementation of the paper's success judge. Paper-table concordance is not
treated as reproduction evidence.
"""
    if cid == "C3":
        return f"""**Verdict - VERIFIED.**

An independent JSONL parser finds `{benchmark["record_count"]}` records,
`{benchmark["unique_ids"]}` unique IDs and
`{benchmark["unique_normalized_queries"]}` unique normalized queries. It
recomputes `{benchmark["category_count"]}` categories and
`{benchmark["stage_count"]}` stages.

Tier counts are `{json.dumps(benchmark["tier_counts"], sort_keys=True)}` and
match the declared Basic/Intermediate/Advanced/Challenging split. Schema,
enumeration, stage-count, placeholder, and tier-complexity mutation tests pass.
"""
    if cid == "C4":
        return """**Verdict - INCONCLUSIVE, MISSING INPUTS AND OUTPUTS.**

The released evaluator implements the five metric families, but the official
dataset does not include the CRC-CODEX evaluation inputs, generated
quantification outputs, or complete ground-truth bundle needed to recompute
the reported 0.953/0.902/0.727/0.368/0.513 values.
"""
    duplicate_rows = annotation["files"][0]["duplicate_cell_id_rows"]
    return f"""**Verdict - INCONCLUSIVE, MISSING PREDICTIONS.**

Four ground-truth files pass structural schema checks, covering
`{annotation["total_rows"]:,}` rows. Original agent predictions are absent, so
the reported average GHK similarity 0.681 cannot be recomputed.

The released evaluator has an order-sensitive equal-length fast path: a
mutation containing the same correctly labeled cells in reversed row order
scores near zero instead of one. One published ground-truth file also contains
`{duplicate_rows:,}` rows with repeated `cellLabel` values. These are
reproducibility qualifications, not evidence that the reported score itself is
false.
"""


def main() -> int:
    audit = json.loads((HERE / "evidence" / "audit_results.json").read_text(encoding="utf-8"))
    annotation = json.loads(
        (HERE / "evidence" / "annotation_ground_truth_audit.json").read_text(encoding="utf-8")
    )
    claims = json.loads((HERE / "claims.json").read_text(encoding="utf-8"))
    create_logbook()

    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-UJcB3XrffF"],
            "paper": {"arxiv_id": "2606.24235"},
            "private": False,
            "repos_public": False,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive_slug = lb.ensure_page(PROJECT, "Executive summary")
    claim_slugs = {cid: lb.ensure_page(PROJECT, title) for cid, title in PAGES.items()}
    conclusion_slug = lb.ensure_page(PROJECT, "Conclusion")

    executive = """**Outcome.** Five SP-Mind claims were audited without
leaderboard or peer-reproduction evidence. The released agent wiring is
**PARTIALLY VERIFIED**, and the complete SP-Bench manifest is **VERIFIED**.
Three numerical performance claims remain **INCONCLUSIVE** because the
original model, predictions, traces, or evaluation inputs are unavailable.

## Scope & cost

| Item | Value |
| --- | --- |
| Prepared score | 3/10 |
| GPU | None |
| LLM/API calls | None |
| Full dataset download | Not required |
| Unit tests | 8 passing |
| Author commit | `d5b889c3649fbdfd1489e5b2f60fc52a0d3ddbc6` |

Provenance: [arXiv](https://arxiv.org/abs/2606.24235),
[author repository](https://github.com/tomtommyyuan/spmind), and the official
[dataset](https://huggingface.co/datasets/tomyuanyucheng/spmind).
"""
    lb.add_markdown_cell(PROJECT, executive_slug, executive, title="Executive summary")
    cell_id = lb.last_cell_id(PROJECT, page=executive_slug)
    lb.set_cell_pinned(PROJECT, cell_id, pinned=True, page=executive_slug)

    poster = """<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ccd5dd;padding:20px;background:#fff;color:#18212a">
<h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #176b87;padding-bottom:8px">SP-Mind: independent artifact audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | UJcB3XrffF | arXiv 2606.24235 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef3f6"><th style="padding:6px;text-align:left">Claim</th><th>Independent evidence</th><th>Verdict</th></tr>
<tr><td style="padding:6px">Agent architecture</td><td>Wiring verified; end-to-end execution not run</td><td style="color:#7a5c13;font-weight:700">PARTIAL</td></tr>
<tr><td style="padding:6px">SP-Bench accuracy</td><td>Retired model and original traces unavailable</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
<tr><td style="padding:6px">Benchmark construction</td><td>102 unique tasks, 18 categories, 8 stages</td><td style="color:#137a46;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">CRC-CODEX</td><td>Required input/output bundle absent</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
<tr><td style="padding:6px">Cell annotation</td><td>Ground truth present; predictions absent</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table><p style="font-size:12px;margin:12px 0 0">Prepared 3/10. No peer reproductions used.</p></div>"""
    lb.add_figure_cell(PROJECT, executive_slug, html=poster, title="Reproduction poster")
    cell_id = lb.last_cell_id(PROJECT, page=executive_slug)
    lb.set_cell_pinned(PROJECT, cell_id, pinned=True, page=executive_slug)

    for claim in claims:
        cid = claim["id"]
        slug = claim_slugs[cid]
        lb.add_markdown_cell(PROJECT, slug, claim_body(cid, audit, annotation), title=PAGES[cid])
        lb.add_code_cell(
            PROJECT,
            slug,
            output=json.dumps(claim, indent=2, sort_keys=True),
            title=f"{cid} verdict record",
            code_paths=["repro_spmind/audit_artifacts.py"],
            language="python",
        )

    conclusion = """SP-Mind's released architecture and benchmark composition
are independently recoverable and internally consistent. The numerical agent
results are not reproducible from the public artifacts alone because essential
original traces, predictions, and some downstream evaluation inputs are
missing.

The annotation evaluator's row-order sensitivity should be corrected by
joining on an explicit globally unique cell identifier. It is recorded as a
qualification and does not inflate the prepared score.
"""
    lb.add_markdown_cell(PROJECT, conclusion_slug, conclusion, title="Conclusion")

    index = [
        f"# {TITLE}",
        "",
        lb.TOC_HEADING,
        "",
        lb.TOC_HEADER,
        lb.TOC_SEP,
        f"| [Executive summary](#/{executive_slug}) |",
    ]
    index.extend(f"| [{PAGES[cid]}](#/{claim_slugs[cid]}) |" for cid in PAGES)
    index.extend([f"| [Conclusion](#/{conclusion_slug}) |", ""])
    (PROJECT / "logbook" / "pages" / "index.md").write_text(
        "\n".join(index), encoding="utf-8"
    )
    lb.write_site_files(PROJECT)
    print(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
