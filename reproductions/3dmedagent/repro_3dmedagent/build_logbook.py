#!/usr/bin/env python3
"""Build the Trackio logbook from frozen 3DMedAgent evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "artifact_audit.json"
SPACE_ID = "kondratevakate/repro-3dmedagent-artifact-audit"
TITLE = "Reproduction: 3DMedAgent architecture and benchmark audit"
PAGE_TITLES = {
    "C1": "Claim 1: released OAMI-CFLT-T1S architecture",
    "C2": "Claim 2: DeepChestVQA composition",
    "C3": "Claim 3: forty-plus-task headline gain",
    "C4": "Claim 4: DeepChestVQA model comparison",
    "C5": "Claim 5: component ablations",
    "C6": "Claim 6: radiologist and turn analyses",
}


def create_logbook() -> None:
    if PROJECT.exists():
        raise RuntimeError(f"Logbook already exists: {PROJECT}")
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


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    benchmark = evidence["benchmark"]
    architecture = evidence["architecture"]
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-TH6pLxCOQ3", "medical-imaging"],
            "paper": {"arxiv_id": "2602.18064"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive = lb.ensure_page(PROJECT, "Executive summary")
    claims = {
        cid: lb.ensure_page(PROJECT, title)
        for cid, title in PAGE_TITLES.items()
    }
    conclusion = lb.ensure_page(PROJECT, "Conclusion")

    summary = f"""**Outcome.** Two release-level claims are independently
verified, preparing **{evidence["prepared_score"]}/{evidence["score_denominator"]}
points**, exactly matching the frozen forecast. The released code contains the
declared OAMI-CFLT-T1S routing, and the released DeepChestVQA CSV has the
claimed 892 scans, 1,020 QA pairs, and 17 subtypes. The clinical performance,
ablation, expert-agreement, and per-turn claims remain inconclusive.

| Item | Result |
| --- | --- |
| Official code commit | `3ee6a49cec9bfba58e0d16da8c05b6b16342139e` |
| DeepChestVQA rows / scans | {benchmark["row_count"]:,} / {benchmark["unique_image_ids"]:,} |
| Categories | 3/180 recognition; 8/480 visual; 6/360 medical |
| Unit/mutation tests | 4 passing |
| Full clinical rerun | blocked by absent volumes, derived artifacts and services |
| Prepared score | {evidence["prepared_score"]}/{evidence["score_denominator"]} |

Primary sources: [arXiv](https://arxiv.org/abs/2602.18064),
[OpenReview](https://openreview.net/forum?id=TH6pLxCOQ3), and the
[official repository](https://github.com/jinlab-imvr/3DMedAgent). No
leaderboard or third-party reproduction verdict contributes evidence.
"""
    lb.add_markdown_cell(PROJECT, executive, summary, title="Executive summary")
    summary_id = lb.last_cell_id(PROJECT, page=executive)
    if summary_id:
        lb.set_cell_pinned(PROJECT, summary_id, pinned=True, page=executive)

    poster = f"""<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #c7d4dc;padding:20px;background:#fff;color:#17212b">
<h2 style="margin:0 0 6px;border-bottom:3px solid #326b78;padding-bottom:8px">3DMedAgent: independent release audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | TH6pLxCOQ3 | arXiv 2602.18064v2 | CPU/static audit</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#edf5f6"><th style="padding:6px;text-align:left">Claim</th><th>Evidence</th><th>Verdict</th></tr>
<tr><td style="padding:6px">C1 architecture</td><td>Static routing, CLI, dry run, compileall</td><td style="color:#137a46;font-weight:700">VERIFIED RELEASE</td></tr>
<tr><td style="padding:6px">C2 benchmark</td><td>1,020 rows; 892 scans; 17 subtypes</td><td style="color:#137a46;font-weight:700">VERIFIED + CAVEATS</td></tr>
<tr><td style="padding:6px">C3-C6 experiments</td><td>Required volumes, outputs and traces absent</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table><p style="font-size:12px;margin:12px 0 0">Prepared {evidence["prepared_score"]}/{evidence["score_denominator"]}. Forecast calibrated. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(
        PROJECT, executive, html=poster, title="Reproduction poster"
    )
    poster_id = lb.last_cell_id(PROJECT, page=executive)
    if poster_id:
        lb.set_cell_pinned(PROJECT, poster_id, pinned=True, page=executive)

    c1 = f"""**VERIFIED AT RELEASE-ARCHITECTURE LEVEL — 2/2.**

All {len(architecture["checks"])} static checks pass: the official release
contains organ-aware memory construction, CT-CLIP global/detail/slice routes,
runtime-tool switches, and a T1S loop whose runtime bound is configurable.
The paper/README invocation sets the bound to
`{architecture["paper_readme_invocation"]}`. The CLI default is
`{architecture["t1s_cli_default"]}`, an important qualification.

The official CLI help and repository-wide compile check pass. A CPU dry run
selected one row from each recognition subtype, accepted the five-turn setting,
and completed with zero schema warnings. It made zero visual iterations because
the required reports, volumes, masks, embeddings and services are absent.
Therefore this is not an end-to-end clinical reproduction.
"""
    lb.add_markdown_cell(PROJECT, claims["C1"], c1, title=PAGE_TITLES["C1"])
    lb.add_code_cell(
        PROJECT,
        claims["C1"],
        output=json.dumps(architecture, indent=2, sort_keys=True),
        title="C1 machine-readable evidence",
        code_paths=[
            "repro_3dmedagent/audit_artifacts.py",
            "repro_3dmedagent/test_audit_artifacts.py",
        ],
        language="python",
    )

    c2 = f"""**VERIFIED WITH QUALITY CAVEATS — 2/2.**

Independent parsing confirms {benchmark["row_count"]:,} rows,
{benchmark["unique_question_ids"]:,} unique question IDs,
{benchmark["unique_image_ids"]:,} unique scans, and
{benchmark["subtype_count"]} subtypes. The three category totals exactly match
the paper: recognition 3/180, visual reasoning 8/480, and medical reasoning
6/360. Every subtype contains 60 rows.

The same audit finds {benchmark["repeated_scan_mcq_pair_rows"]} extra rows
across repeated scan/MCQ groups, {benchmark["conflicting_answer_group_count"]}
groups with conflicting correct options, and
{benchmark["option_only_mcq_count"]} option-only MCQ strings. All
{benchmark["row_count"]:,} rows carry the split label `train`. These are
release-quality caveats; they do not alter the composition counts.
"""
    lb.add_markdown_cell(PROJECT, claims["C2"], c2, title=PAGE_TITLES["C2"])
    lb.add_code_cell(
        PROJECT,
        claims["C2"],
        output=json.dumps(benchmark, indent=2, sort_keys=True),
        title="C2 machine-readable evidence",
        code_paths=["repro_3dmedagent/audit_artifacts.py"],
        language="python",
    )

    common = """**INCONCLUSIVE, NOT RERUN — 0/2.**

The compact official release does not contain the complete CT volumes, masks,
structured reports, intermediate CT-CLIP artifacts, canonical predictions,
and service credentials required for a matched evaluation. Paper tables are
source anchors, not reproduced measurements.
"""
    lb.add_markdown_cell(PROJECT, claims["C3"], common, title=PAGE_TITLES["C3"])
    lb.add_markdown_cell(PROJECT, claims["C4"], common, title=PAGE_TITLES["C4"])
    lb.add_markdown_cell(PROJECT, claims["C5"], common, title=PAGE_TITLES["C5"])
    lb.add_markdown_cell(
        PROJECT,
        claims["C6"],
        """**INCONCLUSIVE, NOT RERUN — 0/2.**

The radiologist labels, slice-ranking traces, and matched per-turn predictions
needed to recompute expert agreement and T1S turn curves are not included in
the compact release. The corresponding figures were visually anchored in the
paper but are not counted as reproduction evidence.
""",
        title=PAGE_TITLES["C6"],
    )

    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        f"""The released implementation and benchmark composition support
**{evidence["prepared_score"]}/{evidence["score_denominator"]}** points. The
forecast was calibrated exactly.

The main new release-level findings are a configuration qualification—the
paper's five T1S iterations require an explicit flag—and benchmark caveats:
repeated scan/MCQ pairs, two conflicting-answer groups, option-only MCQ
strings, and an all-`train` split column. None licenses a claim about clinical
performance.

A full rerun still requires the canonical medical volumes, masks, reports,
derived features, model/service configuration, original predictions, expert
annotations, and matched T1S traces.
""",
        title="Conclusion",
    )

    pages_root = PROJECT / "logbook" / "pages"
    lines = [
        f"# {TITLE}",
        "",
        lb.TOC_HEADING,
        "",
        lb.TOC_HEADER,
        lb.TOC_SEP,
        f"| [Executive summary](#/{executive}) |",
    ]
    lines.extend(
        f"| [{PAGE_TITLES[cid]}](#/{slug}) |" for cid, slug in claims.items()
    )
    lines.extend([f"| [Conclusion](#/{conclusion}) |", ""])
    (pages_root / "index.md").write_text("\n".join(lines), encoding="utf-8")
    lb.write_site_files(PROJECT)
    print(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
