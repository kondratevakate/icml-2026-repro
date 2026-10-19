#!/usr/bin/env python3
"""Build the static Trackio logbook for the GLEAN release audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "audit.json"
SPACE_ID = "kondratevakate/repro-glean-release-audit"
TITLE = (
    "Reproduction: Guideline-Grounded Evidence Accumulation "
    "for High-Stakes Agent Verification"
)
TITLES = {
    "C1": "Claim 1: evidence accumulation and calibration",
    "C2": "Claim 2: data and trajectory construction",
    "C3": "Claim 3: main verification performance",
    "C4": "Claim 4: active verification and ablations",
    "C5": "Claim 5: Best-of-N and computational efficiency",
    "C6": "Claim 6: clinician utility study",
}


def create_logbook() -> None:
    old_cwd, old_find = Path.cwd(), lb.find_project_dir
    try:
        os.chdir(CANDIDATE)
        lb.find_project_dir = lambda *args, **kwargs: None
        created = Path(lb.create_logbook(title=TITLE, space_id=SPACE_ID))
    finally:
        lb.find_project_dir = old_find
        os.chdir(old_cwd)
    if created.resolve() != PROJECT.resolve():
        raise RuntimeError(f"Unexpected project: {created}")


def add_page(
    project: Path, slug: str, title: str, markdown: str, evidence: object
) -> None:
    lb.add_markdown_cell(project, slug, markdown, title=title)
    lb.add_code_cell(
        project,
        slug,
        output=json.dumps(evidence, ensure_ascii=False, indent=2),
        title=f"{title} evidence",
        code_paths=["repro_glean/audit_glean.py"],
        language="python",
    )


def main() -> int:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-FP23eFYhAy", "clinical-verification"],
            "paper": {"arxiv_id": "2603.02798"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive = lb.ensure_page(PROJECT, "Executive summary")
    pages = {key: lb.ensure_page(PROJECT, title) for key, title in TITLES.items()}
    conclusion = lb.ensure_page(PROJECT, "Conclusion")

    clinical = data["clinical_dataset"]
    proof = data["proof_audit"]
    consistency = data["prose_table_consistency"]
    headline = data["headline_arithmetic"]
    executive_text = f"""**Outcome.** The official GLEAN release receives
**{data["prepared_score"]}/{data["score_denominator"]} points**, equal to the
frozen forecast. Both upstream datasets are accessible, but the author
released no GLEAN code, trajectories, predictions, calibration rows, active
traces, or clinician annotations.

| Independent check | Result |
| --- | --- |
| MIMIC-IV-Ext payload checksums | {clinical["official_checksums"]["files_checked"]}/{clinical["official_checksums"]["files_checked"]} match |
| MIMIC-IV-Ext cases | {clinical["pathology_total"]} |
| Discounted accumulation recurrence | {data["accumulation_check"]["absolute_difference"]:.1e} difference |
| Printed Appendix A bound follows | {proof["printed_term_follows_from_assumption"]} |
| Qwen3 diverticulitis prose/Table-1 values conflict | {consistency["all_values_conflict"]} |
| Prepared score | {data["prepared_score"]}/{data["score_denominator"]} |

Primary sources: [arXiv](https://arxiv.org/abs/2603.02798),
[OpenReview](https://openreview.net/forum?id=FP23eFYhAy),
[guideline corpus](https://huggingface.co/datasets/epfl-llm/guidelines), and
[MIMIC-IV-Ext CDM](https://physionet.org/content/mimic-iv-ext-cdm/1.1/).
No leaderboard or third-party verdict contributes evidence. Credentialed
clinical rows are excluded from this Space.
"""
    lb.add_markdown_cell(
        PROJECT, executive, executive_text, title="Executive summary"
    )
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)
    poster = f"""<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #c8d2dc;padding:20px;background:#fff;color:#18222d"><h2 style="border-bottom:3px solid #2563a6;padding-bottom:8px">GLEAN: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Evidence layer</th><th>Result</th></tr><tr><td>Source datasets</td><td>Guidelines metadata and credentialed 2,400-case CDM release verified</td></tr><tr><td>Method</td><td>Accumulation equation executable; Appendix bound has an epsilon-squared error</td></tr><tr><td>Main table</td><td>Not reproducible; prose conflicts with Table 1 in all five quoted example values</td></tr><tr><td>Outputs/study</td><td>No trajectories, predictions, active traces, Best-of-N groups, or clinician ratings</td></tr></table><p><b>Prepared {data["prepared_score"]}/{data["score_denominator"]}</b>. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(
        PROJECT, executive, html=poster, title="Reproduction poster"
    )
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    add_page(
        PROJECT,
        pages["C1"],
        TITLES["C1"],
        f"""**PARTIAL - 1/2.** A synthetic check reproduces the discounted
logit accumulation recurrence to `{data["accumulation_check"]["absolute_difference"]:.1e}`.
The Appendix A proof is invalid as written: its assumption contributes
`2 * epsilon_suff`, but the displayed bound uses
`2 * epsilon_suff^2`. The released paper has no executable implementation.""",
        {
            "accumulation": data["accumulation_check"],
            "proof": proof,
        },
    )
    add_page(
        PROJECT,
        pages["C2"],
        TITLES["C2"],
        f"""**PARTIAL - 1/2.** All official CDM v1.1 payload checksums match
and the pathology map has {clinical["pathology_total"]} cases, including all
three GLEAN diseases. The release has
{clinical["radiology"]["rows"]} radiology rows, one more than the official-page
summary. The paper-specific 4,000 trajectories, balancing decisions, labels,
and retrieved-guideline assignments are absent.""",
        {
            "clinical_dataset": clinical,
            "guidelines": data["guideline_dataset"],
            "notes_availability": data["mimic_iv_note"],
        },
    )
    add_page(
        PROJECT,
        pages["C3"],
        TITLES["C3"],
        f"""**UNSUPPORTED - 0/2.** The approximate headline is recoverable
only as table arithmetic: relative to Self-Consistency averaged across six
cells, AUROC rises by
{100 * headline["versus_self_consistency"]["auroc_relative_gain"]:.2f}% and
Brier falls by
{100 * headline["versus_self_consistency"]["brier_relative_reduction"]:.2f}%.
No row-level predictions are released. Moreover, all five Qwen3
diverticulitis values quoted in the Main Results prose conflict with Table 1.""",
        {
            "headline": headline,
            "prose_table_consistency": consistency,
        },
    )
    add_page(
        PROJECT,
        pages["C4"],
        TITLES["C4"],
        """**UNSUPPORTED - 0/2.** The paper displays active-verification and
component tables, but releases no triggered-case identities, added guidelines,
judge scores, or per-ablation predictions. The tables cannot be independently
regenerated.""",
        {"limits": data["limits"]},
    )
    add_page(
        PROJECT,
        pages["C5"],
        TITLES["C5"],
        """**UNSUPPORTED - 0/2.** Best-of-N accuracy and model-call/token
totals are display-only. Candidate groups, verifier rankings, selected
answers, API traces, and token logs are not released.""",
        {"limits": data["limits"]},
    )
    add_page(
        PROJECT,
        pages["C6"],
        TITLES["C6"],
        """**UNSUPPORTED - 0/2.** Only aggregate clinician-study statistics
and two short qualitative excerpts are provided. There are no anonymized
ratings, error-step annotations, study instrument, or analysis code.""",
        {"limits": data["limits"]},
    )
    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        f"""The release supports **{data["prepared_score"]}/12** points. The
two source datasets and method equations are inspectable, and the audit finds
both a proof error and a prose/table inconsistency. The decisive boundary is
paper-specific evidence: no generated trajectories, verifier outputs,
calibration split, active traces, Best-of-N groups, or clinician annotations
are available. Therefore the empirical claims are not independently
reproducible from the official release.""",
        title="Conclusion",
    )

    root = PROJECT / "logbook" / "pages"
    lines = [
        f"# {TITLE}",
        "",
        lb.TOC_HEADING,
        "",
        lb.TOC_HEADER,
        lb.TOC_SEP,
        f"| [Executive summary](#/{executive}) |",
    ]
    lines += [f"| [{TITLES[key]}](#/{slug}) |" for key, slug in pages.items()]
    lines += [f"| [Conclusion](#/{conclusion}) |", ""]
    (root / "index.md").write_text("\n".join(lines), encoding="utf-8")
    lb.write_site_files(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

