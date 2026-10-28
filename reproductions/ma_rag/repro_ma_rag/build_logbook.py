#!/usr/bin/env python3
"""Build the Trackio logbook from frozen MA-RAG evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "artifact_audit.json"
SPACE_ID = "kondratevakate/repro-ma-rag-release-audit"
TITLE = "Reproduction: MA-RAG release and benchmark audit"
PAGE_TITLES = {
    "C1": "Claim 1: released multi-round architecture",
    "C2": "Claim 2: seven medical QA benchmarks",
    "C3": "Claim 3: 62.2 average and 6.8-point gain",
    "C4": "Claim 4: baseline superiority",
    "C5": "Claim 5: component ablations",
    "C6": "Claim 6: robustness and efficiency",
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
    architecture = evidence["architecture"]
    released = evidence["released_datasets"]
    tables = evidence["paper_table_arithmetic"]
    create_logbook()

    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-S7lpdz7NAW", "medical-qa"],
            "paper": {"arxiv_id": "2603.03292"},
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

    summary = f"""**Outcome.** The official MA-RAG release receives
**{evidence["prepared_score"]}/{evidence["score_denominator"]} points**, below
the frozen 4/12 expectation but inside its 2–4 range. The multi-round loop and
all seven declared benchmark subsets are present. Full credit is prevented by
two independently identified release/specification inconsistencies: history
ranking conflicts with Equation 4, and the single-answer runner cannot score
41 released NEJM multi-answer rows.

| Item | Result |
| --- | --- |
| Official code commit | `423f031cc0ffca679a047c885be999b9604100a9` |
| Released benchmarks / rows | {released["dataset_count"]} / {released["total_rows"]:,} |
| NEJM multi-answer rows incompatible with runner | {released["single_answer_runner_incompatible_rows"]} |
| Mutation/unit tests | 4 passing |
| Full model/RAG rerun | blocked by absent corpus, checkpoint, predictions and services |
| Prepared score | {evidence["prepared_score"]}/{evidence["score_denominator"]} |

Primary sources: [arXiv](https://arxiv.org/abs/2603.03292),
[OpenReview](https://openreview.net/forum?id=S7lpdz7NAW), and the
[official repository](https://github.com/NJU-RL/MA-RAG). No leaderboard or
third-party reproduction verdict contributes evidence.
"""
    lb.add_markdown_cell(PROJECT, executive, summary, title="Executive summary")
    summary_id = lb.last_cell_id(PROJECT, page=executive)
    if summary_id:
        lb.set_cell_pinned(PROJECT, summary_id, pinned=True, page=executive)

    poster = f"""<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #c9d3df;padding:20px;background:#fff;color:#17212b">
<h2 style="margin:0 0 6px;border-bottom:3px solid #5b4aa5;padding-bottom:8px">MA-RAG: independent release audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | S7lpdz7NAW | arXiv 2603.03292v3 | static/CPU audit</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#f2f0fa"><th style="padding:6px;text-align:left">Claim</th><th>Evidence</th><th>Verdict</th></tr>
<tr><td style="padding:6px">C1 architecture</td><td>Loop present; Equation 4/code and epsilon-gate mismatch</td><td style="color:#9a6500;font-weight:700">PARTIAL 1/2</td></tr>
<tr><td style="padding:6px">C2 benchmarks</td><td>7,852 rows; 41 NEJM targets incompatible with runner</td><td style="color:#9a6500;font-weight:700">PARTIAL 1/2</td></tr>
<tr><td style="padding:6px">C3-C6 experiments</td><td>Corpus, predictions, evaluator and services absent</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table><p style="font-size:12px;margin:12px 0 0">Prepared {evidence["prepared_score"]}/{evidence["score_denominator"]}. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(
        PROJECT, executive, html=poster, title="Reproduction poster"
    )
    poster_id = lb.last_cell_id(PROJECT, page=executive)
    if poster_id:
        lb.set_cell_pinned(PROJECT, poster_id, pinned=True, page=executive)

    c1 = f"""**PARTIAL — 1/2.**

Both released branches implement the multi-round solver, conflict-query
retrieval, history construction, unanimity early stop, and final majority
vote. All core-loop static checks pass.

However, Equation 4 requires descending quality. The evaluator code applies
ascending `argsort(scores)`, and the entropy branch places highest entropy
first despite defining quality as negative entropy. Figure 2 visually orders
Lower → Medium → Higher and therefore matches the code while contradicting the
equation and prose. The paper also describes an epsilon confidence gate, but
the release exposes only unanimity-or-max-round stopping.
"""
    lb.add_markdown_cell(PROJECT, claims["C1"], c1, title=PAGE_TITLES["C1"])
    lb.add_code_cell(
        PROJECT,
        claims["C1"],
        output=json.dumps(architecture, indent=2, sort_keys=True),
        title="C1 machine-readable evidence",
        code_paths=["repro_ma_rag/audit_artifacts.py"],
        language="python",
    )

    counts = {
        name: item["rows"] for name, item in released["datasets"].items()
    }
    c2 = f"""**PARTIAL — 1/2.**

All seven paper-declared sizes match exactly and sum to
{released["total_rows"]:,} rows. IDs are unique and option/answer schemas are
valid when NEJM multi-answer labels are interpreted as sets.

The canonical runners nevertheless force `single_answer=True`. NEJM contains
{released["single_answer_runner_incompatible_rows"]} multi-letter targets, so
those rows cannot be scored correctly by the released path. The audit also
finds within-dataset duplicate/conflicting normalized questions and two exact
cross-benchmark overlaps; these are reported rather than silently removed.

Released counts: `{json.dumps(counts, sort_keys=True)}`.
"""
    lb.add_markdown_cell(PROJECT, claims["C2"], c2, title=PAGE_TITLES["C2"])
    lb.add_code_cell(
        PROJECT,
        claims["C2"],
        output=json.dumps(released, indent=2, sort_keys=True),
        title="C2 machine-readable evidence",
        code_paths=[
            "repro_ma_rag/audit_artifacts.py",
            "repro_ma_rag/test_audit_artifacts.py",
        ],
        language="python",
    )

    c3 = f"""**INCONCLUSIVE, NOT RERUN — 0/2.**

Paper-table arithmetic reproduces 55.4 for Qwen3-8B and 62.2 for MA-RAG-ext,
with an unrounded difference of
`{tables["ma_rag_minus_qwen_unrounded"]:.4f}` points. This is source
concordance only. The model, corpus/index, evaluator checkpoint, credentials,
and canonical predictions needed for fresh accuracy are absent.
"""
    lb.add_markdown_cell(PROJECT, claims["C3"], c3, title=PAGE_TITLES["C3"])
    lb.add_markdown_cell(
        PROJECT,
        claims["C4"],
        f"""**INCONCLUSIVE, NOT RERUN — 0/2.**

The printed MA-RAG-ext minus TC-RAG difference recomputes to
`{tables["ma_rag_minus_tc_rag_unrounded"]:.4f}` points. Matched baseline
predictions and complete environments are absent, so no superiority result is
claimed.
""",
        title=PAGE_TITLES["C4"],
    )
    lb.add_markdown_cell(
        PROJECT,
        claims["C5"],
        """**INCONCLUSIVE, NOT RERUN — 0/2.**

The repository lacks canonical ablation outputs, evaluator training CSVs, and
the fine-tuned evaluator checkpoint. Printed cumulative values are not treated
as reproduced component effects.
""",
        title=PAGE_TITLES["C5"],
    )
    lb.add_markdown_cell(
        PROJECT,
        claims["C6"],
        """**INCONCLUSIVE, NOT RERUN — 0/2.**

Four matched stochastic runs and hardware-level timing traces are absent.
Rendered paper tables anchor the claim but cannot establish variance or
efficiency independently.
""",
        title=PAGE_TITLES["C6"],
    )

    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        f"""The official release supports **{evidence["prepared_score"]}/12**
points: one point for a substantial but specification-inconsistent multi-round
implementation, and one point for seven complete-size benchmark subsets whose
NEJM multi-answer rows are incompatible with the released single-answer judge.

The frozen forecast expected 4/12. Independent checks lowered it to 2/12
without using peer verdicts. A paper-level rerun remains blocked by the absent
MedCorp index, Qwen serving environment, evaluator data/checkpoint, canonical
predictions, seeds, and timing traces.
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
