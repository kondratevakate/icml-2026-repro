#!/usr/bin/env python3
"""Build the Trackio logbook for the scRatio release audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "artifact_audit.json"
SPACE_ID = "kondratevakate/repro-scratio-release-audit"
TITLE = "Reproduction: scRatio release and genomics audit"
TITLES = {
    "C1": "Claim 1: single-ODE ratio dynamics",
    "C2": "Claim 2: Gaussian accuracy and runtime",
    "C3": "Claim 3: mutual-information estimation",
    "C4": "Claim 4: differential abundance",
    "C5": "Claim 5: batch-correction evaluation",
    "C6": "Claim 6: treatment-response applications",
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


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-5zbPdMNcl9", "single-cell-genomics"],
            "paper": {"arxiv_id": "2602.24201"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)
    executive = lb.ensure_page(PROJECT, "Executive summary")
    pages = {key: lb.ensure_page(PROJECT, title) for key, title in TITLES.items()}
    conclusion = lb.ensure_page(PROJECT, "Conclusion")

    gaussian = evidence["gaussian"]
    runtime = evidence["runtime"]
    small = runtime["small_training"]
    executive_text = f"""**Outcome.** The official scRatio release receives
**{evidence["prepared_score"]}/{evidence["score_denominator"]} points**, exactly
the frozen forecast. The core single-ODE estimator passes an analytic oracle,
and a fresh small Gaussian run favors direct ratios. The release also preserves
substantial executed application evidence, but paper-scale raw predictions and
checkpoints are external; Figure 2 and the MI notebook have provenance
mismatches.

| Item | Result |
| --- | --- |
| Official commit | `{evidence["repository"]["commit"]}` |
| Analytic-oracle max error | `{runtime["oracle_max_abs_error"]:.2e}` |
| Fresh direct / naive MSE | `{small["direct_mse"]:.4f}` / `{small["naive_mse"]:.4f}` |
| Cached Gaussian comparisons favoring direct | {gaussian["selected_schedule_comparisons"]}/{gaussian["selected_schedule_comparisons"]} |
| Figure 2 caption / cached scRatio runs | {gaussian["paper_caption_training_runs"]} / {gaussian["scratio_cached_values_per_row"][0]} |
| Prepared score | {evidence["prepared_score"]}/{evidence["score_denominator"]} |

Primary sources: [arXiv](https://arxiv.org/abs/2602.24201),
[OpenReview](https://openreview.net/forum?id=5zbPdMNcl9),
[official repository](https://github.com/theislab/scRatio), and
[Zenodo data](https://zenodo.org/records/20822377). No leaderboard or
third-party verdict contributes evidence.
"""
    lb.add_markdown_cell(PROJECT, executive, executive_text, title="Executive summary")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)
    poster = f"""<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #cdd9d4;padding:20px;background:#fff;color:#17241f"><h2 style="border-bottom:3px solid #287a63;padding-bottom:8px">scRatio: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Claim</th><th>Evidence</th><th>Verdict</th></tr><tr><td>C1 ODE</td><td>Analytic oracle error {runtime["oracle_max_abs_error"]:.2e}</td><td>SUPPORTED 2/2</td></tr><tr><td>C2 Gaussian</td><td>Fresh smoke + 30 cached comparisons; 3-vs-5 run mismatch</td><td>PARTIAL 1/2</td></tr><tr><td>C3-C6</td><td>Executed outputs; raw arrays/checkpoints external</td><td>PARTIAL 4/8</td></tr></table><p>Prepared {evidence["prepared_score"]}/{evidence["score_denominator"]}. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(PROJECT, executive, html=poster, title="Reproduction poster")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    lb.add_markdown_cell(
        PROJECT,
        pages["C1"],
        f"""**SUPPORTED - 2/2.** Source tracing finds the paper's divergence and
two score-correction terms in the released ratio wrapper. With exact Gaussian
vector fields and scores, the public `estimate_log_density_ratio` path recovers
three known values with maximum absolute error
`{runtime["oracle_max_abs_error"]:.3e}`.""",
        title=TITLES["C1"],
    )
    lb.add_code_cell(PROJECT, pages["C1"], output=json.dumps(runtime, indent=2), title="C1 evidence", code_paths=["repro_scratio/audit_artifacts.py"], language="python")

    lb.add_markdown_cell(
        PROJECT,
        pages["C2"],
        f"""**PARTIAL - 1/2.** The seeded 1,200-step CPU run is finite and gives
direct/naive MSE `{small["direct_mse"]:.4f}/{small["naive_mse"]:.4f}` with
`{small["speedup"]:.2f}x` speedup. Tracked CSV aggregation favors direct ratios
in all {gaussian["selected_schedule_comparisons"]} displayed comparisons, with
cached speedups {gaussian["speedup_min_median_max"][0]:.2f}x-
{gaussian["speedup_min_median_max"][2]:.2f}x. Yet each scRatio result row has
three values while Figure 2 says five runs; raw predictions and paper-scale
retraining are absent.""",
        title=TITLES["C2"],
    )
    lb.add_code_cell(PROJECT, pages["C2"], output=json.dumps(gaussian, indent=2), title="C2 evidence", code_paths=["repro_scratio/audit_artifacts.py"], language="python")

    mi = evidence["mutual_information"]
    lb.add_markdown_cell(
        PROJECT,
        pages["C3"],
        f"""**PARTIAL - 1/2.** The executed MI notebook keeps the qualitative
ranking, but 0/5 stochastic-path MAEs match current Table 1. The largest
absolute numeric difference is `{mi["largest_absolute_difference"]:.3f}` and
the referenced ratio arrays live outside the clone.""",
        title=TITLES["C3"],
    )
    lb.add_code_cell(PROJECT, pages["C3"], output=json.dumps(mi, indent=2), title="C3 evidence", code_paths=["repro_scratio/audit_artifacts.py"], language="python")

    da = evidence["differential_abundance"]
    lb.add_markdown_cell(
        PROJECT,
        pages["C4"],
        f"""**PARTIAL - 1/2.** Parsed executed outputs reproduce Table 2 at its
displayed precision. scRatio leads {da["scratio_leads_metrics"]}/
{da["metrics_checked"]} metrics; MELD leads correct-sign proportion at high DA.
The per-cell predictions used to calculate the outputs are external.""",
        title=TITLES["C4"],
    )
    lb.add_code_cell(PROJECT, pages["C4"], output=json.dumps(da, indent=2), title="C4 evidence", code_paths=["repro_scratio/audit_artifacts.py"], language="python")

    batch = evidence["batch_correction"]
    lb.add_markdown_cell(
        PROJECT,
        pages["C5"],
        """**PARTIAL - 1/2.** The NeurIPS and C. elegans notebooks preserve
executed LLR matrix shapes and condition-group counts consistent with the
paper. The clone does not contain their NPZ ratio arrays, so the before/after
magnitude reduction is not numerically recomputable from the clone alone.""",
        title=TITLES["C5"],
    )
    lb.add_code_cell(PROJECT, pages["C5"], output=json.dumps(batch, indent=2), title="C5 evidence", code_paths=["repro_scratio/audit_artifacts.py"], language="python")

    treatment = evidence["treatment_response"]
    lb.add_markdown_cell(
        PROJECT,
        pages["C6"],
        f"""**PARTIAL - 1/2.** Independent recomputation over
{treatment["combosciplex_pairs_recovered"]} printed ComboSciPlex pairs gives
Pearson `{treatment["combosciplex_pearson"]:.4f}` and Spearman
`{treatment["combosciplex_spearman"]:.4f}`. The PBMC donor/cytokine notebook
contains the expected analysis and executed plots but no model or raw group
ratios.""",
        title=TITLES["C6"],
    )
    lb.add_code_cell(PROJECT, pages["C6"], output=json.dumps(treatment, indent=2), title="C6 evidence", code_paths=["repro_scratio/audit_artifacts.py"], language="python")

    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        f"""The release supports **{evidence["prepared_score"]}/12** points.
Its core mathematical estimator is unusually checkable and passes a tight
oracle. The main reproducibility boundary is result provenance: application
notebooks are executed, but their inputs and prediction arrays are external,
the Gaussian replicate count conflicts with the caption, and the MI notebook
does not numerically match the current paper table.""",
        title="Conclusion",
    )
    root = PROJECT / "logbook" / "pages"
    lines = [f"# {TITLE}", "", lb.TOC_HEADING, "", lb.TOC_HEADER, lb.TOC_SEP, f"| [Executive summary](#/{executive}) |"]
    lines += [f"| [{TITLES[key]}](#/{slug}) |" for key, slug in pages.items()]
    lines += [f"| [Conclusion](#/{conclusion}) |", ""]
    (root / "index.md").write_text("\n".join(lines), encoding="utf-8")
    lb.write_site_files(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
