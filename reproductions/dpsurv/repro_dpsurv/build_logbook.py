#!/usr/bin/env python3
"""Build the static Trackio logbook for the DPsurv release audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = CANDIDATE / "evidence" / "audit_results.json"
SPACE_ID = "kondratevakate/repro-dpsurv-release-audit"
TITLE = (
    "Reproduction: DPsurv: Dual-Prototype Evidential Fusion for "
    "Uncertainty-Aware and Interpretable Whole-Slide Image Survival Prediction"
)
TITLES = {
    "C1": "Claim 1: five-cohort data and evaluation protocol",
    "C2": "Claim 2: dual-prototype evidential fusion",
    "C3": "Claim 3: GRFN survival bounds and mixture",
    "C4": "Claim 4: headline discrimination",
    "C5": "Claim 5: calibration and uncertainty",
    "C6": "Claim 6: interpretability, ablations, and robustness",
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
    project: Path,
    slug: str,
    title: str,
    markdown: str,
    evidence: object,
) -> None:
    lb.add_markdown_cell(project, slug, markdown, title=title)
    lb.add_code_cell(
        project,
        slug,
        output=json.dumps(evidence, ensure_ascii=False, indent=2),
        title=f"{title} evidence",
        code_paths=["audit_dpsurv.py"],
        language="python",
    )


def main() -> int:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": [
                "icml2026-repro",
                "paper-RKqL4GYXz3",
                "pathology",
                "survival-analysis",
                "uncertainty",
            ],
            "paper": {"arxiv_id": "2510.00053"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive = lb.ensure_page(PROJECT, "Executive summary")
    pages = {key: lb.ensure_page(PROJECT, title) for key, title in TITLES.items()}
    conclusion = lb.ensure_page(PROJECT, "Conclusion")
    splits = data["splits"]
    model = data["model_and_theory"]
    release = data["release"]
    visual = data["visualization"]

    cohort_rows = [
        (
            name,
            item["unique_cases"],
            item["unique_slides"],
            item["folds"][0]["missing_dss_values"],
        )
        for name, item in splits["cohorts"].items()
    ]
    missing_total = sum(row[3] for row in cohort_rows)
    lambda_075 = model["lambda_audit"]["0.75"]

    summary = f"""**Outcome.** The release receives **4/12 points**, one point
below the frozen 5/12 forecast. The outer folds and core evidential model pass,
but non-default lambda semantics, empty components, and the visualization
entrypoint fail; the empirical TCGA outputs are absent.

| Independent check | Result |
| --- | --- |
| Cohorts / outer folds / CSVs | {splits["cohort_count"]} / {splits["fold_directory_count"]} / {splits["csv_count"]} |
| Missing DSS endpoint rows | {missing_total} |
| Synthetic gradient tensors finite | {model["finite_gradient_tensors"]}/{model["gradient_tensor_count"]} |
| lambda=0.75 training/paper max gap | {lambda_075["training_vs_paper_max_abs"]:.3f} |
| lambda=0.75 training survival maximum | {lambda_075["official_training_curve_max"]:.3f} |
| Visualization entrypoint | `{visual["effective_entrypoint_error"]}` |
| Prepared score | 4/12 |

Primary sources: [arXiv](https://arxiv.org/abs/2510.00053) and
[official code](https://github.com/YuchengXing99/DPsurv). No leaderboard or
third-party verdict contributes evidence.
"""
    lb.add_markdown_cell(PROJECT, executive, summary, title="Executive summary")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    poster = """<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #c9bfcf;padding:20px;background:#fcf9ff;color:#251d2a"><h2 style="border-bottom:3px solid #6d3f86;padding-bottom:8px">DPsurv: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Layer</th><th>Result</th></tr><tr><td>Splits</td><td>25 folds; no case/slide leakage; 86 rows missing DSS endpoint</td></tr><tr><td>Mechanics</td><td>Synthetic dual-prototype forward, loss, and backward pass</td></tr><tr><td>Theory/code</td><td>Bel <= Pl; lambda paths disagree away from 0.5</td></tr><tr><td>Empirics</td><td>No features, checkpoints, predictions, summaries, or logs</td></tr></table><p><b>Prepared 4/12</b>. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(PROJECT, executive, html=poster, title="Reproduction poster")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    add_page(
        PROJECT,
        pages["C1"],
        TITLES["C1"],
        """**PARTIAL - 1/2.** All five cohorts and five folds are present.
Every case and slide is held out exactly once, with no outer train/test overlap.
Eighty-six unique rows lack a DSS endpoint and are filtered by the trainer.
WSI, UNI2-h, and PANTHER inputs are external.""",
        splits,
    )
    add_page(
        PROJECT,
        pages["C2"],
        TITLES["C2"],
        """**VERIFIED - 2/2.** A deterministic two-component,
two-prototype forward, prevalence mixture, evidential loss, backward pass, and
all twelve parameter-gradient tensors execute finitely.""",
        model,
    )
    add_page(
        PROJECT,
        pages["C3"],
        TITLES["C3"],
        """**PARTIAL - 1/2.** Numeric checks confirm Bel <= Pl. The
paper specifies lambda*Bel+(1-lambda)*Pl. Training instead multiplies both
terms by lambda, and evaluation reverses the paper's Bel/Pl direction. The
paths coincide only at lambda 0.5.""",
        model["lambda_audit"],
    )
    add_page(
        PROJECT,
        pages["C4"],
        TITLES["C4"],
        """**UNSUPPORTED - 0/2.** No released features, embeddings,
checkpoints, predictions, metric summaries, or logs recover the five-cohort
C-index table.""",
        release,
    )
    add_page(
        PROJECT,
        pages["C5"],
        TITLES["C5"],
        """**UNSUPPORTED - 0/2.** No run-level IBS, IBLL, BPI, PPI, or
coverage artifacts are released, and non-default lambda is not implemented as
stated.""",
        {"release": release, "lambda_audit": model["lambda_audit"]},
    )
    add_page(
        PROJECT,
        pages["C6"],
        TITLES["C6"],
        """**UNSUPPORTED - 0/2.** The cached visualization notebook has
empty data paths. Its effective encoder helper is a shadowing second definition
that calls undefined `create_embedding_model`. No ablation, sensitivity,
CONCH, clinician, or runtime outputs are released.""",
        {"visualization": visual, "release": release},
    )
    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        """The package supports **4/12** points. It verifies the clean outer
fold structure and core dual-prototype mechanics, while identifying
non-default-lambda discrepancies, zero-prototype and visualization failures,
and the absence of run-level evidence for every empirical table.""",
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
