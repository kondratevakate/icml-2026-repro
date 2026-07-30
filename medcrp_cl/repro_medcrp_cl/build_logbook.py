#!/usr/bin/env python3
"""Build the static Trackio logbook for the MedCRP-CL release audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = CANDIDATE / "evidence" / "audit_results.json"
SPACE_ID = "kondratevakate/repro-medcrp-cl-release-audit"
TITLE = (
    "Reproduction: MedCRP-CL: Continual Medical Image Segmentation via "
    "Bayesian Nonparametric Semantic Modality Discovery"
)
TITLES = {
    "C1": "Claim 1: online semantic-modality discovery",
    "C2": "Claim 2: replay-free structure-aware continual learning",
    "C3": "Claim 3: headline continual-segmentation performance",
    "C4": "Claim 4: robustness and ablation evidence",
    "C5": "Claim 5: efficiency and clinical breadth",
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
        code_paths=["audit_medcrp_cl.py"],
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
                "paper-v0DWbfP3b9",
                "medical-image-segmentation",
                "continual-learning",
            ],
            "paper": {"arxiv_id": "2605.20297"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive = lb.ensure_page(PROJECT, "Executive summary")
    pages = {key: lb.ensure_page(PROJECT, title) for key, title in TITLES.items()}
    conclusion = lb.ensure_page(PROJECT, "Conclusion")
    checkpoint = data["checkpoint"]
    theory = data["theory_check"]
    release = data["release"]

    summary = f"""**Outcome.** The release receives **4/10 points**, one point
below the frozen 5/10 forecast. The official checkpoint contains the coherent
five-way task partition and aggregate replay-free EWC state, but it uses CRP
alpha 2.0 rather than the reported 5.0 and provides no metric outputs.

| Independent check | Result |
| --- | --- |
| Tasks / discovered modalities | {checkpoint["total_tasks"]} / {len(checkpoint["group_sizes"])} |
| Checkpoint alpha / paper alpha | {checkpoint["checkpoint_alpha"]:.1f} / {checkpoint["code_and_paper_alpha"]:.1f} |
| Fixed-Gaussian tail error | {theory["fixed_distribution_tail_error"]:.4f} |
| Released run-level metric files | {len(release["run_level_metric_files"])} |
| Deterministic audit tests | 9 passed |
| Prepared score | 4/10 |

Primary sources: [arXiv](https://arxiv.org/abs/2605.20297),
[official code](https://github.com/zygao930/MedCRP-CL), and
[official checkpoint](https://huggingface.co/clg-g/MedCRP-CL). No leaderboard
or third-party verdict contributes evidence.
"""
    lb.add_markdown_cell(PROJECT, executive, summary, title="Executive summary")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    poster = """<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #b9c6cc;padding:20px;background:#fff;color:#17232a"><h2 style="border-bottom:3px solid #0f766e;padding-bottom:8px">MedCRP-CL: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Layer</th><th>Result</th></tr><tr><td>Partition</td><td>Official state: 16 tasks, 5 coherent groups; alpha mismatch 2.0 vs 5.0</td></tr><tr><td>Mechanics</td><td>Modality LoRA isolation and aggregate EWC state independently checked</td></tr><tr><td>Theory</td><td>Fixed Gaussian overlap contradicts the appendix zero-error step</td></tr><tr><td>Empirics</td><td>No processed splits, metric outputs, predictions, or baselines released</td></tr></table><p><b>Prepared 4/10</b>. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(PROJECT, executive, html=poster, title="Reproduction poster")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    add_page(
        PROJECT,
        pages["C1"],
        TITLES["C1"],
        """**PARTIAL - 1/2.** The official state has the coherent cardiac,
polyp, dermoscopy, chest-X-ray, and breast-ultrasound partition. It was not
regenerated from processed prompt data and records alpha 2.0 rather than the
reported 5.0. The appendix zero-error proposition also fails for overlapping
fixed-variance Gaussians.""",
        {"checkpoint": checkpoint, "theory_check": theory},
    )
    add_page(
        PROJECT,
        pages["C2"],
        TITLES["C2"],
        """**VERIFIED - 2/2.** Independent CPU checks verify modality-isolated
LoRA updates and Welford statistics. The official EWC artifact contains Fisher
and anchor tensors plus identifiers, not replay examples. Task-specific
adapter/head Fisher names do not match on later aliases, narrowing EWC coverage
to the stable modality LoRA/enhancer names.""",
        data["ewc_state"],
    )
    add_page(
        PROJECT,
        pages["C3"],
        TITLES["C3"],
        """**UNSUPPORTED - 0/2.** The release has no processed splits,
run-level outputs, predictions, seeds, or metric tables from which to
independently recover 73.3% Dice, 4.1% forgetting, or their uncertainty.""",
        release,
    )
    add_page(
        PROJECT,
        pages["C4"],
        TITLES["C4"],
        """**UNSUPPORTED - 0/2.** Order, prompt-perturbation, ablation, merge,
and baseline numbers appear only in the paper; corresponding outputs and
scripts for aggregating those tables are absent.""",
        release,
    )
    add_page(
        PROJECT,
        pages["C5"],
        TITLES["C5"],
        """**PARTIAL - 1/2.** Source and dataset documentation cover the
architecture and 16-task medical scope, but the 8.6M trainable-parameter
comparison was not independently recomputed and clinical predictions are not
released. The environment export is not pip-installable.""",
        release,
    )
    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        """The package supports **4/10** points. It verifies useful release
mechanics and a coherent cached partition, while identifying a checkpoint
configuration mismatch, a non-vanishing Gaussian overlap missed by the
appendix proof, incomplete EWC coverage for shared task-specific modules, and
the absence of evidence needed for the empirical tables.""",
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

