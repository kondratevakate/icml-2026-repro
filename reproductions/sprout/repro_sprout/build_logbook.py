#!/usr/bin/env python3
"""Build the static Trackio logbook for the SPROUT release audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = CANDIDATE / "evidence" / "audit_results.json"
SPACE_ID = "kondratevakate/repro-sprout-release-audit"
TITLE = (
    "Reproduction: Supervise Less, See More: Training-free Nuclear Instance "
    "Segmentation with Prototype-Guided Prompting"
)
TITLES = {
    "C1": "Claim 1: fully training-free automatic prompting",
    "C2": "Claim 2: self-reference feature calibration",
    "C3": "Claim 3: POT-Scan formulation and guarantee",
    "C4": "Claim 4: containment-aware refinement",
    "C5": "Claim 5: benchmark performance",
    "C6": "Claim 6: robustness and efficiency",
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
        code_paths=["audit_sprout.py"],
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
                "paper-qYfNhYenuu",
                "pathology",
                "instance-segmentation",
                "training-free",
            ],
            "paper": {"arxiv_id": "2511.19953"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive = lb.ensure_page(PROJECT, "Executive summary")
    pages = {key: lb.ensure_page(PROJECT, title) for key, title in TITLES.items()}
    conclusion = lb.ensure_page(PROJECT, "Conclusion")
    ot = data["partial_ot"]
    reference = data["self_reference"]
    refinement = data["refinement"]
    release = data["release"]

    summary = f"""**Outcome.** The release receives **5/12 points**, one point
above the frozen 4/12 forecast. The self-reference and containment mechanisms
execute on deterministic synthetic pathology inputs, but canonical POT crashes,
the minimally completed transport plan is N-scaled, and the empirical tables
have no released run-level artifacts.

| Independent check | Result |
| --- | --- |
| Self-reference precision / recall | {reference["precision"]:.3f} / {reference["recall"]:.3f} |
| Canonical POT | `{ot["canonical_error"]}` |
| Runtime-completed real / target mass | {ot["real_mass"]:.3f} / {ot["paper_target_real_mass"]:.3f} |
| Empty refinement input | `{refinement["empty_input_error"]}` |
| Released empirical artifacts | {len(release["empirical_artifacts"])} |
| Deterministic audit tests | 8 passed |
| Prepared score | 5/12 |

Primary sources: [arXiv](https://arxiv.org/abs/2511.19953),
[official code](https://github.com/Y-Research-SBU/SPROUT), and the
[project page](https://y-research-sbu.github.io/SPROUT/). No leaderboard or
third-party verdict contributes evidence.
"""
    lb.add_markdown_cell(PROJECT, executive, summary, title="Executive summary")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    poster = """<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #d2c3b3;padding:20px;background:#fffaf5;color:#29211b"><h2 style="border-bottom:3px solid #9f4f28;padding-bottom:8px">SPROUT: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Layer</th><th>Result</th></tr><tr><td>Self-reference</td><td>Synthetic H&amp;E precision 1.000, recall 0.607</td></tr><tr><td>Partial OT</td><td>Canonical crash; runtime-completed plan is N-scaled</td></tr><tr><td>Refinement</td><td>Containment penalty works; empty input crashes</td></tr><tr><td>Empirics</td><td>No predictions, processed splits, metrics, or repeat outputs</td></tr></table><p><b>Prepared 5/12</b>. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(PROJECT, executive, html=poster, title="Reproduction poster")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    add_page(
        PROJECT,
        pages["C1"],
        TITLES["C1"],
        """**PARTIAL - 1/2.** The release includes the intended image,
prototype, prompt, SAM, and refinement stages without a training loop.
End-to-end execution is blocked by the canonical POT error, external SAM2,
empty dataset link, and unresolved default backbone.""",
        {"release": release, "partial_ot": ot},
    )
    add_page(
        PROJECT,
        pages["C2"],
        TITLES["C2"],
        """**VERIFIED - 2/2.** The official high-confidence H&E
self-reference branch executes on a deterministic image with nine synthetic
nuclei, reaching precision 1.000 and recall 0.607.""",
        reference,
    )
    add_page(
        PROJECT,
        pages["C3"],
        TITLES["C3"],
        """**PARTIAL - 1/2.** `solve()` reads undefined `numItermax`.
Assigning only that field permits execution, but the resulting coupling uses
unit row sums and has total real mass 2.4 rather than paper-normalized 0.6.
Two distinct zero-cost couplings also refute the proof step from convexity to
equality of particular optimizers.""",
        {"partial_ot": ot, "theory": data["theory"]},
    )
    add_page(
        PROJECT,
        pages["C4"],
        TITLES["C4"],
        """**PARTIAL - 1/2.** Containment-aware soft-NMS independently
penalizes the large enclosing mask from 0.9 to 0.211 while retaining two
smaller masks at 0.8. The refinement path raises `IndexError` when no proposals
are supplied.""",
        refinement,
    )
    add_page(
        PROJECT,
        pages["C5"],
        TITLES["C5"],
        """**UNSUPPORTED - 0/2.** The release has no processed benchmark
splits, predictions, checkpoints, metric tables, or cached outputs from which
to recover the reported four-dataset performance.""",
        release,
    )
    add_page(
        PROJECT,
        pages["C6"],
        TITLES["C6"],
        """**UNSUPPORTED - 0/2.** Backbone, SAM, hyperparameter, repeat, and
runtime results are present only in the paper; their run-level artifacts are
not released.""",
        release,
    )
    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        """The package supports **5/12** points. It verifies a strong
self-reference mechanism and partial containment behavior, while identifying a
canonical POT failure, transport-mass mismatch, proof overreach, empty-input
refinement bug, and the absence of evidence needed for empirical reproduction.""",
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
