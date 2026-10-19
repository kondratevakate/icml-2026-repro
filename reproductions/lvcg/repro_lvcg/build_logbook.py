#!/usr/bin/env python3
"""Build the static Trackio logbook for the LVCG release audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "audit.json"
SPACE_ID = "kondratevakate/repro-lvcg-release-audit"
TITLE = "Reproduction: Learning Cardiac Latent Representations in VCG Space"
TITLES = {
    "C1": "Claim 1: fixed ECG/VCG geometry",
    "C2": "Claim 2: model and training objective",
    "C3": "Claim 3: six-dataset linear probing",
    "C4": "Claim 4: multi-lead reconstruction",
    "C5": "Claim 5: geometry and component ablations",
    "C6": "Claim 6: non-cardiac detection",
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


def add_page(project: Path, slug: str, title: str, markdown: str, evidence: object) -> None:
    lb.add_markdown_cell(project, slug, markdown, title=title)
    lb.add_code_cell(
        project,
        slug,
        output=json.dumps(evidence, ensure_ascii=False, indent=2),
        title=f"{title} evidence",
        code_paths=["repro_lvcg/audit_lvcg.py"],
        language="python",
    )


def main() -> int:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-hS6iw4PM8K", "medical-ecg"],
            "paper": {"arxiv_id": "2605.31249"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive = lb.ensure_page(PROJECT, "Executive summary")
    pages = {key: lb.ensure_page(PROJECT, title) for key, title in TITLES.items()}
    conclusion = lb.ensure_page(PROJECT, "Conclusion")

    runtime = data["runtime_import"]
    splits = data["splits"]
    conformance = data["conformance"]
    executive_text = f"""**Outcome.** The official LVCG release receives
**{data["prepared_score"]}/{data["score_denominator"]} points**, below the
frozen forecast of 4/12. The printed VCG geometry works independently and the
released record splits are disjoint, but the official Python package cannot
import because `lvcg/data` is absent.

| Independent check | Result |
| --- | --- |
| Official package import | `{runtime["last_error_line"]}` |
| Missing internal imports | {len(runtime["missing_relative_imports"])} |
| Synthetic geometry check | {data["geometry"]["passes_1e_4"]} |
| Six task record splits disjoint | {splits["all_record_splits_disjoint"]} |
| Pretrained checkpoints released | {len(data["artifacts"]["repository"]["checkpoints"])} |
| Prepared score | {data["prepared_score"]}/{data["score_denominator"]} |

Primary sources: [arXiv](https://arxiv.org/abs/2605.31249),
[official code](https://github.com/BosonHwang/LVCG), and
[MIMIC-IV-ECG-Ext-ICD](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/).
No leaderboard or third-party verdict contributes evidence, and no clinical
waveforms are included.
"""
    lb.add_markdown_cell(PROJECT, executive, executive_text, title="Executive summary")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    poster = f"""<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #b9c6d3;padding:20px;background:#fff;color:#17212b"><h2 style="border-bottom:3px solid #b42318;padding-bottom:8px">LVCG: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Evidence layer</th><th>Result</th></tr><tr><td>Geometry</td><td>Independent ridge lift/project passes at 1e-4</td></tr><tr><td>Runtime</td><td>Official package cannot import: missing lvcg/data</td></tr><tr><td>Linear probing</td><td>Splits present; no checkpoint/results; PTB-XL Sub label order differs</td></tr><tr><td>Non-cardiac</td><td>Paper uses MIMIC-IV-ECG-Ext-ICD; code config uses AI-READI</td></tr></table><p><b>Prepared {data["prepared_score"]}/{data["score_denominator"]}</b>. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(PROJECT, executive, html=poster, title="Reproduction poster")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    add_page(
        PROJECT,
        pages["C1"],
        TITLES["C1"],
        f"""**PARTIAL - 1/2.** The printed ridge lift and projection recover
synthetic latent trajectories with maximum error
`{data["geometry"]["max_latent_abs_error"]:.2e}`. Official execution is
blocked because the geometry module imports absent `lvcg.data` modules.""",
        {"geometry": data["geometry"], "runtime": runtime},
    )
    add_page(
        PROJECT,
        pages["C2"],
        TITLES["C2"],
        """**UNSUPPORTED - 0/2.** Training cannot import. The released default
GRU path required a post-paper beat-loss shape fix, and the implementation
adds a unit-weight base-beat loss absent from Equation 7.""",
        conformance,
    )
    add_page(
        PROJECT,
        pages["C3"],
        TITLES["C3"],
        """**PARTIAL - 1/2.** Released record and patient-ID splits are
disjoint and Table-1 averages are arithmetically consistent. No checkpoint or
results are released; the checkpoint override is ineffective; and PTB-XL
Sub-Class label order differs between train and validation/test.""",
        {"splits": splits, "arithmetic": data["table_arithmetic"]},
    )
    add_page(
        PROJECT,
        pages["C4"],
        TITLES["C4"],
        """**UNSUPPORTED - 0/2.** The repository contains no checkpoint,
reconstruction predictions, per-lead outputs, or metric files. Table 2 cannot
be regenerated.""",
        data["artifacts"],
    )
    add_page(
        PROJECT,
        pages["C5"],
        TITLES["C5"],
        """**UNSUPPORTED - 0/2.** Ablations are display-only: no ablation
configs, checkpoints, seeds, or outputs are released.""",
        data["artifacts"],
    )
    add_page(
        PROJECT,
        pages["C6"],
        TITLES["C6"],
        """**UNSUPPORTED - 0/2.** The paper evaluates MIMIC-IV-ECG-Ext-ICD,
but the repository supplies an AI-READI provider instead and contains no
paper-dataset loader, folds, labels, checkpoint, or outputs.""",
        conformance,
    )
    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        f"""The release supports **{data["prepared_score"]}/12** points. The
central geometric equation is numerically coherent and the split inventory is
useful, but the package is not runnable and none of the empirical tables has
its required checkpoint or outputs. Paper/code differences in the objective,
PTB-XL Sub-Class label ordering, and non-cardiac dataset make the released
empirical path non-conformant.""",
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
