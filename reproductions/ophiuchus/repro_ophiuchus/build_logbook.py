#!/usr/bin/env python3
"""Build the Trackio logbook for the Ophiuchus release audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "artifact_audit.json"
SPACE_ID = "kondratevakate/repro-ophiuchus-release-audit"
TITLE = "Reproduction: Ophiuchus release and agent audit"
TITLES = {
    "C1": "Claim 1: interleaved medical agent and tools",
    "C2": "Claim 2: three-stage training protocol",
    "C3": "Claim 3: curated 64k trajectory corpus",
    "C4": "Claim 4: eight-benchmark superiority",
    "C5": "Claim 5: grounding, reflection, and ablations",
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
            "tags": ["icml2026-repro", "paper-coJqVkqb03", "medical-agents"],
            "paper": {"arxiv_id": "2512.14157"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)
    executive = lb.ensure_page(PROJECT, "Executive summary")
    pages = {key: lb.ensure_page(PROJECT, title) for key, title in TITLES.items()}
    conclusion = lb.ensure_page(PROJECT, "Conclusion")
    data, code, state = (
        evidence["release_data"],
        evidence["code_and_spec"]["checks"],
        evidence["state_machine"],
    )

    lb.add_markdown_cell(
        PROJECT,
        executive,
        f"""**Outcome.** The official Ophiuchus release receives
**{evidence["prepared_score"]}/{evidence["score_denominator"]} points**, the
lower bound of the frozen 2-4 range. A substantial local tool/reward core and
all three training-stage entrypoints exist, but the released examples,
preprocessor, parser, termination rule, and ATRL configuration do not form a
paper-faithful runnable path.

| Item | Result |
| --- | --- |
| Official commit | `{evidence["official_commit"]}` |
| Released SFT / RL records | {data["sft_records"]} / {data["rl_records"]} |
| Train / test parquet | {data["train_rows"]} / {data["test_rows"]}, byte-identical |
| Example paths existing as written | {data["referenced_paths_existing_as_written"]} / {data["referenced_paths"]} |
| Focused tests | 4 passing |
| Prepared score | {evidence["prepared_score"]}/{evidence["score_denominator"]} |

Primary sources: [arXiv](https://arxiv.org/abs/2512.14157),
[OpenReview](https://openreview.net/forum?id=coJqVkqb03), and the
[official repository](https://github.com/SII-zyj/Ophiuchus). No leaderboard
or third-party verdict contributes evidence.
""",
        title="Executive summary",
    )
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)
    poster = f"""<div style="font-family:system-ui,Arial;max-width:900px;border:1px solid #ccd6df;padding:20px;background:#fff;color:#17212b"><h2 style="border-bottom:3px solid #315f8a;padding-bottom:8px">Ophiuchus: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Claim</th><th>Evidence</th><th>Verdict</th></tr><tr><td>C1 agent</td><td>Core runs; format, mask, and early-stop mismatch</td><td>PARTIAL 1/2</td></tr><tr><td>C2 training</td><td>Stages present; preprocess/KL/path mismatch</td><td>PARTIAL 1/2</td></tr><tr><td>C3-C5</td><td>Corpus, checkpoint, evaluation absent</td><td>INCONCLUSIVE</td></tr></table><p>Prepared {evidence["prepared_score"]}/{evidence["score_denominator"]}. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(PROJECT, executive, html=poster, title="Reproduction poster")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    lb.add_markdown_cell(
        PROJECT,
        pages["C1"],
        f"""**PARTIAL - 1/2.** The three tool handlers, multi-image state,
format/answer/tool rewards, and local zoom path are present. A zoom plus correct
answer yields reward {state["final_reward"]:.1f}. However, the two released SFT
trajectories use a `<tool_call>` form that the environment does not parse.
Identical repeated calls do not early-stop, and the released list-valued
segmentation `mask_path` raises `{state["released_list_mask_reset_error"]}`.""",
        title=TITLES["C1"],
    )
    lb.add_code_cell(PROJECT, pages["C1"], output=json.dumps(state, indent=2), title="C1 evidence", code_paths=["repro_ophiuchus/audit_artifacts.py"], language="python")
    lb.add_markdown_cell(
        PROJECT,
        pages["C2"],
        f"""**PARTIAL - 1/2.** Cold-start SFT, reflective SFT, and ATRL scripts
exist with decomposed rewards. The released preprocessor contains an executable
placeholder and raises `NameError`; ATRL uses placeholder data/model paths.
Its KL assignments are `{code["atrl_kl_assignments"]}`, so the effective
setting enables KL loss with coefficient 0.01 although the paper specifies no
KL penalty.""",
        title=TITLES["C2"],
    )
    lb.add_code_cell(PROJECT, pages["C2"], output=json.dumps(code, indent=2), title="C2 evidence", code_paths=["repro_ophiuchus/audit_artifacts.py"], language="python")
    blockers = {
        "C3": f"Only {data['sft_records']} SFT and {data['rl_records']} RL records are released; the claimed 64k corpus and provenance manifest are absent.",
        "C4": "The trained Ophiuchus checkpoint, complete evaluation commands, benchmark inputs/splits, and baseline predictions are absent.",
        "C5": "Grounded segmentation, reflection, transfer, and reward ablations require the unreleased checkpoint, datasets, trajectories, and outputs.",
    }
    for key in ("C3", "C4", "C5"):
        lb.add_markdown_cell(PROJECT, pages[key], f"**INCONCLUSIVE, NOT RERUN - 0/2.**\n\n{blockers[key]}", title=TITLES[key])
    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        f"""The release supports **{evidence["prepared_score"]}/10** points:
one for the substantial but internally inconsistent medical tool loop and one
for the present but non-runnable/paper-mismatched training entrypoints.
Paper-level claims remain inconclusive until the promised corpus, checkpoint,
evaluation code, splits, predictions, and ablation outputs are released.""",
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
