#!/usr/bin/env python3
"""Build the Trackio logbook from frozen xMAE audit evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "artifact_audit.json"
PVC_EVIDENCE = HERE / "evidence" / "pvc_random_encoder_probe.json"
SPACE_ID = "kondratevakate/repro-xmae-release-audit"
TITLE = "Reproduction: xMAE release and runnable audit"
PAGE_TITLES = {
    "C1": "Claim 1: asymmetric cross-modal reconstruction",
    "C2": "Claim 2: released checkpoint and synthetic PVC example",
    "C3": "Claim 3: 15 of 19 downstream tasks",
    "C4": "Claim 4: 9.4k-hour pretraining cohort",
    "C5": "Claim 5: ECG-PPG timing structure",
    "C6": "Claim 6: six-study generalization",
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
    pvc = json.loads(PVC_EVIDENCE.read_text(encoding="utf-8"))
    checkpoint = evidence["checkpoint"]
    synthetic = evidence["synthetic_data"]
    code = evidence["code_and_spec"]
    create_logbook()

    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-NxbSbkLNMc", "biosignals"],
            "paper": {"arxiv_id": "2605.00973"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive = lb.ensure_page(PROJECT, "Executive summary")
    claims = {
        claim_id: lb.ensure_page(PROJECT, title)
        for claim_id, title in PAGE_TITLES.items()
    }
    conclusion = lb.ensure_page(PROJECT, "Conclusion")

    lb.add_markdown_cell(
        PROJECT,
        executive,
        f"""**Outcome.** The official xMAE release receives
**{evidence["prepared_score"]}/{evidence["score_denominator"]} points**, at the
lower bound of the frozen 2-4 range. The asymmetric cross-modal reconstruction
core is present and the synthetic artifacts are internally consistent.
Full credit is prevented by a disabled/non-updating curriculum and a
placeholder checkpoint that produces NaN embeddings in evaluation mode.

| Item | Result |
| --- | --- |
| Official commit | `{evidence["official_commit"]}` |
| Synthetic artifact | {synthetic["ppg_shape"][0]:,} x {synthetic["ppg_shape"][1]:,} PPG samples; {synthetic["subjects"]} synthetic patient IDs |
| Checkpoint strict load | {checkpoint["strict_load_passed"]} |
| Negative BN running variances | {checkpoint["negative_running_var_values"]} values across {checkpoint["affected_negative_running_var_tensors"]} tensors |
| Eval embedding | {checkpoint["eval_mode_embedding_nan_count"]} NaNs / {checkpoint["eval_mode_embedding_nan_count"]} values |
| Prepared score | {evidence["prepared_score"]}/{evidence["score_denominator"]} |

Primary sources: [arXiv](https://arxiv.org/abs/2605.00973),
[OpenReview](https://openreview.net/forum?id=NxbSbkLNMc), and the
[official repository](https://github.com/hzhou3/xMAE). No leaderboard or
third-party reproduction verdict contributes evidence.
""",
        title="Executive summary",
    )
    summary_id = lb.last_cell_id(PROJECT, page=executive)
    if summary_id:
        lb.set_cell_pinned(PROJECT, summary_id, pinned=True, page=executive)

    poster = f"""<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #cbd5df;padding:20px;background:#fff;color:#17212b">
<h2 style="margin:0 0 6px;border-bottom:3px solid #0c7c86;padding-bottom:8px">xMAE: independent release audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | NxbSbkLNMc | arXiv 2605.00973v1 | CPU/static audit</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#e9f5f5"><th style="padding:6px;text-align:left">Claim</th><th>Independent evidence</th><th>Verdict</th></tr>
<tr><td style="padding:6px">C1 method</td><td>Directional core present; curriculum disabled and never promoted</td><td style="color:#9a6500;font-weight:700">PARTIAL 1/2</td></tr>
<tr><td style="padding:6px">C2 release example</td><td>Artifacts align; eval checkpoint is NaN; PVC uses random encoder</td><td style="color:#9a6500;font-weight:700">PARTIAL 1/2</td></tr>
<tr><td style="padding:6px">C3-C6 empirical claims</td><td>Real weights and evaluated datasets explicitly unreleased</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table><p style="font-size:12px;margin:12px 0 0">Prepared {evidence["prepared_score"]}/{evidence["score_denominator"]}. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(PROJECT, executive, html=poster, title="Reproduction poster")
    poster_id = lb.last_cell_id(PROJECT, page=executive)
    if poster_id:
        lb.set_cell_pinned(PROJECT, poster_id, pinned=True, page=executive)

    lb.add_markdown_cell(
        PROJECT,
        claims["C1"],
        f"""**PARTIAL - 1/2.**

The released model implements the paper's central asymmetric path: full PPG
tokens act as keys/values, ECG tokens act as queries, and a lightweight decoder
reconstructs masked ECG. PPG masking is 0 and ECG masking starts at 0.8.

The paper specifies a loss-driven 80%-to-90% curriculum in 5% increments.
The canonical config sets `use_cl: false`, and the trainer contains no path
that increments `_ladder_idx`. A declared reverse bridge is also never called
({checkpoint["reverse_bridge_parameters_declared_but_unused"]:,} dead
parameters). The released core is therefore substantial but not a faithful
implementation of the full specified training method.
""",
        title=PAGE_TITLES["C1"],
    )
    lb.add_code_cell(
        PROJECT,
        claims["C1"],
        output=json.dumps(code, indent=2, sort_keys=True),
        title="C1 machine-readable evidence",
        code_paths=["repro_xmae/audit_artifacts.py"],
        language="python",
    )

    lb.add_markdown_cell(
        PROJECT,
        claims["C2"],
        f"""**PARTIAL - 1/2.**

The synthetic HDF5 and CSV agree on 3,000 rows, labels, 60 patient IDs, and
150 injected label flips. The 417-key checkpoint strict-loads and produces
finite outputs only while BatchNorm remains in training mode.

In correct evaluation mode, {checkpoint["affected_negative_running_var_tensors"]}
of {checkpoint["running_var_tensors"]} running-variance tensors contain
negative values and all {checkpoint["eval_mode_embedding_nan_count"]} tested
embedding values become NaN. The simple notebook never calls `eval()` and
also passes PPG as its ECG tensor. The PVC notebook does not load a checkpoint.
A seeded reproduction of that random-encoder path yielded finite features and
AUC {pvc["random_encoder_auc"]:.4f}; it validates only the synthetic smoke
pipeline, not pretrained xMAE.
""",
        title=PAGE_TITLES["C2"],
    )
    lb.add_code_cell(
        PROJECT,
        claims["C2"],
        output=json.dumps(
            {"checkpoint": checkpoint, "synthetic_data": synthetic, "pvc_probe": pvc},
            indent=2,
            sort_keys=True,
        ),
        title="C2 machine-readable evidence",
        code_paths=[
            "repro_xmae/audit_artifacts.py",
            "repro_xmae/test_audit_artifacts.py",
        ],
        language="python",
    )

    blockers = {
        "C3": "Real pretrained weights, downstream datasets, splits, baseline predictions, and repeated-run outputs are unreleased.",
        "C4": "The protected MIMIC-III waveform cohort and a frozen post-preprocessing cohort manifest are absent.",
        "C5": "Real pretrained representations, paired held-out signals, peak/onset labels, and canonical timing outputs are absent.",
        "C6": "The six evaluation studies, participant splits, probes, and matched device/acquisition outputs are absent.",
    }
    for claim_id in ("C3", "C4", "C5", "C6"):
        lb.add_markdown_cell(
            PROJECT,
            claims[claim_id],
            f"""**INCONCLUSIVE, NOT RERUN - 0/2.**

{blockers[claim_id]} The paper's printed tables and figures anchor the claim
but are not counted as an independent reproduction.
""",
            title=PAGE_TITLES[claim_id],
        )

    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        f"""The official release supports **{evidence["prepared_score"]}/12**
points: one for the implemented asymmetric reconstruction core and one for
internally consistent but placeholder-only runnable artifacts.

Independent checks lowered neither unsupported empirical claims nor
paper-table arithmetic into reproduced evidence. A paper-level rerun requires
real pretrained weights, the protected waveform cohort, six evaluation
datasets, canonical participant splits, and repeated baseline/probe outputs.
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
        f"| [{PAGE_TITLES[claim_id]}](#/{slug}) |"
        for claim_id, slug in claims.items()
    )
    lines.extend([f"| [Conclusion](#/{conclusion}) |", ""])
    (pages_root / "index.md").write_text("\n".join(lines), encoding="utf-8")
    lb.write_site_files(PROJECT)
    print(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
