#!/usr/bin/env python3
"""Build the Trackio logbook from frozen MedMamba evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "claims_audit.json"
SPACE_ID = "kondratevakate/repro-medmamba-implementation"
TITLE = (
    "Reproduction: MedMamba: Multi-View State Space Models with "
    "Adaptive Graph Learning for Medical Time Series Classification"
)
PAGES = {
    "C1": "Claim 1: channel-wise embedding",
    "C2": "Claim 2: zero-padded difference",
    "C3": "Claim 3: frequency-specific filter",
    "C4": "Claim 4: sample-conditioned graph",
    "C5": "Claim 5: graph diffusion and spatial scan",
    "C6": "Claim 6: five-dataset results",
}


def create_logbook() -> None:
    if PROJECT.exists():
        raise RuntimeError(f"Logbook already exists: {PROJECT}")
    original_cwd = Path.cwd()
    original_find = lb.find_project_dir
    try:
        os.chdir(CANDIDATE)
        lb.find_project_dir = lambda *args, **kwargs: None
        created = Path(lb.create_logbook(title=TITLE, space_id=SPACE_ID))
    finally:
        lb.find_project_dir = original_find
        os.chdir(original_cwd)
    if created.resolve() != PROJECT.resolve():
        raise RuntimeError(f"Unexpected logbook path: {created}")


def claim_markdown(claim: dict) -> str:
    cid = claim["id"]
    if cid == "C1":
        return f"""**Verdict - FALSIFIED IN RELEASED IMPLEMENTATION.**

The paper declares channel-wise `T x C x D` embeddings and depthwise
convolutions. The pinned code returns
`{claim["observed_embedding_shape"]}` (`B x L x D`) and all three convolution
groups are `{claim["conv_groups"]}` for `{claim["input_channels"]}` input
channels. The channel-token axis no longer exists after MCE.
"""
    if cid == "C2":
        return f"""**Verdict - FALSIFIED IN RELEASED IMPLEMENTATION.**

The paper defines `[0; Z[2:T]-Z[1:T-1]]`. On a deterministic nonzero input,
the released operator's first step equals the input and reaches
`{claim["observed_first_step_max_abs"]:.1f}`; the declared operator is exactly
zero. A direct corrected mutation recovers the paper operator.
"""
    if cid == "C3":
        return f"""**Verdict - FALSIFIED IN RELEASED IMPLEMENTATION.**

For a length-16 probe there are `{claim["fft_bins_for_probe"]}` real FFT bins,
but each released real and imaginary filter has shape
`{claim["real_weight_shape"]}`. The same feature-wise gain is broadcast across
every frequency, so the implementation cannot amplify one physiological band
while suppressing another for the same feature.
"""
    if cid == "C4":
        gradients = claim["classification_proxy_graph_gradients"]
        return f"""**Verdict - FALSIFIED IN RELEASED IMPLEMENTATION.**

Two strongly different inputs produce identical adjacency
(`max |delta|={claim["adjacency_input_change_max_abs"]:.1f}`). Mutating graph
parameters changes adjacency by `{claim["adjacency_parameter_mutation_max_abs"]:.6f}`
but changes the returned representation by exactly
`{claim["output_parameter_mutation_max_abs"]:.1f}`.

Backpropagating a classification-output proxy gives these graph gradients:
`{json.dumps(gradients, sort_keys=True)}`. Every value is null. Structure
penalties can optimize this static graph, but classification cannot.
"""
    if cid == "C5":
        return f"""**Verdict - FALSIFIED IN RELEASED IMPLEMENTATION.**

The paper specifies normalized adjacency diffusion and a spatial SSM over
channels at each time. All traced Mamba inputs are
`{claim["mamba_input_shapes"]}`: sequence axis 16 is time, while the input has
4 channels. The block computes adjacency but never uses it in the returned
tensor; its nominal graph branch is a linear projection plus temporal Conv1d.
"""
    return """**Verdict - INCONCLUSIVE, NOT EXECUTED.**

The five datasets and seeds 41-45 were not run. The author repository releases
only an APAVA shell script, delegates data preparation to another repository,
and includes no checkpoints or cached metrics. Paper tables are not treated as
reproduction evidence.
"""


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()

    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-qPqJH0heR0"],
            "paper": {"arxiv_id": "2605.24961"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive_slug = lb.ensure_page(PROJECT, "Executive summary")
    claim_slugs = {
        cid: lb.ensure_page(PROJECT, title) for cid, title in PAGES.items()
    }
    conclusion_slug = lb.ensure_page(PROJECT, "Conclusion")

    executive = """**Outcome.** The pinned author implementation was audited
against six paper claims without using leaderboard or peer-reproduction
evidence. Five architecture claims are **FALSIFIED IN THE RELEASED
IMPLEMENTATION** by executable shape, mutation, invariance, and gradient
tests. The five-dataset empirical claim remains inconclusive.

| Item | Value |
| --- | --- |
| Prepared score | 10/12 |
| GPU | None |
| Medical data | None |
| Unit tests | 6 passing |
| Author commit | `418da50664338bc1d766394ee9c231496ab4de97` |

The identity Mamba substitute is used only to expose the released integration
code's tensor axes and dependencies. No accuracy conclusion depends on it.

Provenance: [arXiv](https://arxiv.org/abs/2605.24961) and
[author repository](https://github.com/zhangda1018/MedMamba).
"""
    lb.add_markdown_cell(PROJECT, executive_slug, executive, title="Executive summary")
    cell_id = lb.last_cell_id(PROJECT, page=executive_slug)
    lb.set_cell_pinned(PROJECT, cell_id, pinned=True, page=executive_slug)

    poster = """<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ccd5dd;padding:20px;background:#fff;color:#18212a">
<h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #176b87;padding-bottom:8px">MedMamba: released implementation audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | qPqJH0heR0 | arXiv 2605.24961v1 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef3f6"><th style="padding:6px;text-align:left">Claim</th><th>Executable finding</th><th>Verdict</th></tr>
<tr><td style="padding:6px">MCE</td><td>Channel axis collapsed; groups=1</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Difference</td><td>First step equals x[0], not zero</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Frequency filter</td><td>D weights broadcast over all FFT bins</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Adaptive graph</td><td>Input invariant; no output or classification gradient path</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Spatial SGM</td><td>Scans time; adjacency unused</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Five datasets</td><td>Fresh runs unavailable</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table><p style="font-size:12px;margin:12px 0 0">Prepared 10/12. No peer reproductions used.</p></div>"""
    lb.add_figure_cell(PROJECT, executive_slug, html=poster, title="Reproduction poster")
    cell_id = lb.last_cell_id(PROJECT, page=executive_slug)
    lb.set_cell_pinned(PROJECT, cell_id, pinned=True, page=executive_slug)

    for cid, slug in claim_slugs.items():
        claim = next(item for item in evidence["claims"] if item["id"] == cid)
        lb.add_markdown_cell(PROJECT, slug, claim_markdown(claim), title=PAGES[cid])
        lb.add_code_cell(
            PROJECT,
            slug,
            output=json.dumps(claim, indent=2, sort_keys=True),
            title=f"{cid} machine-readable evidence",
            code_paths=["repro_medmamba/audit_claims.py"],
            language="python",
        )

    conclusion = """The released code does not implement the paper's defining
channel-preserving, frequency-selective, or adaptive graph pathways as
specified. Most consequentially, the learned adjacency is disconnected from
the classifier: changing it changes no output, and classification gradients
cannot reach it.

These findings concern the pinned public implementation. They do not establish
that no private implementation could produce the reported empirical tables.
Those tables remain unverified pending a complete five-dataset execution.
"""
    lb.add_markdown_cell(PROJECT, conclusion_slug, conclusion, title="Conclusion")

    index = [
        f"# {TITLE}",
        "",
        lb.TOC_HEADING,
        "",
        lb.TOC_HEADER,
        lb.TOC_SEP,
        f"| [Executive summary](#/{executive_slug}) |",
    ]
    index.extend(f"| [{PAGES[cid]}](#/{claim_slugs[cid]}) |" for cid in PAGES)
    index.extend([f"| [Conclusion](#/{conclusion_slug}) |", ""])
    (PROJECT / "logbook" / "pages" / "index.md").write_text(
        "\n".join(index), encoding="utf-8"
    )
    lb.write_site_files(PROJECT)
    print(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
