"""Assemble results.json from the plan, the time log and work/logs/C00126_summary.json (host-side helper)."""
import csv
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1]
rows = list(csv.reader(open(D / "work" / "time_log.csv")))[1:]
total = round(sum(float(r[1]) for r in rows))
s = json.load(open(D / "work" / "logs" / "C00126_summary.json"))
assert all(s[k]["pass"] for k in ["S1", "S2", "S3", "S4"])

res = {
    "key": "icml-2026__SRGpmDQjJM",
    "status": "done",
    "repo_url": "https://github.com/JasonZuu/MedTPE",
    "commit": "59d64313d52698c2ebd62f9ceb255ad30d638012",
    "licence": "Apache-2.0 (LICENSE file in repository root)",
    "image": "p9pilot-srgpmdqjjm (python:3.12-slim, torch 2.14.1+cpu, transformers 4.57.6, tokenizers 0.22.2, datasets 5.0.1, accelerate 1.15.0, bitsandbytes 0.50.2; no vLLM)",
    "time_to_running_pipeline_min": 45,
    "container_seconds_total": total,
    "claims": [
        {
            "claim_id": "C00126", "type": "mechanism", "checkable_on_synthetic": True,
            "check": "checks/check_C00126.py: repo scripts create_tpe_tokenizer.py, llm_fit_tpe_tokenizer.py and clean_sft_QA.py run unmodified on synthetic CMedQA2-schema data with a byte-level BPE tokenizer (6000 tokens) and a tiny Qwen2 LM (1.06M params, pretrained on synthetic text). S1 lossless pair merge; S2 dependency-aware replacement (size preserved, no dependent or special token evicted, least-frequent eviction, 3.3 percent); S3 only new-token rows change plus parameter fraction on the real Qwen2.5-1.5B config; S4 pseudo-label bits-per-character before/after SSFT on 100 held-out prompts. SSFT was run with a minimal fp32 loop around the repo's embedding-split module because llm_sft.py could not finish on the shared CPU (see notes).",
            "outcome": "consistent",
            "rationale": "On synthetic data the repo's tokenizer step produced 200 two-token merges with exact decode round trip on 600/600 held-out texts and CR=0.2118, kept |V*|=|V|=6000 with identical id sets, and evicted 200 non-special tokens none of which lies in the merge-path dependent set (E∩D_paper=0, E∩D_repo=0, sub_tokens_missing=0; log lines [S1], [S2]). After SSFT all non-new tensors were identical, 200/200 new rows changed, the repo-printed trainable count was 25600 = 200 x 128, and the real Qwen2.5-1.5B config gives 7.68M/1.5437B = 0.4975 percent (rounds to 0.5, the bottom of the claimed 0.5-1.0 range; log [S3]). On held-out prompts the pseudo-label bits per character fell from 1.363 (mean-initialised MedTPE) to 0.730 after SSFT (original model 0.253; paired-bootstrap CI of the difference [-0.655, -0.610]; 57 percent of the gap closed; log [S4]), but the final-layer hidden-state cosine to the original model fell from 0.9975 to 0.9737 (CI entirely below 0), a qualification of the 'aligns with the pretrained latent space' wording.",
            "log": "work/logs/C00126.log",
        },
        {
            "claim_id": "C00445", "type": "headline_performance", "checkable_on_synthetic": False,
            "check": "None planned: F1 on MIMIC-IV ICU Mortality/Phenotyping and wall-clock minutes for Qwen2.5-1.5B depend on the real MIMIC-IV distribution, real pretrained weights and the authors' GPU; the MIMIC-IV pipeline is not in the public repository.",
            "outcome": "not_checkable_on_synthetic",
            "rationale": "No synthetic observation can be consistent or inconsistent with F1 0.122->0.122 / 0.201->0.218 or 38.3->23.5 minutes; the public repository only ships CMedQA2 and ECTSum workflows (README: MIMIC-IV and EHRSHOT are not included). Negative control, no run.",
            "log": None,
        },
        {
            "claim_id": "C00583", "type": "data_protocol", "checkable_on_synthetic": False,
            "check": "None planned: the claim lists the real datasets, tasks and backbones of the authors' evaluation protocol.",
            "outcome": "not_checkable_on_synthetic",
            "rationale": "The claim states which real datasets (MIMIC-IV, EHRSHOT), four tasks and backbones including Meditron3-8B were evaluated; this cannot be observed from synthetic data, and the repository scripts only expose the CMEDQA2 and ECTSUM choices. Negative control, no run.",
            "log": None,
        },
        {
            "claim_id": "C00590", "type": "secondary_empirical", "checkable_on_synthetic": False,
            "check": "None planned: generalisation to ARC-Challenge, ECTSum and CMedQA2 is an empirical performance statement on real text with real LLM weights.",
            "outcome": "not_checkable_on_synthetic",
            "rationale": "Random synthetic text with a tiny stand-in LM carries no information about performance on ARC-Challenge, ECTSum or CMedQA2; the repository's CMedQA2/ECTSum scripts need the real Kaggle data and vLLM on GPU. Negative control, no run.",
            "log": None,
        },
        {
            "claim_id": "C00959", "type": "headline_performance", "checkable_on_synthetic": False,
            "check": "None planned: compression rates 22.8-32.4 percent, token reduction up to 31 percent and latency reduction 34-63 percent are properties of the real EHR token distributions and GPU inference.",
            "outcome": "not_checkable_on_synthetic",
            "rationale": "The compression rate of a synthetic corpus is set by how the corpus was generated (0.2118 in our synthetic run says nothing about 22.8-32.4 percent on MIMIC-IV/EHRSHOT), and latency was not measured on GPU. Negative control, no run.",
            "log": None,
        },
    ],
    "blocked_reason": None,
    "notes": (
        "All data synthetic (checks/gen_synthetic.py); no real dataset, model weights or tokenizer downloaded; only two public config.json files (Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct) were fetched on the host to count parameters on the meta device. "
        "Plan: checks/plan.json (version 2, written before any claim run; amendments and 'changed_after_run' entries are in the file). "
        "Deviations: (1) the 'pretrained LLM' is a 1.06M-parameter Qwen2-architecture model pretrained for 394 steps on synthetic text (its directory carries the repo's default model name only so name-based lookups work); the first prep run crashed in generate() on an unused token_type_ids kwarg after the weights were saved, the second run reused them. "
        "(2) The repo trainer llm_sft.py was started three times (bf16 as shipped; fp32/AdamW 10 epochs; fp32/AdamW 1 epoch) and killed each time because it ran at about 0.1 micro-step/s on the shared CPU (2614 s for roughly 250 of 500 steps); S3/S4 therefore use checks/ssft_minimal.py (fp32, AdamW lr 5e-4, 117 updates of 4 sequences, train loss 5.58->2.90, 330 s cap) built from the repo's set_trainable_embeddings/restore_embeddings/sft_map_postfix_fn. S3/S4 thus test the repo's embedding-split module and objective, not llm_sft.py's accelerate/Adam8bit/bf16 path, at a scale about two orders below the paper's hours of SSFT. "
        "(3) The S2 least-common-token test is weak here: all 200 evicted tokens and the rarest retained eligible token have corpus frequency 0 (ties). "
        "(4) The repo tokenizer step dropped 430 byte-level merges for 200 evicted tokens (5741->5311), leaving 252 retained tokens whose merge path passes through an evicted token; on the 600 held-out texts only 1 text diverged from the base tokenisation and it contained an evicted token directly. "
        "(5) The repo-printed trainable percentage (7.96 percent for the tiny model) uses a denominator that excludes the frozen embedding matrix; for Qwen2.5-1.5B that convention gives 0.58 percent versus 0.4975 percent with the full parameter count. For untied Qwen2.5-7B the fraction is 0.235 percent (unique trainable rows) or 0.471 percent (input plus output rows), i.e. below the claimed 0.5-1.0 range under the first convention (reported as a qualification, not part of the pass rule). "
        "(6) Container time includes four killed runs (first prep 1185 s, of which the 394 pretraining steps were used; bf16 SFT 701 s; fp32 SFT attempts 139 s, 312 s and 2614 s). The docker image build (about 4 min, network on) is not counted. "
        "time_to_running_pipeline_min counts from the start of the task to the first successful repo CLI run (create_tpe_tokenizer.py, 11:53), about 20 min of which was pretraining the tiny stand-in LM. No claim was judged on real data."
    ),
}
json.dump(res, open(D / "results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("container minutes", total / 60)
