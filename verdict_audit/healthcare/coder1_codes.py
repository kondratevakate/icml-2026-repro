"""Coder 1 codes for protocol 3, written as data with consistency checks.

Run from this folder to regenerate papers_coder1.csv and claims_coder1.csv.
"""
import csv

D, W, R, C = "derived_artifacts_missing", "weights_missing", "raw_data_unavailable", "code_missing_or_broken"
X, S, U, O = "paper_code_divergence", "compute_or_scale", "underspecified_paper", "other"
PRECEDENCE = [X, C, R, W, D, S, U, O]

PAPERS = [
    ("3dmedagent", "TH6pLxCOQ3", "yes", 4, 12, 4, 12, "partial"),
    ("bayes_causal_meta", "k76ll7aQyE", "yes", 5, 10, 5, 10, "full"),
    ("came_grad", "T7y2wavrFM", "yes", 2, 10, 2, 10, "partial"),
    ("caml", "Fisw2kc7EY", "no", 7, 12, 4, 12, "partial"),
    ("dc_pnpdp", "jEBkuuETjr", "yes", 6, 12, 5, 12, "partial"),
    ("dpsurv", "RKqL4GYXz3", "yes", 5, 12, 4, 12, "full"),
    ("flow_density_ratio", "5zbPdMNcl9", "yes", 7, 12, 7, 12, "full"),
    ("glean", "FP23eFYhAy", "yes", 2, 12, 2, 12, "none"),
    ("globalhealthatlas", "02Nq73lCe6", "yes", 6, 12, 6, 12, "partial"),
    ("lvcg", "hS6iw4PM8K", "yes", 4, 12, 2, 12, "partial"),
    ("ma_rag", "S7lpdz7NAW", "yes", 4, 12, 2, 12, "partial"),
    ("medcrp_cl", "v0DWbfP3b9", "yes", 5, 10, 4, 10, "full"),
    ("ophiuchus", "coJqVkqb03", "yes", 3, 10, 2, 10, "partial"),
    ("spmind", "UJcB3XrffF", "yes", 4, 10, 3, 10, "full"),
    ("sprout", "qYfNhYenuu", "yes", 4, 12, 5, 12, "partial"),
    ("xmae", "NxbSbkLNMc", "yes", 4, 12, 2, 12, "partial"),
]

# folder, claim_type, score, blockers, divergence, pointer
CLAIMS = [
    ("3dmedagent", "mechanism", 2, [], "no", "Routing/config verified by static checks and CPU dry run"),
    ("3dmedagent", "data_protocol", 2, [], "no", "DeepChestVQA CSV composition recounted independently"),
    ("3dmedagent", "headline_performance", 0, [R, D, O], "no", "CT volumes, masks, reports, caches, predictions, credentials omitted"),
    ("3dmedagent", "headline_performance", 0, [R, D, O], "no", "Model comparisons not rerun; release omits inputs and predictions"),
    ("3dmedagent", "secondary_empirical", 0, [R, D], "no", "Staged ablations not rerun; intermediate caches absent"),
    ("3dmedagent", "secondary_empirical", 0, [R, D], "no", "Radiologist agreement and per-turn outputs, expert traces absent"),
    ("bayes_causal_meta", "mechanism", 1, [X], "yes", "Released prior normalizes Wz, unlike paper linear theta+Wz"),
    ("bayes_causal_meta", "theory", 2, [], "no", "TV/Pinsker bound and triangle decomposition check out"),
    ("bayes_causal_meta", "theory", 1, [O], "no", "Appendix A.2 constant drops condition number; 2D counterexample"),
    ("bayes_causal_meta", "secondary_empirical", 1, [C, S], "no", "Toy path runs after workarounds; training seed not set; not canonical 30 runs"),
    ("bayes_causal_meta", "headline_performance", 0, [R, D, W], "no", "UK Biobank data, clinical outputs, checkpoints absent"),
    ("came_grad", "theory", 2, [], "no", "Equations 8-14 reimplemented, residuals at machine precision"),
    ("came_grad", "theory", 0, [O], "no", "SDE/flat-minimum mechanism not established by printed equations"),
    ("came_grad", "headline_performance", 0, [C, W, D], "no", "Core optimizer withheld; weights, splits, logs absent"),
    ("came_grad", "secondary_empirical", 0, [C, W, D], "no", "Optimizer comparison not rerun; core optimizer withheld"),
    ("came_grad", "secondary_empirical", 0, [C, W, D], "no", "Stage ablations not rerun; core optimizer withheld"),
    ("caml", "theory", 0, [O], "no", "Headline theorem false: N(u)=u^2 counterexample"),
    ("caml", "mechanism", 2, [], "no", "Closed-form offset matches independent implementation exactly"),
    ("caml", "headline_performance", 1, [X, S], "yes", "Released Heat settings differ from paper (boundary weight, stop threshold)"),
    ("caml", "headline_performance", 0, [X, D], "yes", "Five code/paper configuration mismatches; no cached outputs"),
    ("caml", "secondary_empirical", 0, [C], "no", "Entrypoints hardcoded to MLP; four baseline losses absent"),
    ("caml", "secondary_empirical", 1, [C], "no", "No ablation or sensitivity runner released"),
    ("dc_pnpdp", "theory", 2, [], "no", "Dual-coupled fixed point verified on scalar quadratic"),
    ("dc_pnpdp", "theory", 1, [O], "no", "Loose-PnP fixed point is Moreau-envelope stationarity; incompatible theorem scalings"),
    ("dc_pnpdp", "mechanism", 1, [X], "yes", "Code scales full Gaussian FFT; paper samples fixed amplitude with random phase"),
    ("dc_pnpdp", "headline_performance", 0, [X, R, D], "yes", "No data or reconstructions; example runs DiffPIR with w_tik=0 vs paper rho"),
    ("dc_pnpdp", "secondary_empirical", 0, [C], "no", "No MRI implementation or configuration in repository"),
    ("dc_pnpdp", "secondary_empirical", 1, [D, S], "no", "Ablation and A100 timing not cached or regenerable"),
    ("dpsurv", "data_protocol", 1, [D, S], "no", "Splits verified; WSI/UNI2/PANTHER inputs absent"),
    ("dpsurv", "mechanism", 2, [], "no", "Synthetic forward/backward of evidential fusion executes"),
    ("dpsurv", "mechanism", 1, [X], "yes", "lambda*Bel+(1-lambda)*Pl implemented incorrectly; training uses lambda twice"),
    ("dpsurv", "headline_performance", 0, [W, D], "no", "No features, checkpoints, predictions or logs for C-index"),
    ("dpsurv", "secondary_empirical", 0, [X, D], "yes", "No calibration outputs; lambda mismatch invalidates non-default path"),
    ("dpsurv", "secondary_empirical", 0, [C, D], "no", "Notebook NameError; no ablation or runtime outputs"),
    ("flow_density_ratio", "mechanism", 2, [], "no", "Estimator recovers analytic Gaussian ratio; matches Proposition 4.1"),
    ("flow_density_ratio", "headline_performance", 1, [D, S, O], "no", "Raw predictions absent; paper-scale retraining not done; 3 vs 5 runs"),
    ("flow_density_ratio", "headline_performance", 1, [D, O], "no", "Notebook MAEs do not match Table 1; ratio arrays external"),
    ("flow_density_ratio", "headline_performance", 1, [D], "no", "Table 2 reproduced from outputs; per-cell predictions external"),
    ("flow_density_ratio", "secondary_empirical", 1, [D], "no", "NPZ arrays absent; magnitude decrease not recomputable"),
    ("flow_density_ratio", "secondary_empirical", 1, [D], "no", "Model, raw ratios and group summary absent"),
    ("glean", "theory", 1, [O], "no", "Appendix A bound uses epsilon squared; derivation supports 2*epsilon"),
    ("glean", "data_protocol", 1, [C, D], "no", "MIMIC payload verified; trajectories, labels, judge outputs not released"),
    ("glean", "headline_performance", 0, [C, D, O], "no", "Table 1 not regenerable; prose conflicts with Table 1"),
    ("glean", "secondary_empirical", 0, [C, D], "no", "No triggered cases, guidelines, judge scores or predictions"),
    ("glean", "secondary_empirical", 0, [C, D], "no", "Best-of-N display-only; no candidate groups or logs"),
    ("glean", "secondary_empirical", 0, [C, D], "no", "Clinician study: no ratings, instrument or analysis code"),
    ("globalhealthatlas", "data_protocol", 1, [R], "no", "280,210-row corpus and source inventory not in artifacts"),
    ("globalhealthatlas", "data_protocol", 1, [R], "no", "Row-level corpus and 14,010 expert-audit rows absent"),
    ("globalhealthatlas", "secondary_empirical", 0, [R, D], "no", "Expert labels, evaluator predictions, repeated-run matrices absent"),
    ("globalhealthatlas", "headline_performance", 2, [], "no", "Table 5 reconstructed from 16 per-model CSVs"),
    ("globalhealthatlas", "secondary_empirical", 1, [X, O], "yes", "train_lora.sh not the paper recipe; GPQA value mismatch"),
    ("globalhealthatlas", "secondary_empirical", 1, [R, D], "no", "Robustness from cached outputs; raw questions and predictions absent"),
    ("lvcg", "mechanism", 1, [C], "no", "Geometry module import fails: lvcg.data absent from history"),
    ("lvcg", "mechanism", 0, [C, X], "yes", "Training entrypoint fails; config adds base_beat_loss not in Eq. 7"),
    ("lvcg", "headline_performance", 1, [C, W, D], "no", "Splits disjoint; checkpoint override bug, label column misorder, no checkpoints"),
    ("lvcg", "headline_performance", 0, [W, D], "no", "No checkpoint or reconstruction outputs for Table 2"),
    ("lvcg", "secondary_empirical", 0, [C, W, D], "no", "No ablation configs, checkpoints, seeds or outputs"),
    ("lvcg", "secondary_empirical", 0, [X, C], "yes", "No MIMIC-IV-ECG path; repository configures AI-READI instead"),
    ("ma_rag", "mechanism", 1, [X], "yes", "History sorted ascending vs Eq. 4 descending; epsilon threshold absent"),
    ("ma_rag", "data_protocol", 1, [X, O], "yes", "Runner forces single_answer=True for multi-answer NEJM"),
    ("ma_rag", "headline_performance", 0, [R, W, D, O], "no", "MedCorp index, evaluator checkpoint, predictions, credentials absent"),
    ("ma_rag", "headline_performance", 0, [R, W, D, O], "no", "Baseline margins not rerun; release omits corpus and predictions"),
    ("ma_rag", "secondary_empirical", 0, [R, W, D], "no", "Cumulative ablations not rerun"),
    ("ma_rag", "secondary_empirical", 0, [R, D, O], "no", "Four-run variance and timing: hardware traces absent"),
    ("medcrp_cl", "mechanism", 1, [X, D], "yes", "Released state uses CRP alpha 2.0, paper reports 5.0; prompts unavailable"),
    ("medcrp_cl", "mechanism", 2, [], "no", "Modality-isolated LoRA and EWC aggregates verified"),
    ("medcrp_cl", "headline_performance", 0, [D], "no", "No processed splits, run outputs, seeds, metric tables"),
    ("medcrp_cl", "secondary_empirical", 0, [D], "no", "No order, perturbation, ablation or merge outputs"),
    ("medcrp_cl", "secondary_empirical", 1, [D, O], "no", "8.6M count not recomputed; no clinical predictions"),
    ("ophiuchus", "mechanism", 1, [X, C], "yes", "Tool-call syntax not parsed; duplicate calls do not early-stop, contrary to paper"),
    ("ophiuchus", "mechanism", 1, [X, C], "yes", "Preprocess crashes; ATRL KL setting conflicts with paper"),
    ("ophiuchus", "data_protocol", 0, [R], "no", "Only two SFT and two RL records of 64k corpus released"),
    ("ophiuchus", "headline_performance", 0, [W, D], "no", "Checkpoint and complete evaluation release absent"),
    ("ophiuchus", "secondary_empirical", 0, [W, R, D], "no", "Checkpoint, datasets, predictions, ablations absent"),
    ("spmind", "mechanism", 1, [O], "no", "Loop delegated to Claude Agent SDK; novelty claim unverified"),
    ("spmind", "headline_performance", 0, [C, D, O], "no", "Model retired, no API key, traces unpublished, no success-criteria code"),
    ("spmind", "data_protocol", 2, [], "no", "SP-Bench JSONL validated independently"),
    ("spmind", "secondary_empirical", 0, [R, D], "no", "CRC-CODEX images, outputs and ground truth not available"),
    ("spmind", "secondary_empirical", 0, [D], "no", "Annotation predictions absent"),
    ("sprout", "mechanism", 1, [C], "no", "Canonical pipeline not executable: POT error, empty dataset link"),
    ("sprout", "mechanism", 2, [], "no", "Reference-mask branch executes on synthetic H&E image"),
    ("sprout", "theory", 1, [X, C, O], "yes", "Solver mass 2.4 vs paper-normalized 0.6; convexity counterexample"),
    ("sprout", "mechanism", 1, [C], "no", "Soft-NMS works; crashes on empty proposal set"),
    ("sprout", "headline_performance", 0, [W, D], "no", "No predictions, splits, checkpoints or metric tables"),
    ("sprout", "secondary_empirical", 0, [D], "no", "No repeat outputs or runtime traces"),
    ("xmae", "mechanism", 1, [X], "yes", "Canonical config disables curriculum; ladder never advanced"),
    ("xmae", "mechanism", 1, [C], "no", "Checkpoint yields all-NaN embeddings in eval mode"),
    ("xmae", "headline_performance", 0, [W, R], "no", "Pretrained weights and evaluated datasets unreleased"),
    ("xmae", "data_protocol", 0, [R], "no", "Protected waveform cohort and manifest absent"),
    ("xmae", "secondary_empirical", 0, [W, R], "no", "Real weights and held-out paired data absent"),
    ("xmae", "secondary_empirical", 0, [R, D], "no", "Six evaluation datasets and split/probe outputs absent"),
]


def main():
    with open("papers_coder1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow("folder,orid,healthcare,forecast_score,forecast_max,prepared_score,prepared_max,code_release".split(","))
        w.writerows(PAPERS)
    count, total = {}, {}
    with open("claims_coder1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow("folder,claim_no,claim_type,score,max,outcome,blockers,primary_blocker,divergence,pointer".split(","))
        for folder, ctype, score, blockers, div, pointer in CLAIMS:
            count[folder] = count.get(folder, 0) + 1
            total[folder] = total.get(folder, 0) + score
            outcome = "full" if score == 2 else ("none" if score == 0 else "partial")
            assert (outcome == "full") == (not blockers), (folder, count[folder])
            primary = next((b for b in PRECEDENCE if b in blockers), "none")
            w.writerow([folder, count[folder], ctype, score, 2, outcome, ";".join(blockers), primary, div, pointer])
    for folder, _, _, _, _, prepared, pmax, _ in PAPERS:
        assert total[folder] == prepared and 2 * count[folder] == pmax, (folder, total[folder], count[folder])
    print("ok", len(PAPERS), "papers", len(CLAIMS), "claims")


if __name__ == "__main__":
    main()
