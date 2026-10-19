"""C00126 (mechanism): TPE + dependency-aware replacement (~3 percent least-common tokens)
+ SSFT of only the new-token embeddings (0.5-1.0 percent of parameters).

Runs inside the container (python 3.12, CPU torch). All data are synthetic
(checks/gen_synthetic.py). Stages (each is a separate container run):

  prep       pretrain the tiny LM on synthetic text, sample pseudo-labels (T=0.7), write SFT_QA parquet
  tokenizer  repo: preprocessing/create_tpe_tokenizer.py
  fit        repo: preprocessing/llm_fit_tpe_tokenizer.py
  clean      repo: preprocessing/clean_sft_QA.py (train, tuning)
  sft        repo: llm_sft.py via checks/run_sft_cpu_patched.py   (arg: variant)
  measure    S1-S4 assertions -> work/logs/C00126_summary.json

Pass/fail rules are those of checks/plan.json (version 2).
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/work/work")
REPO = ROOT / "repo"
DATA = ROOT / "data"
RUN = ROOT / "run"
CHECKS = Path("/work/checks")
MODEL_DIR = DATA / "hf_models" / "Qwen--Qwen2.5-1.5B-Instruct"
MODEL_NAME = "Qwen2.5-1.5B-Instruct"
TASK, DATASET = "cmedqa2", "CMEDQA2"
M, N = 200, 2
DSDIR = "CMEDQA2_max-in-4k_max-out-2k"
MT = DATA / "MedTPE_data"
SFT_QA = MT / "SFT_QA"
CLEAN = MT / "cleaned_SFT_QA"
TOK_ROOT = MT / "tpe_tokenizers"
TPE_ROOT = MT / "tpe_models"
VARIANT_DIRS = {}
TAG = f"{MODEL_NAME}_task-{TASK}_maxN-{N}_maxM-{M}"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CHECKS))


def sh(cmd, cwd=RUN, name=None):
    RUN.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    t0 = time.time()
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       env={**os.environ, "PYTHONPATH": str(REPO), "WANDB_MODE": "offline"})
    out = p.stdout
    if name:
        Path(f"/work/work/logs/C00126_full_{name}.log").write_text(out, encoding="utf-8")
    tail = out if len(out) < 6000 else out[:1500] + "\n...[truncated]...\n" + out[-4000:]
    print(tail, flush=True)
    print(f"[exit {p.returncode}, {time.time() - t0:.1f}s]", flush=True)
    if p.returncode != 0:
        raise SystemExit(f"command failed: {cmd}")
    return out


# ------------------------------------------------------------------------------------------
def stage_prep():
    import numpy as np
    import polars as pl
    import torch
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    from dataset.map_fn import cmedqa2_sample_mapping_fn, load_prompt
    import gen_synthetic as gs

    torch.set_num_threads(4)
    torch.manual_seed(1)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = Qwen2ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    lex = gs.Lexicon()
    if (MODEL_DIR / "PRETRAINED.txt").exists():
        print("[prep] pretrained weights found (" + (MODEL_DIR / "PRETRAINED.txt").read_text().strip() + "); skipping pretraining", flush=True)
        model.eval()
    else:
        rs = np.random.RandomState(5)  # pretraining corpus seed, disjoint from train(1)/val(2)/test(3) seeds
        prompt_tpl = load_prompt("cmedqa2")["prompt"]
        docs = []
        for _ in range(3000):
            q, first = gs.ehr_text(lex, rs, 16, return_first=True)
            ans = "Assessment: " + "; ".join(first) + "."
            docs.append(tok.apply_chat_template(
                [{"role": "user", "content": prompt_tpl.format(question=q)}, {"role": "assistant", "content": ans}],
                tokenize=False))
        for _ in range(1000):
            docs.append(gs.general_text(lex, rs, 250))
        ids = []
        for d in docs:
            ids.extend(tok(d, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
        ids = torch.tensor(ids, dtype=torch.long)
        L, B = 384, 8
        nseq = len(ids) // L
        data = ids[: nseq * L].view(nseq, L)
        print(f"[prep] pretraining tokens={len(ids)} sequences={nseq} (L={L})", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.01)
        model.train()
        g = torch.Generator().manual_seed(1)

        def batch():
            idx = torch.randint(0, nseq, (B,), generator=g)
            x = data[idx]
            return x, x.clone()

        # benchmark: 2 warm-up steps (discarded) then 4 timed steps; schedule length from a 540 s cap
        for k in range(6):
            if k == 2:
                t0 = time.time()
            x, y = batch()
            loss = model(input_ids=x, labels=y).loss
            loss.backward()
            opt.zero_grad()
        step_t = (time.time() - t0) / 4
        total = int(max(100, min(1500, 540 / step_t)))
        print(f"[prep] step_time={step_t:.2f}s -> pretraining steps={total}", flush=True)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
        first_loss = None
        for s in range(total):
            x, y = batch()
            loss = model(input_ids=x, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            if first_loss is None:
                first_loss = loss.item()
            if s % 50 == 0 or s == total - 1:
                print(f"[prep] step {s} loss {loss.item():.3f}", flush=True)
        model.eval()
        model.save_pretrained(MODEL_DIR)
        print(f"[prep] pretraining loss {first_loss:.3f} -> {loss.item():.3f}", flush=True)
        (MODEL_DIR / "PRETRAINED.txt").write_text(f"pretraining loss {first_loss:.3f} -> {loss.item():.3f}, steps={total}" + chr(10))

    # ---- pseudo-labels: sample the original tiny LM at temperature 0.7 (LLMConfig.temperature) ----
    tok.padding_side = "left"
    out_dir = SFT_QA / DSDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = {"train": ("train", 1000), "tuning": ("validation", 100), "held_out": ("test", 100)}
    torch.manual_seed(1)
    exact = {}
    for split, (fname, n) in plan.items():
        recs = json.load(open(DATA / "medtpe_data" / "cmedqa2" / f"{fname}.json", encoding="utf-8"))[:n]
        batch_in = {"question": [r["question"] for r in recs], "answer": [r["answer"] for r in recs]}
        mapped = cmedqa2_sample_mapping_fn(batch_in, instruction=None, llm_tokenizer=tok, max_input_length=4096)
        messages, labels = mapped["message"], mapped["label"]
        gens = []
        for i in range(0, len(messages), 16):
            texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages[i:i + 16]]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            with torch.no_grad():
                out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], do_sample=True, temperature=0.7, top_p=1.0, top_k=0, max_new_tokens=64,
                                     eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            for row in out[:, enc["input_ids"].shape[1]:]:
                row = row.tolist()
                if tok.eos_token_id in row:
                    row = row[: row.index(tok.eos_token_id)]
                gens.append(tok.decode(row, skip_special_tokens=True))
        exact[split] = float(np.mean([g_ == l_ for g_, l_ in zip(gens, labels)]))
        df = pl.DataFrame({"message": messages, "label": labels, "generated_text_0": gens})
        fpath = out_dir / f"{TASK}_nl_{MODEL_NAME}_set-{split}.parquet"
        df.write_parquet(fpath)
        print(f"[prep] wrote {fpath.name} rows={len(df)} exact-match-with-true-answer={exact[split]:.2f}", flush=True)
        print(f"[prep] example pseudo-label: {gens[0]!r}", flush=True)


def stage_tokenizer():
    sh([sys.executable, REPO / "preprocessing/create_tpe_tokenizer.py", "--model_path", MODEL_DIR,
        "--data_dir", SFT_QA, "--log_dir", TOK_ROOT, "--dataset", DATASET, "--task", TASK,
        "--max_m", M, "--max_n", N, "--data_format", "nl"])


def stage_fit():
    sh([sys.executable, REPO / "preprocessing/llm_fit_tpe_tokenizer.py", "--model_path", MODEL_DIR,
        "--tokenizer_dir", TOK_ROOT, "--log_dir", TPE_ROOT, "--dataset", DATASET, "--task", TASK,
        "--max_m", M, "--max_n", N, "--data_format", "nl"])


def stage_clean():
    for split in ["train", "tuning"]:
        sh([sys.executable, REPO / "preprocessing/clean_sft_QA.py", "--model_path", MODEL_DIR,
            "--data_dir", SFT_QA, "--tpe_tokenizer_dir", TOK_ROOT, "--output_dir", CLEAN, "--dataset", DATASET,
            "--task", TASK, "--data_split", split, "--num_workers", 1, "--max_m", M, "--max_n", N])


def stage_sft(variant, epochs, lr):
    # 'shipped' (bf16 + real Adam8bit) is only run at small scale (see plan.json changed_after_run)
    data_dir = MT / "cleaned_SFT_QA_small" if variant == "shipped" else CLEAN
    logdir = MT / f"sft_{variant}"
    sh([sys.executable, CHECKS / "run_sft_cpu_patched.py", variant,
        "--model_path", TPE_ROOT / TAG, "--task", TASK, "--data_dir", data_dir, "--log_dir", logdir,
        "--dataset", DATASET, "--data_format", "nl", "--max_length", 512, "--num_proc", 1,
        "--batch_size", 2, "--grad_accum_steps", 2, "--lr", lr, "--num_train_epochs", epochs,
        "--val_steps", 250, "--wandb_mode", "offline", "--gpu_ids", 0], name=f"sft_{variant}")


# ------------------------------------------------------------------------------------------
def boot_ci(diff, n=10000, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    diff = np.asarray(diff)
    means = np.array([diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(n)])
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stage_measure():
    import numpy as np
    import polars as pl
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, AutoModelForCausalLM

    from dataset.map_fn import tpe_sample_batch_mapping_fn, dataset_to_Dtok_batch_mapping_fn
    from tpe.tpe_tokenizer_fast import TPETokenizerFast
    from tpe.vocab_modify_tpe import (count_old_token_frequencies, prepare_n_tokens, cnt_freqs)
    from datasets import Dataset

    torch.set_num_threads(4)
    res = {"S1": {}, "S2": {}, "S3": {}, "S4": {}}
    tokdir = TOK_ROOT / TAG
    tpedir = TPE_ROOT / TAG
    info = json.load(open(tokdir / "new_token_info.json"))
    base_tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tpe_tok = TPETokenizerFast.from_pretrained(tpedir)
    base_vocab = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json")).get_vocab(with_added_tokens=True)
    new_vocab = Tokenizer.from_file(str(tokdir / "tokenizer.json")).get_vocab(with_added_tokens=True)
    base_json = json.load(open(MODEL_DIR / "tokenizer.json"))
    merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in base_json["model"]["merges"]]
    cfg = json.load(open(MODEL_DIR / "config.json"))
    d_model = cfg["hidden_size"]

    # ------------------------------ S1: TPE lossless token-pair merge ------------------------------
    n_new = len(info)
    s1_pairs_ok = all(len(v["sub_tokens"]) == 2 and "".join(v["sub_tokens"]) == k for k, v in info.items())
    texts = [r["question"] for r in json.load(open(DATA / "medtpe_data" / "cmedqa2" / "test.json", encoding="utf-8"))][:600]
    n_rt_ok, len_base, len_tpe = 0, 0, 0
    rt_bad = []
    expand_viol = []
    evicted = set(base_vocab) - set(new_vocab)
    newtoks = set(info.keys())
    n_diverge, n_diverge_explained, n_diverge_direct, n_diverge_via_path = 0, 0, 0, 0
    # multi-valued merge map (independent of the repo's single-valued dict) and BPE merge-path closure
    mm = {}
    for a, b in merges:
        mm.setdefault(a + b, []).append((a, b))

    def comps(t, seen=None):
        seen = set() if seen is None else seen
        if t in seen:
            return seen
        seen.add(t)
        for a, b in mm.get(t, []):
            comps(a, seen)
            comps(b, seen)
        return seen

    # tokens whose recursive BPE derivation passes through an evicted token (plan v2 (vi), amended before any run)
    tainted = {t for t in base_vocab if t not in evicted and (comps(t) & evicted)}
    for i, t in enumerate(texts):
        ids = tpe_tok(t, add_special_tokens=False)["input_ids"]
        dec = tpe_tok.decode(ids)
        if dec == t:
            n_rt_ok += 1
        elif len(rt_bad) < 3:
            rt_bad.append(i)
        b_ids = base_tok(t, add_special_tokens=False)["input_ids"]
        len_base += len(b_ids)
        len_tpe += len(ids)
        b_toks = base_tok.convert_ids_to_tokens(b_ids)
        m_toks = [tpe_tok.convert_ids_to_tokens(j) for j in ids]
        expanded = []
        for tkn in m_toks:
            expanded.extend(info[tkn]["sub_tokens"] if tkn in newtoks else [tkn])
        if expanded != b_toks:
            n_diverge += 1
            direct = any(tk in evicted for tk in b_toks)
            via = any(tk in tainted for tk in b_toks)
            if direct or via:
                n_diverge_explained += 1
                n_diverge_direct += int(direct)
                n_diverge_via_path += int((not direct) and via)
            else:
                expand_viol.append(i)
    cr = 1 - len_tpe / len_base
    res["S1"] = {"n_new_tokens": n_new, "all_new_tokens_are_2_token_merges": s1_pairs_ok,
                 "n_texts": len(texts), "roundtrip_ok": n_rt_ok, "roundtrip_bad_examples": rt_bad,
                 "CR": cr, "base_tokens": len_base, "medtpe_tokens": len_tpe}
    res["S1"]["pass"] = bool(s1_pairs_ok and n_rt_ok == len(texts) and cr > 0 and n_new > 0)
    print(f"[S1] new_tokens={n_new} 2-token-merges={s1_pairs_ok} roundtrip={n_rt_ok}/{len(texts)} CR={cr:.4f} "
          f"(base {len_base} -> MedTPE {len_tpe} tokens)  => {'PASS' if res['S1']['pass'] else 'FAIL'}", flush=True)

    # ------------------------------ S2: dependency-aware replacement ------------------------------
    sd = Dataset.from_parquet(str(SFT_QA / DSDIR / f"{TASK}_nl_{MODEL_NAME}_set-train.parquet"))
    sd = sd.map(tpe_sample_batch_mapping_fn, batched=True, remove_columns=["message", "generated_text_0"])
    sd = sd.map(dataset_to_Dtok_batch_mapping_fn, fn_kwargs={"tokenizer": base_tok, "filter_numbers": True},
                batched=True, batch_size=1000)
    Dtok = [x["token"] for x in sd]
    vocab_old = {k: v for k, v in base_vocab.items() if v < len(base_tok)}
    freq = count_old_token_frequencies(Dtok, vocab_old, disable_progress=True)
    ngr = prepare_n_tokens(Dtok, N)
    Fn = cnt_freqs(Dtok, ngr, N)
    cand = [t for t in ngr if Fn[t] >= 2]
    D_paper = set()
    for k, v in info.items():
        for st in v["sub_tokens"]:
            D_paper |= comps(st)
    D_repo = set()
    for ng in cand:
        sub = ng.split(" ")
        if all(s in vocab_old for s in sub):
            for s in sub:
                D_repo |= comps(s)
    special = set(base_tok.all_special_tokens) | set(base_tok.added_tokens_encoder.keys())
    ids_old, ids_new = sorted(base_vocab.values()), sorted(new_vocab.values())
    sizes_ok = len(new_vocab) == len(base_vocab) and ids_old == ids_new
    count_ok = n_new == M
    ev_special = sorted(evicted & special)
    ev_vs_paper = sorted(evicted & D_paper)
    ev_vs_repo = sorted(evicted & D_repo)
    sub_missing = sorted({s for v in info.values() for s in v["sub_tokens"] if s not in new_vocab})
    eligible_retained = [t for t in vocab_old if t in new_vocab and t not in newtoks and t not in special
                         and len(t) > 1 and t not in D_repo]
    ev_freq_max = max(freq.get(t, 0) for t in evicted) if evicted else 0
    ret_freq_min = min(freq.get(t, 0) for t in eligible_retained)
    least_common_ok = ev_freq_max <= ret_freq_min
    paper_vs_repo_rarer = sum(1 for t in vocab_old if t in new_vocab and t not in newtoks and t not in D_paper
                              and t not in special and len(t) > 1 and freq.get(t, 0) < ev_freq_max)
    frac_syn = n_new / len(base_vocab)
    qcfg = json.load(open(ROOT / "configs" / "Qwen2.5-1.5B-Instruct.json"))
    frac_real = 5000 / qcfg["vocab_size"]
    frac_ok = 0.025 <= frac_syn <= 0.035 and 0.025 <= frac_real <= 0.035
    res["S2"] = {"vocab_size_old": len(base_vocab), "vocab_size_new": len(new_vocab), "id_sets_equal": ids_old == ids_new,
                 "n_new": n_new, "M": M, "n_evicted": len(evicted), "evicted_special": ev_special,
                 "evicted_in_D_paper": ev_vs_paper, "evicted_in_D_repo": ev_vs_repo, "sub_tokens_missing_in_Vstar": sub_missing,
                 "max_freq_evicted": ev_freq_max, "min_freq_retained_eligible": ret_freq_min,
                 "n_evicted_with_freq0": sum(1 for t in evicted if freq.get(t, 0) == 0),
                 "retained_outside_D_paper_but_rarer_than_rarest_evicted (info)": paper_vs_repo_rarer,
                 "|D_paper|": len(D_paper), "|D_repo|": len(D_repo),
                 "tokenisation_integrity_texts_diverging": n_diverge, "diverging_explained_by_evicted_token_or_its_descendants": n_diverge_explained,
                 "diverging_with_directly_evicted_token": n_diverge_direct, "diverging_only_via_merge_path_of_evicted": n_diverge_via_path,
                 "n_tainted_retained_tokens": len(tainted),
                 "diverging_unexplained": len(expand_viol),
                 "fraction_replaced_synthetic": frac_syn, "fraction_replaced_real_qwen2.5_5000_over_vocab_size": frac_real}
    res["S2"]["pass"] = bool(sizes_ok and count_ok and not ev_special and not ev_vs_paper and not ev_vs_repo
                             and not sub_missing and least_common_ok and not expand_viol and frac_ok)
    print(f"[S2] |V|={len(base_vocab)} |V*|={len(new_vocab)} ids_equal={ids_old == ids_new} new={n_new}/{M} evicted={len(evicted)} "
          f"evicted_special={len(ev_special)} E∩D_paper={len(ev_vs_paper)} E∩D_repo={len(ev_vs_repo)} "
          f"sub_tokens_missing={len(sub_missing)} maxfreq(E)={ev_freq_max} minfreq(retained eligible)={ret_freq_min} "
          f"texts_diverging={n_diverge} (explained {n_diverge_explained}, unexplained {len(expand_viol)}) "
          f"frac_syn={frac_syn:.4f} frac_real={frac_real:.4f} info_paper_vs_repo_rarer={paper_vs_repo_rarer}  "
          f"=> {'PASS' if res['S2']['pass'] else 'FAIL'}", flush=True)

    # ------------------------------ S3: SSFT updates only new rows; parameter fraction ------------------------------
    pre = Qwen2ForCausalLM.from_pretrained(tpedir, torch_dtype=torch.float32)
    new_ids = [v["token_id"] for v in info.values()]
    total_full = sum(p.numel() for p in pre.parameters())
    res["S3"]["tiny_model"] = {"d": d_model, "M_new": len(new_ids), "total_params_full": total_full,
                               "tied": pre.config.tie_word_embeddings,
                               "M*d/total_full": len(new_ids) * d_model / total_full}
    variants = {}
    for v in ["shipped", "adamw", "adamw_fp32", "minimal_fp32"]:
        p = MT / f"sft_{v}" / "sft_models" / TAG
        if (p / "config.json").exists():
            variants[v] = p
    res["S3"]["variants_found"] = list(variants)
    pre_sd = pre.state_dict()
    for v, p in variants.items():
        post = Qwen2ForCausalLM.from_pretrained(p, torch_dtype="auto")
        post_sd = post.state_dict()
        mism, new_changed, rows_checked = [], 0, 0
        dtype = next(iter(post_sd.values())).dtype
        for k, w in pre_sd.items():
            if k not in post_sd:
                mism.append(f"missing:{k}")
                continue
            w2 = w.to(dtype)
            if k in ("model.embed_tokens.weight", "lm_head.weight"):
                mask = torch.ones(w2.shape[0], dtype=torch.bool)
                mask[new_ids] = False
                if not torch.equal(w2[mask], post_sd[k][mask]):
                    mism.append(k + "[non-new rows]")
                ch = (w2[new_ids] != post_sd[k][new_ids]).any(dim=1)
                if k == "model.embed_tokens.weight":
                    new_changed = int(ch.sum())
                    maxdiff = float((w2[new_ids].float() - post_sd[k][new_ids].float()).abs().max())
            elif not torch.equal(w2, post_sd[k]):
                mism.append(k)
        # repo-printed trainable parameter line
        logtxt = open(f"/work/work/logs/C00126_full_sft_{v}.log", encoding="utf-8", errors="ignore").read()
        mm_ = re.findall(r"\[Embedding\] trainable params: (\d+)/(\d+) = ([\d.]+)%", logtxt)
        res["S3"][v] = {"dtype": str(dtype), "non_new_tensors_differing": mism, "new_rows_changed": new_changed,
                        "new_rows_total": len(new_ids), "max_abs_change_new_rows": maxdiff,
                        "repo_printed_trainable_lines": mm_[-3:]}
        print(f"[S3:{v}] dtype={dtype} frozen-tensors-differing={mism} new_rows_changed={new_changed}/{len(new_ids)} "
              f"max|delta|={maxdiff:.4g}", flush=True)
    # repo-printed numerator check (log lines are per run; check each ends with X == M'*d)
    printed = [t for v in variants for t in res["S3"][v]["repo_printed_trainable_lines"]]
    numer_ok = bool(printed) and all(int(x) == len(new_ids) * d_model for x, _, _ in printed)
    res["S3"]["printed_numerator_equals_M*d"] = numer_ok
    # real architectures from public config.json (meta device, no weights)
    real = {}
    for name, M_real in [("Qwen2.5-1.5B-Instruct", 5000), ("Qwen2.5-7B-Instruct", 5000)]:
        qc = Qwen2Config.from_json_file(str(ROOT / "configs" / f"{name}.json"))
        with torch.device("meta"):
            mm_model = Qwen2ForCausalLM(qc)
        tot = sum(p.numel() for p in mm_model.parameters())
        emb = qc.vocab_size * qc.hidden_size
        trainable = M_real * qc.hidden_size
        real[name] = {"total_params": tot, "tied": qc.tie_word_embeddings, "d": qc.hidden_size, "vocab_size": qc.vocab_size,
                      "trainable_unique(M*d)": trainable,
                      "pct_of_total(M*d/total)": 100 * trainable / tot,
                      "pct_in_plus_out_rows(2*M*d/total) if untied": 100 * 2 * trainable / tot,
                      "pct_repo_style_denominator(excl. frozen embedding matrix)": 100 * trainable / (tot - emb + trainable)}
    res["S3"]["real_architectures"] = real
    pct = real["Qwen2.5-1.5B-Instruct"]["pct_of_total(M*d/total)"]
    pct_rounded = round(pct, 1)
    frozen_ok = all(not res["S3"][v]["non_new_tensors_differing"] for v in variants)
    moved_ok = any(res["S3"][v]["new_rows_changed"] > 0 for v in variants)
    res["S3"]["qwen1.5B_pct_rounded"] = pct_rounded
    res["S3"]["pass"] = bool(variants and frozen_ok and moved_ok and numer_ok and 0.5 <= pct_rounded <= 1.0)
    print(f"[S3] frozen_ok={frozen_ok} new_rows_moved_in_some_variant={moved_ok} printed_numerator==M*d:{numer_ok} "
          f"Qwen2.5-1.5B trainable={pct:.4f}% (rounded {pct_rounded}) "
          f"7B untied: {real['Qwen2.5-7B-Instruct']['pct_of_total(M*d/total)']:.3f}% (M*d) / "
          f"{real['Qwen2.5-7B-Instruct']['pct_in_plus_out_rows(2*M*d/total) if untied']:.3f}% (2*M*d) "
          f"=> {'PASS' if res['S3']['pass'] else 'FAIL'}", flush=True)

    # ------------------------------ S4: alignment (BPC of pseudo-labels, hidden-state cosine) ------------------------------
    held = pl.read_parquet(SFT_QA / DSDIR / f"{TASK}_nl_{MODEL_NAME}_set-held_out.parquet")
    msgs, gens = held["message"].to_list(), held["generated_text_0"].to_list()
    post_txt = "<|im_end|>\n"
    prompts = [base_tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]

    def load_bf16_round(path):
        m = Qwen2ForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        m = m.to(torch.bfloat16).to(torch.float32)
        return m.eval()

    @torch.no_grad()
    def score(model, tok):
        bpc, hid, plen = [], [], []
        for pr, g in zip(prompts, gens):
            p_ids = tok(pr, add_special_tokens=False)["input_ids"]
            a_str = g + post_txt
            a_ids = tok(a_str, add_special_tokens=False)["input_ids"]
            x = torch.tensor([p_ids + a_ids])
            out = model(input_ids=x, output_hidden_states=True)
            logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
            tgt = x[0, 1:]
            sel = logp[torch.arange(len(tgt)), tgt][len(p_ids) - 1:]
            nll = -sel.sum().item()
            bpc.append(nll / math.log(2) / len(a_str))
            hid.append(out.hidden_states[-1][0, len(p_ids) - 1].clone())  # last layer, post final norm
            plen.append(len(p_ids))
        return np.array(bpc), torch.stack(hid), np.array(plen)

    m_a = load_bf16_round(MODEL_DIR)
    bpc_a, h_a, pl_a = score(m_a, base_tok)
    del m_a
    m_b = load_bf16_round(tpedir)
    bpc_b, h_b, pl_b = score(m_b, tpe_tok)
    del m_b
    cos_b = torch.nn.functional.cosine_similarity(h_a, h_b, dim=-1).numpy()
    s4 = {"n_prompts": len(prompts), "prompt_tokens_base_mean": float(pl_a.mean()), "prompt_tokens_medtpe_mean": float(pl_b.mean()),
          "BPC_original_mean": float(bpc_a.mean()), "BPC_medtpe_before_mean": float(bpc_b.mean()),
          "cos_before_mean": float(cos_b.mean()), "variants": {}}
    for v, p in variants.items():
        m_c = load_bf16_round(p)
        bpc_c, h_c, _ = score(m_c, tpe_tok)
        del m_c
        cos_c = torch.nn.functional.cosine_similarity(h_a, h_c, dim=-1).numpy()
        dm, dl, dh = boot_ci(bpc_c - bpc_b)
        cm, cl, ch = boot_ci(cos_c - cos_b)
        gap = (bpc_b.mean() - bpc_c.mean()) / max(bpc_b.mean() - bpc_a.mean(), 1e-12)
        s4["variants"][v] = {"BPC_after_mean": float(bpc_c.mean()), "BPC_diff_after_minus_before": [dm, dl, dh],
                             "gap_closure": float(gap), "cos_after_mean": float(cos_c.mean()),
                             "cos_diff_after_minus_before": [cm, cl, ch]}
        print(f"[S4:{v}] BPC original={bpc_a.mean():.4f} medtpe-before={bpc_b.mean():.4f} medtpe-after={bpc_c.mean():.4f} "
              f"diff(after-before)={dm:.4f} CI[{dl:.4f},{dh:.4f}] gap_closure={gap:.3f} | cos before={cos_b.mean():.4f} "
              f"after={cos_c.mean():.4f} diff={cm:.4f} CI[{cl:.4f},{ch:.4f}]", flush=True)
    # judged variant (plan.json changed_after_run): the largest-scale variant that completed = adamw_fp32
    judged = "minimal_fp32" if "minimal_fp32" in variants else ("adamw_fp32" if "adamw_fp32" in variants else ("shipped" if "shipped" in variants else None))
    s4["judged_variant"] = judged
    if judged:
        dm, dl, dh = s4["variants"][judged]["BPC_diff_after_minus_before"]
        s4["pass"] = bool(dh < 0)
    else:
        s4["pass"] = False
    res["S4"] = s4
    print(f"[S4] judged variant={judged}: BPC diff CI upper bound {dh if judged else None} < 0 => {'PASS' if s4['pass'] else 'FAIL'}", flush=True)

    res["overall_all_pass"] = bool(all(res[k].get("pass") for k in ["S1", "S2", "S3", "S4"]))
    json.dump(res, open("/work/work/logs/C00126_summary.json", "w"), indent=1, default=str)
    print("[SUMMARY]", json.dumps({k: res[k].get("pass") for k in ["S1", "S2", "S3", "S4"]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "tokenizer", "fit", "clean", "sft", "measure"])
    ap.add_argument("variant", nargs="?", default="shipped")
    ap.add_argument("epochs", nargs="?", type=int, default=10)
    ap.add_argument("lr", nargs="?", default="5e-5")
    a = ap.parse_args()
    print(f"==== C00126 stage={a.stage} {time.strftime('%H:%M:%S')} ====", flush=True)
    {"prep": stage_prep, "tokenizer": stage_tokenizer, "fit": stage_fit, "clean": stage_clean,
     "measure": stage_measure}.get(a.stage, lambda: stage_sft(a.variant, a.epochs, a.lr))()
