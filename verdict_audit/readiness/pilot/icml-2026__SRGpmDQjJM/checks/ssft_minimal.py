"""Minimal fp32 SSFT loop that uses the repository's embedding-split module and data mapping.

Why: the repository's own trainer (llm_sft.py -> run_fn/llm_sft_fn.py: accelerate + bf16 autocast +
gradient checkpointing + bitsandbytes Adam8bit) was started three times on this shared CPU and ran at
about 0.1 micro-step/s (see plan.json changed_after_run); it could not finish inside the 2 h budget.
Repo code used here: dataset.map_fn.sft_map_postfix_fn, run_fn.set_embed_fn.set_trainable_embeddings /
restore_embeddings, tpe.tpe_tokenizer_fast.TPETokenizerFast, utils.constants.generation_postfixs.
Objective, data, learning-rate schedule shape (10% warm-up + cosine) follow llm_sft_fn.py; lr is 5e-4 (see plan).
"""
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

sys.path.insert(0, "/work/work/repo")
from dataset.map_fn import sft_map_postfix_fn
from run_fn.set_embed_fn import set_trainable_embeddings, restore_embeddings
from tpe.tpe_tokenizer_fast import TPETokenizerFast
from utils.constants import generation_postfixs

MT = Path("/work/work/data/MedTPE_data")
TAG = "Qwen2.5-1.5B-Instruct_task-cmedqa2_maxN-2_maxM-200"
DS = "CMEDQA2_max-in-4k_max-out-2k"
UPDATES, BATCH, LR, CAP_S, MAXLEN = 200, 4, 5e-4, 330, 512
torch.set_num_threads(4)
torch.manual_seed(1)
tok = TPETokenizerFast.from_pretrained(MT / "tpe_models" / TAG)
model = AutoModelForCausalLM.from_pretrained(MT / "tpe_models" / TAG, use_cache=False, torch_dtype=torch.float32)
ds = Dataset.from_parquet(str(MT / "cleaned_SFT_QA" / DS / "cmedqa2_nl_Qwen2.5-1.5B-Instruct_set-train.parquet"),
                          columns=["message", "generated_text_0"])
ds = ds.map(sft_map_postfix_fn, fn_kwargs={"tokenizer": tok, "max_seq_length": MAXLEN,
                                           "generation_postfix": generation_postfixs["Qwen2.5-1.5B-Instruct"]},
            batched=True, batch_size=1000, remove_columns=["message", "generated_text_0"])
new_ids = [v["token_id"] for v in json.load(open(MT / "tpe_models" / TAG / "new_token_info.json")).values()]
model = set_trainable_embeddings(model, new_ids, freeze_params=True)   # prints '[Embedding] trainable params: X/Y = Z%'
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR)
sched = get_cosine_schedule_with_warmup(opt, int(0.1 * UPDATES), UPDATES)
g = torch.Generator().manual_seed(1)
perm = torch.randperm(len(ds), generator=g).tolist()
model.train()
t0 = time.time()
losses = []
for u in range(UPDATES):
    idx = [perm[(u * BATCH + k) % len(perm)] for k in range(BATCH)]
    b = ds[idx]
    ids, att, lab = (torch.tensor(b[k]).long() for k in ("input_ids", "attention_mask", "labels"))
    L = int(att.sum(dim=1).max())
    out = model(input_ids=ids[:, :L], attention_mask=att[:, :L], labels=lab[:, :L])
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step(); opt.zero_grad()
    losses.append(out.loss.item())
    if u % 10 == 0:
        print(f"[ssft] update {u} loss {out.loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} t={time.time() - t0:.0f}s", flush=True)
    if time.time() - t0 > CAP_S:
        print(f"[ssft] wall-clock cap reached after {u + 1} updates", flush=True)
        break
n = len(losses)
print(f"[ssft] updates={n} mean loss first10={sum(losses[:10]) / min(10, n):.4f} last10={sum(losses[-10:]) / min(10, n):.4f}", flush=True)
model = restore_embeddings(model, new_ids)
out_dir = MT / "sft_minimal_fp32" / "sft_models" / TAG
out_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out_dir)
print("[ssft] saved", out_dir, flush=True)
