"""Schema-faithful SYNTHETIC data for the CAME-Grad pilot. No real data is read or downloaded.

Schema sources (all taken from the paper / repository, values are random):
  * modules/trainer.py batch tuple:
      (images, context_images, captions, cls_labels, context_cls_labels,
       context_ids, context_segids, context_attmasks, has_progress)
  * main_train.py: --image_size 224 (3-channel), tokenizer = bert-base-uncased (+ '[DEC]' bos,
      + 4 tokens '[BLA]','[POS]','[NEG]','[UNC]') -> vocab 30522 + 5 = 30527,
      prompt of 14 '[BLA]' tokens, gen_max_len 150, base_probs.json = 14 floats.
  * modules/metrics_clinical.py / models/blip.py: 14 CheXbert conditions, 4 states per
      condition (0 blank, 1 positive, 2 negative, 3 uncertain); heads output 14*4 logits.
  * Paper Sec. 4.1: splits train / val / test (sizes here are tiny; the real sizes are
      270,790 / 2,130 / 3,858 and are NOT reproduced).
Captions are strings of pseudo-words ("w0123"), not clinical text.
"""
import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/work/data"
SIZES = {"train": 32, "val": 8, "test": 8}
IMG = 224
N_COND = 14
N_STATE = 4
BASE_VOCAB = 30522
EXTRA = ["[DEC]", "[BLA]", "[POS]", "[NEG]", "[UNC]"]
VOCAB = BASE_VOCAB + len(EXTRA)  # 30527
BOS_ID = BASE_VOCAB              # [DEC]
BLA_ID = BASE_VOCAB + 1
CAP_LEN = 40                     # report tokens (padded)
PROMPT_LEN = N_COND              # 14 x [BLA]
CTX_LEN = 32


def make_split(name, n, rng):
    d = {}
    d["images"] = rng.standard_normal((n, 3, IMG, IMG)).astype(np.float32)
    d["context_images"] = rng.standard_normal((n, 3, IMG, IMG)).astype(np.float32)
    d["cls_labels"] = rng.integers(0, N_STATE, size=(n, N_COND)).astype(np.int64)
    d["context_cls_labels"] = rng.integers(0, N_STATE, size=(n, N_COND)).astype(np.int64)
    d["has_progress"] = rng.integers(0, 2, size=(n,)).astype(np.int64)
    # decoder input: [DEC] + 14 x [BLA] prompt + report tokens
    lens = rng.integers(CAP_LEN // 2, CAP_LEN + 1, size=n)
    ids = np.zeros((n, 1 + PROMPT_LEN + CAP_LEN), dtype=np.int64)
    att = np.zeros_like(ids)
    for i in range(n):
        body = rng.integers(1000, BASE_VOCAB, size=lens[i])
        row = np.concatenate([[BOS_ID], np.full(PROMPT_LEN, BLA_ID), body])
        ids[i, : len(row)] = row
        att[i, : len(row)] = 1
    d["caption_ids"] = ids
    d["caption_attmask"] = att
    d["context_ids"] = rng.integers(1000, BASE_VOCAB, size=(n, CTX_LEN)).astype(np.int64)
    d["context_segids"] = np.zeros((n, CTX_LEN), dtype=np.int64)
    d["context_attmasks"] = np.ones((n, CTX_LEN), dtype=np.int64)
    words = np.array(["w%04d" % k for k in range(300)])
    d["captions"] = np.array([" ".join(rng.choice(words, size=int(l))) for l in lens])
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20260519)
    schema = {"vocab_size": VOCAB, "bos_id": BOS_ID, "prompt_id": BLA_ID, "n_conditions": N_COND,
              "n_states": N_STATE, "image_size": IMG, "splits": {}}
    for name, n in SIZES.items():
        d = make_split(name, n, rng)
        np.savez(os.path.join(OUT, f"{name}.npz"), **d)
        schema["splits"][name] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in d.items()}
        print(name, {k: (v.shape, str(v.dtype)) for k, v in d.items()})
    base_probs = rng.uniform(0.05, 1.0, size=N_COND).round(4).tolist()
    with open(os.path.join(OUT, "base_probs.json"), "w") as f:
        json.dump(base_probs, f)
    with open(os.path.join(OUT, "schema.json"), "w") as f:
        json.dump(schema, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
