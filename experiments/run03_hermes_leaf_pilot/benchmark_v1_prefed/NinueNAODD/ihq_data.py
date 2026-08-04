"""Load the IHQ dataset (Mabyduck/CLIC2024-test-human-eval) into Comparisons.

Each row = one 2AFC trial with two ratings; score 1.0 = chosen, -1.0 = not.
Item identity = the `condition` string (codec @ bpp). Training trials
(is_training=True) are excluded.
"""
import numpy as np
import pandas as pd
from bbq_core import Comparisons

DATA = {"screened": "data/screened.parquet",
        "unscreened": "data/unscreened.parquet"}


def load_split(split, drop_training=True):
    df = pd.read_parquet(DATA[split])
    if drop_training:
        df = df[~df.is_training.astype(bool)]
    rows = []
    for _, row in df.iterrows():
        rats = list(row["ratings"])
        if len(rats) != 2:
            continue
        a, b = rats
        if float(a["score"]) > float(b["score"]):
            win, lose = a["condition"], b["condition"]
        elif float(b["score"]) > float(a["score"]):
            win, lose = b["condition"], a["condition"]
        else:
            continue
        rows.append((int(row["rater_id"]), win, lose))
    raters = sorted({r for r, _, _ in rows})
    items = sorted({c for _, w, l in rows for c in (w, l)})
    rmap = {r: k for k, r in enumerate(raters)}
    imap = {c: k for k, c in enumerate(items)}
    triples = [(rmap[r], imap[w], imap[l]) for r, w, l in rows]
    return Comparisons.from_triples(triples, len(raters), len(items)), items, raters


def load_all():
    """IHQ-all = screened + unscreened pooled (distinct rater id spaces)."""
    cs, items_s, rs = load_split("screened")
    cu, items_u, ru = load_split("unscreened")
    items = sorted(set(items_s) | set(items_u))
    imap = {c: k for k, c in enumerate(items)}
    K = len(items)
    R = cs.R + cu.R
    w = np.zeros((R, K, K))
    for off, (c, it) in enumerate([(cs, items_s), (cu, items_u)]):
        base = 0 if off == 0 else cs.R
        for a, name_a in enumerate(it):
            for b, name_b in enumerate(it):
                if c.w[:, a, b].any():
                    w[base:base + c.R, imap[name_a], imap[name_b]] += c.w[:, a, b]
    return Comparisons(w), items, list(rs) + list(ru)


if __name__ == "__main__":
    for s in DATA:
        c, items, raters = load_split(s)
        print(f"{s}: comparisons={int(c.total())} raters={c.R} items={c.K}")
    c, items, raters = load_all()
    print(f"all: comparisons={int(c.total())} raters={c.R} items={c.K}")
