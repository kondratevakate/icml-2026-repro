"""paper_tables.py — parse Table 1 and Table 2 out of the PDF-extracted paper text.

The arXiv PDF text extraction emits one token per line inside tables, so a row is
"<model name>" followed by optional size/reasoning/MoE markers and then N numbers.
No paper text is hard-coded here except row/section anchors.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "paper", "paper.txt")

NUM = re.compile(r"^\d+(?:\.\d+)?$")
SIZE = re.compile(r"^\d+B$")
SKIP = {"\u2713", "\u2014", "MoE", "Reasoning"}


def _lines():
    with open(PAPER, encoding="utf-8", errors="replace") as fh:
        return [ln.strip() for ln in fh]


def _slice(start_pat, end_pat):
    ls = _lines()
    i = next(k for k, ln in enumerate(ls) if re.search(start_pat, ln))
    j = next(k for k, ln in enumerate(ls) if k > i and re.search(end_pat, ln))
    return ls[i:j]


def _rows(block, ncols, skip_tokens=()):
    """Greedy row assembly: name, then ncols numeric cells."""
    out, name, buf = [], None, []
    for ln in block:
        if not ln:
            continue
        if NUM.match(ln):
            buf.append(float(ln))
            if len(buf) == ncols:
                out.append((name, buf))
                name, buf = None, []
        elif ln in SKIP or SIZE.match(ln) or ln in skip_tokens:
            continue
        elif not buf:
            name = ln
    return out


TYPES = {"Text-to-SQL", "SQL debugging"}


def table1():
    """{benchmark_name: dict(type, n_test, tok, line, func, depth, width)}; '—' cells -> None."""
    blk = _slice(r"^Table 1\.", r"^LATERAL VIEW Required")
    keys = ("n_test", "tok", "line", "func", "depth", "width")
    out, name, row, cells = {}, None, None, []
    for ln in blk:
        if not ln:
            continue
        if ln in TYPES:
            row, cells = {"type": ln}, []
            continue
        if row is not None:
            tok = ln.replace(",", "")
            if NUM.match(tok):
                cells.append(float(tok))
            elif ln == "\u2014":
                cells.append(None)
            else:
                continue
            if len(cells) == 6:
                row.update(zip(keys, cells))
                out[name] = row
                row = None
        elif not NUM.match(ln.replace(",", "")) and ln not in SKIP:
            name = ln
    return out


def table2():
    """{model_label: dict(syn_em, syn_gm, syn_mb, sem_em, sem_gm, sem_mb)} for the 24 evaluated LLMs.

    The three SFT variants below the 'Comparison of different SFT method' header are excluded:
    they are this paper's own fine-tuned baselines, not 'evaluated LLMs'.
    """
    blk = _slice(r"^Table 2\.", r"^Comparison of different SFT method")
    rows = _rows(blk, 6, skip_tokens={"Open Source", "Closed Source", "Model", "Size",
                                      "EM", "GM", "MB", "Squirrel-Syntax", "Squirrel-Semantic"})
    keys = ("syn_em", "syn_gm", "syn_mb", "sem_em", "sem_gm", "sem_mb")
    out, seen = {}, {}
    for name, v in rows:
        if name is None:
            continue
        seen[name] = seen.get(name, 0) + 1
        label = name if seen[name] == 1 else f"{name}#{seen[name]}"
        out[label] = dict(zip(keys, v))
    return out


def sft_rows():
    blk = _slice(r"^Comparison of different SFT method", r"^DeepSeek, Claude, GPT")
    rows = _rows(blk, 6, skip_tokens=set())
    keys = ("syn_em", "syn_gm", "syn_mb", "sem_em", "sem_gm", "sem_mb")
    return {n: dict(zip(keys, v)) for n, v in rows if n}


if __name__ == "__main__":
    import json
    print(json.dumps({"table1": table1(), "table2": table2(), "sft": sft_rows()}, indent=1))
