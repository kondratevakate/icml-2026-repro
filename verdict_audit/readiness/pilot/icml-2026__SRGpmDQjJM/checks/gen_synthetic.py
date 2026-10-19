"""Synthetic, schema-faithful fixtures for the MedTPE repository (no real data).

Schemas taken from the repository's loaders:
  * dataset/map_fn.py::cmedqa2_sample_mapping_fn reads columns `question`, `answer`
    from {train,validation,test}.json (preprocessing/prepare_public_datasets.py,
    infer: data/medtpe_data/cmedqa2/{train,validation,test}.json).
  * dataset/map_fn.py::ectsum_sample_mapping_fn reads `text`, `summary` from
    data/medtpe_data/ectsum/{train,val,test}.json.
  * The SFT_QA parquet (create_sft_QA.py output) has columns `message`
    (list of {role, content}), `label` (str), `generated_text_0` (str);
    it is written later by check_C00126.py after pseudo-labels exist.
  * HF model directory named like the repo's default (`Qwen--Qwen2.5-1.5B-Instruct`)
    because the repo parses model names and looks up generation_postfixs by name.
    The content is a tiny random Qwen2-architecture model + a byte-level BPE
    tokenizer trained on SYNTHETIC text only (no real weights, no real vocab).

Text is random: pseudo-words are built from consonant-vowel syllables; "concepts"
are 2-3 pseudo-word phrases (the analogue of "blood pressure monitor") emitted by a
Markov chain with Zipf frequencies, rendered as EHR-style event lines.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

DATA = Path("/work/work/data")
CONS = list("bcdfghjklmnprstvz")
VOW = list("aeiou")
UNITS = ["mg", "mmHg", "bpm", "mL", "g/dL", "mEq/L", "U/L", "%"]

PROMPT_INSTR = (
    "## 数据\n{question}\n\n## 输出指令\n1. 分析数据中病人的情况和问题\n"
    "2. 提供一个简洁的、医学相关的回答\n3. 返回中文纯文本"
)  # same instruction text as dataset/prompt/cmedqa2.yaml (read by the repo itself at run time)


def make_word(rng, nsyl):
    out = []
    for _ in range(nsyl):
        out.append(rng.choice(CONS) + rng.choice(VOW) + (rng.choice(CONS) if rng.random() < 0.3 else ""))
    return "".join(out)


class Lexicon:
    """General-domain words and domain concepts (fixed by `seed`)."""

    def __init__(self, seed=1234, n_general=2500, n_concepts=700):
        rng = random.Random(seed)
        self.general = sorted({make_word(rng, rng.choice([1, 1, 2, 2, 3])) for _ in range(n_general * 2)})[:n_general]
        rng.shuffle(self.general)
        heads = sorted({make_word(rng, rng.choice([3, 4, 5])) for _ in range(900)})
        rng.shuffle(heads)
        concepts = []
        for i in range(n_concepts):
            k = rng.choice([2, 2, 3])
            words = [heads[(i * 3 + j * 7 + rng.randrange(40)) % len(heads)] for j in range(k)]
            concepts.append(" ".join(words).capitalize())
        self.concepts = concepts
        ranks = np.arange(1, n_concepts + 1)
        w = 1.0 / ranks ** 1.1
        self.concept_p = w / w.sum()
        granks = np.arange(1, len(self.general) + 1)
        gw = 1.0 / granks ** 1.05
        self.general_p = gw / gw.sum()
        # Markov successors (5 likely successors per concept)
        nrng = np.random.RandomState(seed)
        self.succ = [nrng.choice(n_concepts, size=5, p=self.concept_p, replace=False) for _ in range(n_concepts)]


def ehr_text(lex, rs, n_events, return_first=False):
    """EHR-style event lines from a concept Markov chain (rs: np RandomState)."""
    c = rs.choice(len(lex.concepts), p=lex.concept_p)
    lines = []
    first = []
    for _ in range(n_events):
        if rs.rand() < 0.6:
            c = int(rs.choice(lex.succ[c]))
        else:
            c = int(rs.choice(len(lex.concepts), p=lex.concept_p))
        val = round(float(rs.gamma(5.0, 18.0)), 1)
        lines.append(f"{lex.concepts[c]}: {val} {UNITS[rs.randint(len(UNITS))]}")
        if len(first) < 3:
            first.append(lex.concepts[c])
    text = "Patient " + str(rs.randint(10000, 99999)) + ", age " + str(rs.randint(18, 95)) + ".\n" + "\n".join(lines)
    return (text, first) if return_first else text


def general_text(lex, rs, n_words):
    ws = rs.choice(len(lex.general), size=n_words, p=lex.general_p)
    words = [lex.general[i] for i in ws]
    out, i = [], 0
    while i < len(words):
        k = int(rs.randint(5, 14))
        out.append(" ".join(words[i:i + k]).capitalize() + ".")
        i += k
    return " ".join(out)


def summary_text(lex, rs, n=3):
    cs = rs.choice(len(lex.concepts), size=n, p=lex.concept_p)
    return "Assessment: " + "; ".join(lex.concepts[i] for i in cs) + "."


def write_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def main():
    lex = Lexicon()
    # ---- CMedQA2-schema files: question, answer -----------------------------------
    sizes = {"train": (1, 2500, 16), "validation": (2, 200, 16), "test": (3, 600, 16)}
    for split, (seed, n, ne) in sizes.items():
        rs = np.random.RandomState(seed * 1000)
        recs = []
        for _ in range(n):
            q, first = ehr_text(lex, rs, ne, return_first=True)
            # answer is a function of the question (first three event concepts) so that the
            # tiny LM's response depends on the prompt content
            recs.append({"question": q, "answer": "Assessment: " + "; ".join(first) + "."})
        write_json(DATA / "medtpe_data" / "cmedqa2" / f"{split}.json", recs)
    # ---- ECTSum-schema files: sample_id, text, summary (long text, short summary) ---
    for split, (seed, n) in {"train": (11, 100), "val": (12, 20), "test": (13, 20)}.items():
        rs = np.random.RandomState(seed * 1000)
        recs = [{"sample_id": f"synth-{split}-{i}", "text": general_text(lex, rs, 600),
                 "summary": general_text(lex, rs, 40)} for i in range(n)]
        write_json(DATA / "medtpe_data" / "ectsum" / f"{split}.json", recs)

    # ---- "pretrained" tokenizer: byte-level BPE (Qwen2-style) on GENERAL text with a
    #      small share of domain text, so domain concepts are fragmented like medical
    #      terms are by a general tokenizer.
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers, normalizers, Regex
    from transformers import PreTrainedTokenizerFast

    rs = np.random.RandomState(777)
    corpus = [general_text(lex, rs, 400) for _ in range(1500)]
    corpus += [ehr_text(lex, rs, 16) for _ in range(300)]  # ~7 percent domain text
    corpus += [PROMPT_INSTR.format(question="")] * 50
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    qwen_pat = Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(qwen_pat, behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    trainer = trainers.BpeTrainer(vocab_size=6000, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(corpus, trainer=trainer)
    chat_template = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
                     "message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
                     "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   bos_token=None, unk_token=None, clean_up_tokenization_spaces=False,
                                   model_max_length=8192, chat_template=chat_template)
    model_dir = DATA / "hf_models" / "Qwen--Qwen2.5-1.5B-Instruct"
    model_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(model_dir)

    # ---- tiny random Qwen2-architecture model (tied embeddings, like Qwen2.5-1.5B) --
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(1)
    cfg = Qwen2Config(vocab_size=len(fast), hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      rope_theta=1000000.0, rms_norm_eps=1e-6, tie_word_embeddings=True,
                      bos_token_id=fast.convert_tokens_to_ids("<|endoftext|>"),
                      eos_token_id=fast.convert_tokens_to_ids("<|im_end|>"),
                      pad_token_id=fast.convert_tokens_to_ids("<|endoftext|>"))
    model = Qwen2ForCausalLM(cfg)
    model.save_pretrained(model_dir)
    (model_dir / "SYNTHETIC_README.txt").write_text(
        "Synthetic stand-in. Tiny random Qwen2-architecture model and a byte-level BPE tokenizer trained on "
        "synthetic text. Directory name only mimics the repository default so name-based lookups work.\n")
    print(json.dumps({
        "tokenizer_len": len(fast), "n_params_tiny_model": sum(p.numel() for p in model.parameters()),
        "cmedqa2_files": {s: len(json.load(open(DATA / 'medtpe_data' / 'cmedqa2' / f'{s}.json', encoding='utf-8')))
                          for s in sizes},
        "example_question_head": json.load(open(DATA / 'medtpe_data' / 'cmedqa2' / 'train.json',
                                                encoding='utf-8'))[0]["question"][:200]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
