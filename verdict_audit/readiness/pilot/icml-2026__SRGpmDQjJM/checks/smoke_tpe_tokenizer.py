# smoke test: TPETokenizerFast (repo) constructs and encodes/decodes with the installed transformers
import sys, json
sys.path.insert(0, "/work/work/repo")
from transformers import AutoTokenizer
from tpe.tpe_tokenizer_fast import TPETokenizerFast
base = AutoTokenizer.from_pretrained("/work/work/data/hf_models/Qwen--Qwen2.5-1.5B-Instruct")
t = json.load(open("/work/work/data/medtpe_data/cmedqa2/test.json", encoding="utf-8"))[0]["question"]
toks = base.tokenize(t)
merges = [[toks[0], toks[1]]]
info = {toks[0] + toks[1]: {"token_id": 0, "token": toks[0] + toks[1], "token_ngram": toks[0] + " " + toks[1], "sub_token_ids": [], "sub_tokens": [toks[0], toks[1]]}}
tp = TPETokenizerFast(original_tokenizer=base, tok_merges=merges, new_tok_info=info)
print("class ok; chat_template present:", bool(tp.chat_template))
ids = tp(t, add_special_tokens=False)["input_ids"]
print("base tokens", len(toks), "tpe ids", len(ids), "decode roundtrip", tp.decode(ids) == t)
print(tp.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)[:80].replace("\n", "\n"))
