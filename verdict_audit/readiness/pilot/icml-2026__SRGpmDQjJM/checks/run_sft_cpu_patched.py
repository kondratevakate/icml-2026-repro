"""Wrapper that runs the repository's llm_sft.py unchanged on CPU.

usage: python run_sft_cpu_patched.py <variant> <llm_sft.py args...>
  variant 'shipped'    : no patch at all (real bitsandbytes Adam8bit, bf16 weights)
  variant 'adamw'      : Adam8bit replaced by torch.optim.AdamW (bitsandbytes 8-bit optimisers need CUDA)
  variant 'adamw_fp32' : as 'adamw' and AutoModelForCausalLM.from_pretrained forced to torch_dtype=float32
The repository files are never modified; patches live only in this process.
"""
import runpy
import sys

import torch

variant = sys.argv.pop(1)
sys.path.insert(0, "/work/work/repo")
if variant in ("adamw", "adamw_fp32"):
    import bitsandbytes
    import bitsandbytes.optim
    bitsandbytes.optim.Adam8bit = torch.optim.AdamW
if variant == "adamw_fp32":
    import transformers
    _orig = transformers.AutoModelForCausalLM.from_pretrained

    def _fp32(cls, *a, **k):
        k["torch_dtype"] = torch.float32
        return _orig(*a, **k)

    transformers.AutoModelForCausalLM.from_pretrained = classmethod(_fp32)
print(f"[wrapper] variant={variant} argv={sys.argv}", flush=True)
sys.argv[0] = "/work/work/repo/llm_sft.py"
runpy.run_path("/work/work/repo/llm_sft.py", run_name="__main__")
