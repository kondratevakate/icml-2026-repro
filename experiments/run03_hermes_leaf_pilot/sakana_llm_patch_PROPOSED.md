# PROPOSED patch for AI-Scientist-v2/ai_scientist/llm.py (arm 3 of ICML-2026-repro benchmark).
# NOT APPLIED. Pending Bill security review (LLM-written-code execution).
#
# Insert this branch in create_client(), BEFORE the final `else: raise ValueError(...)`
# (currently line 543, the `elif 'gemini' in model:` block ends at 542).
#
# This routes Sakana's LLM calls to our LOCAL Hermes router (hy3:free via Nous),
# so the benchmark compares architectures on ONE free model (closes V2 cost-confound).

    elif "hy3" in model or model.startswith("tencent"):
        # Route to local Hermes router (hy3:free via Nous). No provider key needed;
        # router authenticates via PROXY_API_KEY and fans out to the free upstream.
        return (
            openai.OpenAI(
                api_key=os.environ.get("PROXY_API_KEY", "sk-local"),
                base_url="http://localhost:8319/v1",
            ),
            model,
        )

# Usage after applying:
#   export PROXY_API_KEY=<router proxy key>
#   python launch_scientist_bfts.py --model tencent/hy3:free  (or --model hy3:free)
#
# Caveats (for Bill):
#   - Sakana executes LLM-generated code (perform_experiments_*). With hy3:free routed
#     through our router, the "agent" writing that code is a model we don't fully control.
#   - Sandboxing: Sakana runs experiments in-process (no docker by default). On this host
#     (WSL, no GPU) its experiment exec needs sklearn/torch which paper1's official code
#     requires; agent must reimplement on numpy/scipy per our TASK.md. GPU-bound papers
#     won't execute (ceiling = CPU-feasible subset).
#   - No network egress beyond localhost:8319 (router). Router itself calls Nous (external).
