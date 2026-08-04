## Limitations / honesty statement
* No GPU work, no data, no model was needed or attempted — the paper contains no empirical component,
  so there is nothing to reproduce beyond the mathematics. The "Waluigi effect" claims are verified as
  statements about the paper's probabilistic model, **not** as claims about any real LLM; the paper itself
  says so ("Scope of the interpretation", Sec. 5).
* Impossibility claims (2, 3-binary) are verified by dense/exhaustive numerical search plus a re-derived
  analytic argument, which is strong evidence but not a formal proof check (no Lean/Coq formalisation).
* Claim 4's strict regime required constructed rather than random witnesses (documented above); claim 5's
  strictness is conditional on a hypothesis the main text omits (documented above).
* All searches are seeded (`20260802`) and rerun deterministically.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Limitations / honesty statement"}\n-->
