# CalPro reproduction bundle

Independent from-the-paper reproduction of CalPro (ICML 2026, arXiv 2601.07201) on the
paper's own non-biological heteroscedastic-regression-under-shift setting. CPU only.

## Rerun
    python verify_calpro.py            # Phase 1: split conformal, coverage collapses under shift
    python verify_calpro.py --selftest # in-distribution coverage lands on nominal tau
    python verify_calpro_phase2.py     # Phase 2: evidential NIG + normalized conformal restores coverage
    python claim2_ece.py               # Claim 2: ECE reduction vs homoscedastic baseline

Requires: numpy, torch (CPU). No GPU, no external data (synthetic).
