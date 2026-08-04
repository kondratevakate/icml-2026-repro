"""Claim 3 -- Corollary 26 (App E): FIRST quantum algorithm for LASSO regression,
O~(r sqrt(mn)/eps + poly(n, 1/eps)), versus O~(m n^2 + n^3) classically.  REAL
verification via qrepro_driver.run_corollary (lasso = l2 + l1 penalty; sparsifier
uses leverage scores, p=2, as the paper embeds Lasso via augmented l2 points).
"""
from qrepro_driver import run_corollary

if __name__ == "__main__":
    run_corollary(3,
        "Corollary 26: first quantum Lasso regression O~(r sqrt(mn)/eps + "
        "poly(n,1/eps)) vs O~(m n^2 + n^3) classically (quadratic speedup in m)",
        "arXiv:2509.24757 App E / Corollary 26",
        dict(name="lasso", lam=1.0),
        eps=0.25, lewis_p=2.0)
