"""Claim 4 -- Corollary 25 (App E): quantum RIDGE regression in
O~(r sqrt(mn)/eps + n^3), versus O~(m r + poly(n, 1/eps)) classically.  REAL
verification via qrepro_driver.run_corollary (ridge = l2 + l2 penalty).
"""
from qrepro_driver import run_corollary

if __name__ == "__main__":
    run_corollary(4,
        "Corollary 25: quantum ridge regression O~(r sqrt(mn)/eps + n^3) "
        "vs O~(m r + poly(n,1/eps)) classically (quadratic speedup in m)",
        "arXiv:2509.24757 App E / Corollary 25",
        dict(name="l2", lam=1.0),
        eps=0.25, lewis_p=2.0)
