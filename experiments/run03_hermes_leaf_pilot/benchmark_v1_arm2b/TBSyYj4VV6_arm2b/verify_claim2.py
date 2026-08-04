"""Claim 2 -- Corollary 23 (App E): quantum LINEAR regression in
O~(r sqrt(mn)/eps + n^3), versus O~(m r + n^3) classically.  Quadratic speedup
in m (the sqrt(mn) term).  REAL verification via qrepro_driver.run_corollary.
"""
from qrepro_driver import run_corollary

if __name__ == "__main__":
    run_corollary(2,
        "Corollary 23: quantum linear regression O~(r sqrt(mn)/eps + n^3) "
        "vs O~(m r + n^3) classically (quadratic speedup in m)",
        "arXiv:2509.24757 App E / Corollary 23",
        dict(name="l2", kw=dict()),
        eps=0.25, lewis_p=2.0)
