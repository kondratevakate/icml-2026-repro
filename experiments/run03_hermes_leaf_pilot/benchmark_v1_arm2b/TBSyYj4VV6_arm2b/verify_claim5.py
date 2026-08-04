"""Claim 5 -- Corollary 12 (Sec 4.2): quantum gamma_p regression (Huber-like,
p=1) in O~(r sqrt(mn)/eps + poly(n, 1/eps)), via the gamma_p-loss framework.
REAL verification via qrepro_driver.run_corollary (loss gamma_p, p=1; Lewis
weights p=1).
"""
from qrepro_driver import run_corollary

if __name__ == "__main__":
    run_corollary(5,
        "Corollary 12: quantum gamma_p (Huber, p=1) regression O~(r sqrt(mn)/eps "
        "+ poly(n,1/eps)) via gamma_p-loss framework (quadratic speedup in m)",
        "arXiv:2509.24757 Sec 4.2 / Corollary 12",
        dict(name="gamma_p", p=1.0),
        eps=0.25, lewis_p=1.0)
