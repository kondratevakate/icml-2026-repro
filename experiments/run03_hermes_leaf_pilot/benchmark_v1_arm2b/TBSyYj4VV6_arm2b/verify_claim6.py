"""Claim 6 -- Corollary 11 (Sec 4.2): quantum ell_p regression for p in (0,2]
(p=1.5 tested), achieving a quadratic speedup in the sample parameter m, which
the authors note dominates when m >> n.  REAL verification via
qrepro_driver.run_corollary (loss lp, p=1.5; Lewis weights p=1.5).
"""
from qrepro_driver import run_corollary

if __name__ == "__main__":
    run_corollary(6,
        "Corollary 11: quantum ell_p regression (p=1.5 in (0,2]) quadratic "
        "speedup in m; m-term dominates when m >> n",
        "arXiv:2509.24757 Sec 4.2 / Corollary 11",
        dict(name="lp", p=1.5),
        eps=0.25, lewis_p=1.5)
