# Claim 3: Real-data UNOS/OPTN results not reproducible


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_949593b76e96", "created_at": "2026-07-23T09:12:58+00:00", "title": "Claim 3: Real-data UNOS/OPTN results not reproducible"}
-->
**Paper claim (Section 3).** Applied to kidney transplant data, FunCQNet yields precise multi-quantile predictions and reveals clinically significant, age-dependent interaction patterns between donor type and recipient survival.

**Data provenance (confirmed from the PDF).** The motivating dataset is the **United Network for Organ Sharing / Organ Procurement Transplantation Network (UNOS/OPTN)** — explicitly named in Section 3 and in the References (OPTN/SRTR annual data report). This resolves a provenance question that was previously unverified: the registry is UNOS/OPTN, not a public cohort.

**Why it is not reproducible.**

- UNOS/OPTN data are distributed as **access-controlled STAR files**, released only under a signed Data Use Agreement, with an application/review process (weeks to months) and a fee. There is **no public version** of the cohort.
- The paper releases **no code, no processed data, no checkpoint** (OpenReview only — no arXiv, no GitHub, no Hugging Face artifacts).
- Therefore the real-data quantitative results (multi-quantile predictions, the age × donor-type interaction surfaces) cannot be independently re-run or verified without first obtaining the private registry and re-implementing the model.

**Verdict: NOT REPRODUCIBLE — data private.** This is a data-access blocker, not a modeling failure: the claim may well be correct, but no third party can confirm it from public materials. The reproducible evidence for the method is the simulation study (Claims 1–2).
