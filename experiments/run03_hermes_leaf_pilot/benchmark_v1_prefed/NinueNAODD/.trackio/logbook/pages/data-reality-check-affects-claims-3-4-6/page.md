## Data reality check (affects claims 3, 4, 6)

The public HF release is **smaller** than what Appendix D Table 6 reports:

| split | comparisons (HF) | raters (HF) | comparisons (paper) | raters (paper) |
|---|---|---|---|---|
| IHQ-screened | 1,920 | 48 | 2,012 | 50 |
| IHQ-unscreened | 1,921 | 59 | 2,062 | 62 |
| IHQ-all | 3,841 | 107 | 4,074 | 112 |

Also, the **gt** used for IHQ in the paper is the *official CLIC 2024 leaderboard*, which is not redistributed. We fall back to the paper's own alternative gt ("the ranking achieved on the whole dataset"). All three models do agree on the top item (the uncompressed `reference`), consistent with the paper's statement. HUMAINE, MT-Bench, WD, HiFiC and ConHa are not obtainable here; Google's Crowd-BT implementation was likewise not used — ours is a reimplementation.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Data reality check (affects claims 3, 4, 6)"}\n-->
