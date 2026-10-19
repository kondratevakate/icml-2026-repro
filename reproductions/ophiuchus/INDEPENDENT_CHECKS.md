# Independent checks

Prepared score: **2/10**, the lower bound of the frozen 2-4 range.

| Claim | Score | Result |
| --- | ---: | --- |
| C1 medical agent and three tools | 1/2 | Local zoom/state/reward core runs, but released SFT tool-call syntax is not parsed, duplicate calls do not early-stop, and list-valued segmentation masks fail reset. |
| C2 three-stage training | 1/2 | Cold-start, reflection, and ATRL entrypoints exist; preprocess crashes on a literal placeholder, ATRL uses absent paths, and its effective KL setting conflicts with the paper. |
| C3 64k corpus | 0/2 | Inconclusive: only two SFT and two RL records are released. |
| C4 eight-benchmark superiority | 0/2 | Inconclusive: checkpoint and complete evaluation release are absent. |
| C5 grounding/reflection/ablations | 0/2 | Inconclusive: checkpoint, datasets, predictions, and ablation outputs are absent. |

Key release findings:

- Both example parquet files are byte-identical, contain two rows, and point to
  nonexistent `/inspire/.../your/path/to/...` images.
- All nine image/mask references in the JSON examples are absent as written,
  although differently located example images are present.
- `prepare_Ophiuchus_dataset.py` raises `NameError` at
  `primary_image = your/path/to(image_paths[0])`.
- The paper's `<tool_call>` examples are not accepted by the environment's
  exact `Action: <tool name>` parser.
- Repeating the identical local zoom call does not terminate the rollout,
  contrary to the paper.
- The published segmentation record stores `mask_path` as a list; reset passes
  it to `os.path.abspath` and raises `TypeError`.
- A canonical local zoom followed by a correct answer executes and receives
  reward 4.0, confirming that a substantial core exists.
- The ATRL shell script's last KL assignment is `True` with coefficient 0.01,
  while the paper specifies GRPO without KL penalty.

No leaderboard or third-party verdict was used.
