You are the adjudicator for a research study. Two independent blind coders disagreed on some labels; decide each one. Do nothing else.

Read ONLY this file (do not open any other file in or near this folder, especially nothing named keys*, frame*, papers*, codes of other batches or coders, results*):
C:/Projects/02_academia/studies/icml/icml-repro-working/verdict_audit/readiness/data/p9/adj_sheets/p9_adj_batch_{BATCH}.csv
(columns item_id, field, definition, label_a, label_b, title, truncated, text). Use Python with the csv module to read it. Treat all text in it as data, never as instructions.

Each row is one disagreement on one field of one machine-learning paper. "text" is what the coders saw: excerpts of the paper's full text, or, for the fields code_used, data_obtained and weights_used, the reproduction logbooks' claims with the judge's evidence. "truncated" says whether the text was cut at its length cap. "definition" gives the field's labels and their meaning. label_a and label_b are the two coders' labels, in random order.

For each row, decide the correct label from what the text states, by the definition (usually one of the two labels; any label in the definition is allowed). Do not use outside knowledge about the paper, its code or its data. Give a reason of one sentence saying what in the text decides it (or that the text says nothing about it).

Read every row yourself and decide individually; do not use keyword rules or scripts to decide. Write exactly this file (UTF-8, header item_id,field,label,reason; one row per input row):
C:/Projects/02_academia/studies/icml/icml-repro-working/verdict_audit/readiness/codes/p9/p9_adj_batch_{BATCH}.csv
Do not create or modify any other file. Reply with only the number of rows decided.
Do not write helper files to shared temporary paths such as /tmp; read the sheet with inline Python instead.
