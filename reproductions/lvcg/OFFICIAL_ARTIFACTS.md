# Official artifacts

## Paper and repository

- OpenReview: `hS6iw4PM8K`
- arXiv: `2605.31249v1`
- PDF SHA-256:
  `199E996C786E10CFFE8B39ECA2EF26DFAE5F2474573C63B3A3A9F3C556F3CDEA`
- source archive SHA-256:
  `09577A0FE30F6FE5F76B65E6DEFB30169E146BFDB3082FE146C2832FFC4E74AF`
- official repository: `BosonHwang/LVCG`
- audited commit: `0fcacbf34784cd876b4c197599253a9130707f93`

The repository has 108 tracked files but no checkpoint, empirical result file,
or Python test. Its referenced `checkpoints/m5s1k1/final.pt` is absent.

The latest commit was made after the paper release and repairs a beat-loss
shape mismatch that made the default GRU training path crash. The repository
history contains no version of the missing `lvcg/data` package.

## Required datasets

| Role | Paper dataset | Release support |
| --- | --- | --- |
| pretraining | MIMIC-IV-ECG v1.0 | manifest builder present; waveform download pending locally |
| linear probing | PTB-XL v1.0.3 | split CSVs present; raw waveforms absent |
| linear probing | CPSC 2018 / ICBEB | split CSVs present; raw waveforms absent |
| linear probing | Chapman-Shaoxing | split CSVs present; raw waveforms absent |
| non-cardiac | MIMIC-IV-ECG-Ext-ICD | no loader, split, or config present |

The release instead adds an AI-READI provider and calls it the seventh probing
dataset. AI-READI is not the non-cardiac dataset reported in the paper.

No clinical waveform, patient identifier, or credentialed row is copied into
the audit package.
