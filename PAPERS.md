# ICML 2026 Reproduction — Paper Reference (Kate)

Полный справочник по статьям, отобранным под интересы Kate из 6341 статьи ICML 2026.
Ссылка = OpenReview forum. Очки = 2 x число проверяемых claims (2/claim за полное
воспроизведение ИЛИ полное опровержение, 1 toy, 0 нет).

Легенда фита: A=ядро (оценка/надёжность/сегментация/нейродегенерация), B=рекон/робастность/3D-SSL,
C=fMRI-энкодеры, D=brain-графы/психиатрия, E=клиника/wearables/digital twin.
Доступ: OPEN=скачать сразу, REG=регистрация, CRED=credentialed PhysioNet (у Kate есть), REST=по заявке.
Код-статус: репозиторий проверялся вручную там, где указан URL; "walled" = OpenReview за ботозащитой, не подтверждено.

---

# СТАТУС И ПРИОРИТЕТЫ (обновлено 2026-07-19)

**Фильтр отбора:** применение к медданным (прямо или косвенно) plus воспроизводимость plus claim, который
можно взять на вооружение. **Единственный непрыгаемый критерий — доступность данных.** Код и текст не критерии.
**Статьи НЕ эксклюзивны** — повторные воспроизведения приветствуются, «занято» не отсеивает.
**Режим:** сначала всё без GPU, потом отобранное пачкой на GPU-сервер. Полное состояние — в `HANDOFF.md`.

**ВАЖНО про поиск:** этот список и `medical_neuro_papers.csv` строились по area-тегу и **теряют медицину**.
Скан всех 6341 по заголовкам даёт **368 медицинских статей, 113 из них вне 255** (Causality, LLM, CV,
Time Series, Evaluation, Privacy). uqct, PyHealth и ProConMV никогда не были в 255. Искать по `all_papers.csv`.
**Доступность данных не видна из абстракта** — читать PDF (ORBIT: в интро «public», в теле закрытый госпитальный набор).

## ЛОГБУКИ (8 опубликовано, все канонические, валидатор зелёный)

| статья | orid | результат |
|---|---|---|
| ConfSleepNet | fve4MEzeSp | claim 1 аналитически; **Ур. 8 как напечатано — вероятно опечатка**. claims 2-3 блокированы |
| Solvable AE | wm3ABfhE7P | claim 2 убедительно; claim 1 k\*=2 чисто, k\*=3 маргинально (как у авторов) |
| CalPro | 3LRWjJTp0Y | **ИСПРАВЛЕНО 2026-07-19:** опубликованный вердикт claim 1 был на одном сиде (0.984 покрытия — везение). На 5-8 сидах (`calpro/verify_claim1_degradation.py`) деградация покрытия 17.5pp средняя / 38.8pp худшая — **порог claim (5pp) не выполняется**, направление эффекта верное, но заявленная величина не воспроизводится. Опубликованный логбook пока не обновлён |
| uqct | a6vCNSBBeq | гарантия покрытия SLM на классическом пути (1.000 при δ=0.1 и 0.05); «глубокие туже» под GPU |
| Entropy/mislabeled | BUxrIaf7Zc | **toy на реальных CIFAR-100N**: claim 1 по направлению, claim 2 SEI-прокси AUC 0.71 |
| Conditional Coverage Diagnostics | vaApZm6MKM | claims 1-2 воспроизведены: при незаданных группах CovGap слепнет (0.005 против истины 0.095), ERT достаёт 92%; over/under разделение совпало с теорией |
| When Can We Trust Survival Model Evaluation | Y9gsOEdaNE | **claims 1-2 воспроизведены**: смещение метрик механизм-зависимо (IBS: административное −0.080, зависящее от ковариат ~0), а при близких моделях ранжирование совпадает с оракулом лишь в 29-42% при 80% цензурирования |
| SurvFD / SurvSHAP-IQ | SldP4LGjdz | **claims 1-2**: Theorem 3.2 воспроизведена точно (3/3 сценария, common random numbers); n-Shapley взаимодействие в 1.96x больше при истинном взаимодействии (честная оговорка про exp-связь на S-шкале) |

## ОЧЕРЕДЬ БЕЗ GPU (делать сейчас)

- **LERD** `B5DAV1EA8Z` — нейродегенерация/Альцгеймер EEG. Claim 1 включает **синтетические бенчмарки** → часть без загрузок. Реальные данные: OpenNeuro **ds004504** (открыт).
- **DC-PnPDP** `jEBkuuETjr` claim 1 — теорема сходимости, аудит plus малый синтетический тест.
- **ProConMV** `4F4Ziv1d9G` Corollary 1 — уже проверено (`proconmv/verify_corollary1.py`); полные claims заблокированы (нет кода и концепт-аннотаций).

## ОЧЕРЕДЬ НА GPU-СЕРВЕР (собрать пачкой)

- **EviScreen** `FCVPVqRLDJ` — самый готовый медицинский: eval-only путь plus released-веса plus открытые fundus-данные.
- **uqct** «глубокие дают более узкие регионы» — U-Net/ансамбли/диффузия plus LIDC-IDRI.
- **Entropy claim 3** — полный SEI plus три медицинских датасета (ISIC, DeepDRiD).
- **MedCRP-CL**, **DC-PnPDP** claim 2, **TACO**, **ECG-R1**, **SigmaPPG**, **xMAE**, **CortexMAE**, **MoQiswth2n**.


## ЛОКАЛЬНЫЕ ДАННЫЕ (для toy-проверок без GPU)

- **CIFAR-100N** (`data/cifar100n/`, 340 МБ) — CIFAR-100 plus человеческие шумные метки. Использован для Entropy.
- **MFIDDR sample** (`data/mfiddr_sample/`, 43 МБ) — 96 fundus-изображений, 4 поля/пациент, git-клонируемый сэмпл.
  Прямого свободного применения не нашлось: fundus-статьи в очереди (EviScreen, ProConMV) требуют GPU/тяжёлых моделей
  либо заблокированы (ProConMV — код/аннотации не выложены). Держим про запас под будущий fundus-таргет.
- **MIMIC-III Clinical Database Demo** (`data/mimic3demo/`, 38 МБ, 26 таблиц) — **открыт без авторизации**
  (physionet.org/content/mimiciii-demo/1.4/, 100 пациентов). Разблокировал **PyHealth 2.0** (`gMLVFN9hl8`),
  которая была помечена «нужен credentialed MIMIC» — реально совместима с demo-форматом. **Рабочая среда:
  WSL Ubuntu-24.04 (Python 3.12.3, версия PyHealth без хаков), не Windows.** Установка через `uv` (без sudo):
  `uv venv --python 3.12` plus `uv pip install 'git+https://github.com/sunlabuiuc/PyHealth.git@master'`.
  **Toy-прогон подтверждён end-to-end за 6.8с:** загрузка MIMIC-III demo (100 пациентов, 13030 событий) →
  задача `MortalityPredictionMIMIC3` → `Transformer` → обучение 3 эпохи → интерпретируемость. На Windows этот
  же пайплайн дважды зависал — причина оказалась не платформенной, а отсутствием `if __name__ ==
  "__main__":` guard в скрипте (dask спавнил воркеров, каждый переимпортировал модуль и пересоздавал
  кэш — бесконечная рекурсия процессов). Claim 2 (39x быстрее/20x меньше памяти v1 vs v2) не пытались —
  нужен контролируемый бенчмарк.

## ЗАБЛОКИРОВАНО (триггеры пересмотра)

- **TCSeg** `bGxMpQsIuN` — репо README-only (0 .py). Триггер: релиз кода.
- **ProConMV** `4F4Ziv1d9G` — репо пустое, шесть lesion-концептов (вклад авторов) не выложены. Данные: MFIDDR сэмпл клонируется, полный plus DRTiD — по анкете.
- **ORBIT** `pd6GY2x8FG` — лонгитюдный fundus/OCT, Shanghai General Hospital, IRB, не выложен. **Выбывает по данным.**
- **SzCORE**, **S2M-Net** — закрытый held-out / отозвана.

---

## БРАТЬ СЕЙЧАС (данные доступны, код есть)

| # | Статья | Фит | Очки | Данные (доступ) | Код |
|---|---|---|---|---|---|
| 1 | [PyHealth 2.0](https://openreview.net/forum?id=gMLVFN9hl8) · arXiv 2601.16414 | E | 10 | MIMIC-III/IV, eICU, SleepEDF, SHHS (CRED, есть) | [sunlabuiuc/PyHealth](https://github.com/sunlabuiuc/PyHealth) · LIGHT |
| 2 | [TCSeg (Are We Overconfident?)](https://openreview.net/forum?id=bGxMpQsIuN) · arXiv 2605.25561 | A | 12 | Pancreas-CT, LA, BraTS2019 (TCIA, OPEN) | [DirkLiii/TCSeg](https://github.com/DirkLiii/TCSeg) — **README-only, кода нет (GitHub API 2026-07-18: 0 .py). BLOCKED до релиза кода** |
| 3 | [Principled Confidence CT](https://openreview.net/forum?id=a6vCNSBBeq) · arXiv 2602.05812 | A | 12 | LIDC-IDRI (TCIA, OPEN); 3 claim на закрытом industrial CT → реальный потолок ~6 | [SwissDataScienceCenter/uqct](https://github.com/SwissDataScienceCenter/uqct) · MEDIUM |
| 4 | [MedCRP-CL](https://openreview.net/forum?id=v0DWbfP3b9) · arXiv 2605.20297 | A | 10 | 16 seg-задач (endoscopy/derm/US/CXR, OPEN, собирать) | [zygao930/MedCRP-CL](https://github.com/zygao930/MedCRP-CL) · MEDIUM |
| 5 | [DC-PnPDP](https://openreview.net/forum?id=jEBkuuETjr) · arXiv 2602.23214 | B | 12 | AbdomenCT-1K, fastMRI (OPEN) | [duchenhe/DC-PnPDP](https://github.com/duchenhe/DC-PnPDP) · MEDIUM |
| 6 | [EviScreen](https://openreview.net/forum?id=FCVPVqRLDJ) · arXiv 2605.15171 | A/E | 12 | JSIEC, RIADD, CheXpert, Derm12345 (OPEN; CheXpert большой, брать small) | [DopamineLcy/EviScreen](https://github.com/DopamineLcy/EviScreen) · MEDIUM |
| 7 | [ConfSleepNet](https://openreview.net/forum?id=fve4MEzeSp) · arXiv 2605.17021 | E-wear | 10 | SleepEDF-20/78 (OPEN), MASS (REST), SHHS (REG) | [By4te/ConfSleepNet_ICML2026](https://github.com/By4te/ConfSleepNet_ICML2026) — **README only, кода нет** |

**ConfSleepNet: СТАТУС РЕПРОДУКЦИИ.** Claim 5 (Proposition 1 и 2) воспроизведён независимо (CPU, скрипт `confsleepnet/verify_prop.py`): пропозиции держатся при нормировке Σb+u=1; Ур. 8 как напечатано вероятно с опечаткой (проваливает Prop 2 при C→1). Логбук опубликован: https://huggingface.co/spaces/kondratevakate/repro-a-conflict-aware-evidential-framework-for-reliable-sleep-stage-classification.

## ДОСТУПНО, НО ТЯЖЁЛОЕ (код и данные есть, но foundation-масштаб)

| # | Статья | Фит | Очки | Данные (доступ) | Код |
|---|---|---|---|---|---|
| 8 | [CortexMAE (ViT fMRI)](https://openreview.net/forum?id=s8cdRWLTCc) · arXiv 2510.13768 | C | 10 | HCP, NSD (REG, есть); UK Biobank-часть пропустить | [MedARC-AI/fmri-fm](https://github.com/MedARC-AI/fmri-fm) · HEAVY |
| 9 | [Foundation VAE CT](https://openreview.net/forum?id=f51haqkwRQ) | B | 4 | LiTS, KiTS19, MSD, CT-RATE (OPEN) | [qic999/Foundation-VAE](https://github.com/qic999/Foundation-VAE) · HEAVY |
| 10 | [ECG-R1](https://openreview.net/forum?id=dcLjhvVU47) · arXiv 2602.04279 | E-wear | 12 | PTB-XL, Code-15, CPSC2018, Chapman-Shaoxing, Georgia (OPEN), MIMIC-IV-ECG (CRED) | [PKUDigitalHealth/ECG-R1](https://github.com/PKUDigitalHealth/ECG-R1) · HEAVY |
| 11 | [SIGMA-PPG](https://openreview.net/forum?id=PKgGsbORt9) · arXiv 2601.21031 | E-wear | 12 | MIMIC-III waveform, BIDMC (CRED, есть) | [ZonghengGuo/SigmaPPG](https://github.com/ZonghengGuo/SigmaPPG) · HEAVY |
| 12 | [xMAE](https://openreview.net/forum?id=NxbSbkLNMc) · arXiv 2605.00973 | E-wear | 12 | MIMIC-III waveform matched (CRED, есть) | [hzhou3/xMAE](https://github.com/hzhou3/xMAE) — веса не выложены · MED/HEAVY |
| 13 | [TACO (Beyond Instance-Level SSL 3D)](https://openreview.net/forum?id=Vtps0ZUlgv) · arXiv 2605.14654 | B | 12 | UPENN-GBM (TCIA), BraTS, ISLES22, ADNI, ADHD-200, IXI (OPEN) | [Ashespt/TACO](https://github.com/Ashespt/TACO) — «по публикации» · HEAVY |
| 14 | [Omni-fMRI](https://openreview.net/forum?id=Rc8th2rXXe) · arXiv 2601.23090 | C | 12 | 9 датасетов, не названы | [OneMore1/Omni-fMRI](https://github.com/OneMore1/Omni-fMRI) (заявлен) · HEAVY |

## WATCH-LIST (важные, кода пока нет, пересмотр ~6 мес)

| # | Статья | Фит | Очки | Данные | Триггер |
|---|---|---|---|---|---|
| 15 | [PhenoBrain](https://openreview.net/forum?id=9NqKL9QQ4a) **Spotlight** | D | 4 | ADNI/ABIDE (вероятно) | релиз кода |
| 16 | [Deep Learning for BioImaging: What Are We Learning?](https://openreview.net/forum?id=ZgJlDylia4) | A | 6 | RxRx3, JUMP-CP, HEST (OPEN) | релиз кода |
| 17 | [T-measure (seg-метрика)](https://openreview.net/forum?id=tRpJyIzxDj) | A | 6 | walled | релиз/абстракт |
| 18 | [Real-World Unsupervised Predict Brain Responses OOD](https://openreview.net/forum?id=H1HHss5Zj4) | C | 4 | вероятно NSD | релиз кода |
| 19 | [LERD (нейродегенерация)](https://openreview.net/forum?id=B5DAV1EA8Z) · arXiv 2602.18195 | A | 12 | OpenNeuro ds004504 (OPEN) | данные открыты, можно переписать |
| 20 | [Geometry-Guided Brain Graphs](https://openreview.net/forum?id=ieUVj77Hsz) · arXiv 2511.04539 | D | 6 | HCP rest+WM fMRI, 1067 subj (REG, есть) | данные есть, переписать |
| 21 | [Med-SegLens](https://openreview.net/forum?id=vaRFU0xKQa) · arXiv 2602.10508 | A | 12 | BraTS (OPEN) | релиз кода |
| 22 | [Disease-Centric VLP 3D CT](https://openreview.net/forum?id=LcSPuepCDU) · arXiv 2606.25546 | B | 12 | CT-RATE, Rad-ChestCT (OPEN) | релиз кода |
| 23 | [Brain Networks Should Be Learned, Not Constructed](https://openreview.net/forum?id=VRJKiV0qGM) | D | 4 | ABIDE (вероятно) | код заявлен на yangliang.github.io, ссылка не открылась |
| 24 | [On the Spectral Unreachability of Brain Graph Learning](https://openreview.net/forum?id=n3ilFhSRIx) | D | 4 | brain-граф бенчмарки | код заявлен там же, не открылся |
| 25 | [Multimodal Scaling Laws Visual Cortex](https://openreview.net/forum?id=OQ6jQHJPTT) | C | 6 | NSD/Brain-Score (REG) | репо прошлой версии у EPFL NeuroAI, найти для этой |
| 26 | [NeurIPS: sphere-based brain decoding](https://openreview.net/forum?id=gXTtBRI1yj) · arXiv 2605.24993 | C | 10 | Natural Scenes Dataset (REG) | релиз кода |
| 27 | [FACT (multi-label дх)](https://openreview.net/forum?id=T5u3haVBgi) | A | 6 | walled (вероятно CXR) | абстракт/код |
| 28 | [Reference-Free Meta-Learning MRI recon](https://openreview.net/forum?id=jPVGiAUlNa) | B | 6 | walled (fastMRI?) | абстракт/код |
| 29 | [SI-IGCL (психиатрия)](https://openreview.net/forum?id=ZoYQxFtDVM) | D | 4 | ABIDE/ADHD-200 (вероятно) | релиз кода |
| 30 | [Structured Multi-modal Graph (психиатрия)](https://openreview.net/forum?id=cX3r7kAqr1) | D | 4 | REST-meta-MDD/ABIDE (вероятно) | релиз кода |
| 31 | [MuHL (гиперграф)](https://openreview.net/forum?id=j5vTJrpoeL) · arXiv 2606.03310 | D | 12 | ABIDE/ADNI (вероятно) | релиз кода |
| 32 | [ADHD Detection](https://openreview.net/forum?id=IaZmw8c9U4) | D | 4 | ADHD-200 (OPEN) | релиз кода |
| 33 | [Alignment between Brains and AI](https://openreview.net/forum?id=XrRqY9QOSa) · arXiv 2507.01966 | C | 12 | Brain-Score-style | релиз кода |
| 34 | [Local Intrinsic Dimension](https://openreview.net/forum?id=B3Z3rWo4t6) · arXiv 2601.22722 | C | 10 | не названы | абстракт/код |
| 35 | [Mind the State (EEG-to-fMRI)](https://openreview.net/forum?id=Kw1Z8gOTvk) | C | 6 | не названы | абстракт/код |
| 36 | [Uncovering Latent Communication (AFR-Net)](https://openreview.net/forum?id=clFn9vQ8cK) · arXiv 2602.00561 | D | 10 | ABCD (REST), PPMI (REG) — НЕ HCP | только анонимный репо |
| 37 | [OSF (sleep FM)](https://openreview.net/forum?id=rlVGCyjqnt) · arXiv 2603.00190 | E-wear | 10 | NSRR: SHHS/MESA/MrOS (REG) plus SleepBench | релиз кода |
| 38 | [Solvable AE: Test Loss Misaligns](https://openreview.net/forum?id=wm3ABfhE7P) · arXiv 2602.10680 | идейно | 10 | синтетика (OPEN) | LIGHT, переписать за вечер (теория, идейно близко к Dice vs ICC) |
| 39 | [TwinWeaver (digital twin)](https://openreview.net/forum?id=oWqRhYpJb6) · arXiv 2601.20906 | E | 12 | GENIE (REST) | данные и код |
| 40 | [Marrying SDOH (digital twin)](https://openreview.net/forum?id=9GKURFWGeh) · arXiv 2605.09771 | E | 12 | UK Biobank (REST) | данные и код |

## ИЗБЕГАТЬ (сломано для репродукции)

| # | Статья | Причина |
|---|---|---|
| 41 | [S2M-Net](https://openreview.net/forum?id=eh48NIgu9z) · arXiv 2601.01285 | отозвана с arXiv авторами |
| 42 | [SzCORE](https://openreview.net/forum?id=3vhn9kVVY5) · arXiv 2505.18191 · [esl-epfl/szcore](https://github.com/esl-epfl/szcore) | ключевая цифра на закрытом held-out тесте |
| 43 | [SleepMaMi](https://openreview.net/forum?id=e9hvlHlUas) · arXiv 2602.07628 / [SleepLM](https://openreview.net/forum?id=9wpwfSJCp9) · arXiv 2602.23605 | проприетарные PSG-данные, кода нет |
| 44 | [Cardio-mmFlow](https://openreview.net/forum?id=e43JwTIb2V) | открытых данных нет, кода нет |

---

# И БОЛЬШЕ — смежные кластеры (найдены, но вне ядра 44)

## EEG foundation / brain-decoding (адъяцентно, НЕ ядро Kate)
Большой кластер, EEG это не её ниша. Список на случай, если тема понадобится:
- [PATCHCODE](https://openreview.net/forum?id=NWcZ5vualM) · [Let EEG Models Learn EEG](https://openreview.net/forum?id=TP8OuKKmsf) (arXiv 2605.21280) · [EEG-FM-Bench](https://openreview.net/forum?id=vGeNaFHdET) (2508.17742) · [EmBrace](https://openreview.net/forum?id=BJ5rYj8O8W) · [MEG-XL](https://openreview.net/forum?id=cefix4VmhS) (2602.02494)
- [ViEEG](https://openreview.net/forum?id=DkK7GUr8n3) (2505.12408) · [PCRNet](https://openreview.net/forum?id=89MeH5Ax8r) · [EEG Hyperbolic MoC](https://openreview.net/forum?id=VSn4wLFd2p) (2604.12579) · [ROAMM](https://openreview.net/forum?id=zqLPdt09fE) · [See the Emotion](https://openreview.net/forum?id=AdTBZw18aH) · [Harnessing Spectrum Video](https://openreview.net/forum?id=iCjSoADDPs)
- [BIT-LLM (fMRI-to-Text)](https://openreview.net/forum?id=HZhlRnJOHq) · [KAST-BAR](https://openreview.net/forum?id=Ee4j4zMir5) (2605.13133) · [BrainJanus](https://openreview.net/forum?id=nJxailqsUW) · [Mind-Omni](https://openreview.net/forum?id=3gCdh3u2GK) (2605.29591) · [NeuroCLUS](https://openreview.net/forum?id=pFweJM4Uw8) · [Physiology as Language (resp→EEG sleep)](https://openreview.net/forum?id=ceErPsyO38) (2602.00526)

## MIMIC / EHR-траектории (близко к process-mining углу, датасеты вероятно MIMIC)
- [PathwayLLM (sepsis, clinical trajectory)](https://openreview.net/forum?id=oB7gZX7MKP) — ближе всего к process/pathway
- [Time-Conditioned Foreseeing (EHR foundation, irregular events)](https://openreview.net/forum?id=IalpB5Mzaz)
- [STT-LLM (longitudinal clinical profiles)](https://openreview.net/forum?id=oLtn3L43Ej)
- [HEARTS (LLM reasoning on health time series)](https://openreview.net/forum?id=qj4EnIvNU4) (2603.06638)

## Биосигналы / wearables (клинические PPG/ECG/сон, не consumer)
- [SGERA (ECG-report)](https://openreview.net/forum?id=iEtOxzAs51) · [A robust PPG foundation model](https://openreview.net/forum?id=JF4045xnhl) (2606.07365) · [Learning Cardiac Latent (VCG)](https://openreview.net/forum?id=hS6iw4PM8K) (2605.31249) · [Cross-Modal Biosignal VAE](https://openreview.net/forum?id=aComqAqP6j)
- (см. также в основном списке: ECG-R1 #10, SIGMA-PPG #11, xMAE #12, OSF #37, ConfSleepNet #7)

## Digital twin (медицинский) — тонко
- TwinWeaver #39, Marrying SDOH #40 (оба в основном списке, данные закрыты). Digital twin мозга/визуализации на ICML 2026 нет.

---

# ПУСТЫЕ НИШИ (проверено, на ICML 2026 практически нет)
Это не «неважно», а «не тот зал» — область недопредставлена, значит потенциал для собственной статьи Kate, а не репродукции:
- **Brain-age / face-age:** 0 статей (проверено по brain age, biological age, chronological age, aging, faceage).
- **Process mining / event logs / MIMICEL:** 0 (это ICPM/BPM/CHIL/ML4H, не ICML).
- **Марковские модели по медицине (HMM/semi-Markov):** 0 по заголовкам.
- **Digital twin мозга/визуализации:** 0.

---

# ВЕРИФИЦИРОВАННЫЕ РЕПОЗИТОРИИ (реальный код виден, GitHub API 2026-07-18)
SwissDataScienceCenter/uqct (54 .py, но данные/веса «on request», зенодо позже) · zygao930/MedCRP-CL (7) · duchenhe/DC-PnPDP (22) · DopamineLcy/EviScreen (25, есть reproduce_directly eval-only путь plus открытые fundus-данные) · qic999/Foundation-VAE (402) · Ashespt/TACO (20) · sunlabuiuc/PyHealth (542) · OneMore1/Omni-fMRI (23) · MedARC-AI/fmri-fm (CortexMAE, 17) · ZonghengGuo/SigmaPPG (23) · PKUDigitalHealth/ECG-R1 (641) · hzhou3/xMAE (10) · esl-epfl/szcore. **README only (кода нет):** By4te/ConfSleepNet_ICML2026, **DirkLiii/TCSeg** (пометка verified была ошибочной).

# ДАТАСЕТЫ, ЧТО У KATE ЕСТЬ / ДОБУДЕТ
TCIA (Pancreas-CT, LIDC-IDRI, UPENN-GBM), OpenNeuro (ds004504), MIMIC-III/IV (credentialed), HCP (registration). Плюс открытые: BraTS, fastMRI, AbdomenCT-1K, CT-RATE, PTB-XL, SleepEDF, ADHD-200, IXI, NSD.

# КАК Я ЭТО ДОБЫЛ (для перепроверки)
Данные статей: `fetchICML2026Papers()` из `icml2026-data.js` на `icml-2026-agent-repro-challenge.static.hf.space`. Claims: `claims.json` plus `claims_anchored.json`, keyed by orid. Логбуки/агенты: HF spaces API `filter=icml2026-repro`, теги `paper-<orid>`. По коду/данным walled-статей: OpenReview за ботозащитой, помечено «вероятно/не подтверждено».
