# Процедура выкладки логбуков (ICML-2026 repro challenge)

Единый воспроизводимый пайплайн для любого arm (arm1…arm4). Каждый arm пишет плоский
`logbook.md`, затем запускает `publish_logbook.py`, который строит Trackio-структуру,
прогоняет официальный валидатор и публикует в HF Space.

## Что делает каждый arm (минимально)
1. Воспроизвести claims статьи, писать артефакты в `<orid>/results/`, `verify_claim*.py`, `check_reproducibility.py`.
2. Написать `<orid>/logbook.md` в **плоском формате** (образец — любой из run03):
   - `# logbook.md — ...` + мета (paper, orid, env).
   - `## Summary table` с колонками `| Claim | Verdict | One-line evidence |` и verdict'ами
     `**verified**` / `**falsified**` / `**toy**` / `**inconclusive**` (валидатор и `publish_logbook.py`
     извлекают verdict отсюда).
   - `## Claim N — <title>` на каждый claim: source (секция/строка), script, числа, mutation-тест,
     `**Verdict: ...**`.
   - `## Evidence boundary` (что НЕ покрыто — отдельным разделом, не внутри вердикта).
   - `## Artifacts` (список файлов).
3. Запустить публикацию (см. ниже).

## Предварительные требования (один раз на машине)
```bash
# base venv run03 уже содержит numpy/scipy/sympy/torch; добавляем publish-deps:
/tmp/run03_base_venv/bin/pip install -r experiments/run03_hermes_leaf_pilot/requirements_publish.txt
# HF token (уже есть):
ls ~/.cache/huggingface/token   # kondratevakate, org ICML-2026-agent-repro
```

## Публикация одной статьи
```bash
cd experiments/run03_hermes_leaf_pilot
/tmp/run03_base_venv/bin/python publish_logbook.py <orid>
#   --no-publish  — только build + validate (без пуша в HF)
#   --all         — все orid из run03_targets.json (кроме skip)
```
Скрипт:
1. Читает `<orid>/logbook.md`.
2. Строит `<orid>/.trackio/logbook/` (pages/index.md, executive-summary, claim-*, conclusion,
   README.md, logbook.json) + `<orid>/.trackio/metadata.json` (теги `icml2026-repro` + `paper-<orid>`).
3. Копирует runtime-файлы (logbook.js, index.html, logbook.css, bucket-icon.svg) из `ccd/.trackio/logbook/`.
4. `trackio logbook sync` → регенерирует site-файлы.
5. Официальный `validate_icml_logbook.py` (должен вывести `Logbook validation passed`).
6. `trackio logbook publish kondratevakate/repro-<slug>` → HF Space.

## Критичные требования валидатора (почему падает)
- `metadata.json` должен лежать на уровне `.trackio/` (НЕ внутри `logbook/`).
- `index.md` = только `# Reproduction: <title>` + `## Pages` таблица. Никаких ссылок/intro между ними.
- Executive summary: pinned markdown-cell с title `Executive summary` + pinned figure `Reproduction poster`.
- Слаги: `executive-summary` первый, `conclusion` последний, остальные `claim-\d+*`.
- Space-name ≤96 символов, начинается с `repro-`, не совпадает с OpenReview-id.
- `run03_targets.json` содержит все orid + короткие slug'и (укорочены те, что были >96).

## Известные дубликаты (skip)
- `vaApZm6MKM` — уже опубликован как `ccd/` (OpenReview `vaApZm6MKM` = Conditional Coverage
  Diagnostics). В `run03_targets.json` помечен `skip`, скрипт его пропустит.

## Роутер
Оставляем в cheap-режиме (каскады выкл, Gemini исключён) — бэкапы
`router_state.json.bak_20260801_215807` + `.env.bak_20260801_215807` на месте для отката.

## Итоговая таблица валидации
См. `VALIDATION_TABLE.md` (все армы/статьи/метрики).
