# HANDOFF — ICML-2026-Repro Ground Truth (agentic-research)

**Дата:** 2026-07-31 | **Ветка:** `hermes-agentic-research` | **Статус:** готов к запуску экспериментов

## Что уже построено (ground truth для оценки агентов)

Цель проекта: собрать корпус ground truth воспроизведений по 2005 статьям (все домены),
использовать для оценки других моделей-агентов по cost/quality, лучшего спустить на medical.

### Слой 1 — label-GT (`corpus.py`, коммиты 37f0af0, 456eed0)
- `out/corpus_all.json` (2005 статей) + `out/corpus_medical.json` (240).
- hard-GT: консенсус ≥3 участников, ≥60% согласия. soft-GT: эмпирическое распределение вердиктов для disputed-слотов.
- **CRITICAL:** area/subarea для ALL есть в `hf_dataset/papers.json` (HF датасет `ICML-2026-agent-repro/challenge`), НЕ в `audit_snapshot/public_repro_audit.csv` (там только 240 medical).

### Слой 2 — process-GT Уровень А (`process_gt.py`, коммит 4bbba9e + 3c079ee)
- Структура решения из логбуков: mutation-тест, seeds, compute_cost, evidence_layers.
- Частичный порядок доминации (исправлено на СТРОГИЙ Pareto — коммит 3c079ee, убрал взаимную доминацию).
- `out/process_gt_all.json` (11 статей) + `out/process_gt_medical.json` (8). Покрытие мало — парсили только codex-ветку (34 статьи).

### Слой 3 — process-GT Уровень Б (`process_gt_levelB.py`, коммиты b5fd3b9, 3c079ee)
- 23 мед. claim'а оценены `tencent/hy3:free` (через hermes-router localhost:8319, БЕСПЛАТНО) на validity: "реально воспроизведено или просто убедительно написано".
- `out/process_gt_levelB_medical.json`. Оценки 0.0–1.0 (NOT ATTEMPTED → 0.0, запущенное с цифрами → 0.9–1.0).
- Скрипт имеет resume (не перезапрашивает hy3, если validity уже есть).

### Слой 4 — отбор для прогона агентов (`selected_for_runs.json`, коммит 81e0a9f)
- **45 статей** (ALL, не medical), CPU-feasible по логбукам лидерборда, нетeоретических, ВСЕ disputed, 7 areas.
- Точная мапа `orid→logbook` через endpoint лидерборда `filter=icml2026-repro` (2046 логбуков, тег `paper-<orid>` в `logbook.json`).
- Каждая запись: orid, title, area, n_claims, disputed, space (HF-логбук), hw_signals.
- Проверено вручную топ-10 через реальные логбуки — 9/10 реально CPU.

## Где что лежит
```
groundtruth/corpus.py                      # Слой 1 (hard+soft GT)
groundtruth/process_gt.py                   # Слой 2 (структура + иерархия)
groundtruth/process_gt_levelB.py            # Слой 3 (hy3 validity scoring)
groundtruth/out/corpus_{all,medical}.json   # Слой 1
groundtruth/out/process_gt_{all,medical}.json + levelB_medical.json
groundtruth/out/selected_for_runs.json      # 45 статей для экспериментов
groundtruth/hf_dataset/                     # papers.json, challenge.json, abstracts.json (HF датасет, скачано локально)
/tmp/lb_cache/orid2space.json               # точная мапа orid→HF-space (2046 записей)
experiments/run02_hermes_leaf/.../logbook.md # пример логбука Хермес-агента (1 статья)
```

## Как запускать эксперименты в другом чате
1. Взять `out/selected_for_runs.json` — это пул 45 статей.
2. Прогнать каждую статью кандидат-моделью (разные AI-структуры исследователя): дать claim-страницы из логбука (`space` поле → `https://huggingface.co/spaces/<space>`), получить решение.
3. Валидировать через process-GT: извлечь структуру решения (mutation/seeds/cost), сравнить с `process_gt_*.json`, оценить validity через hy3 (`process_gt_levelB.py` как шаблон, переделать под ALL).
4. Метрика: не совпадение лейбла, а **минимально достаточное решение по компьюту** (Pareto-фронт) + validity-оценка hy3.

## Открытые вопросы / блокеры
- **Браузер Хермес не привязан** (browser-use cloud 404, нет ключа в auth.json; локальный Chrome на localhost:9222 жив, но Хермес не цепляется без `cdp_url`+рестарт шлюза). Для парсинга HF хватало `web_extract`/прямого urllib. TODO в памяти: настроить browser-use ключ ИЛИ cdp_url.
- **Покрытие process-GT мало** (11/2005 all, 8/240 medical) — парсили только codex-ветку. Для полного покрытия нужно извлечь ВСЕ логбуки (через `filter=icml2026-repro` endpoint — уже работает, 2046 логбуков).
- **medical финал** — когда агент отобран на ALL, спустить на medical (240 статей, врач вручную валидирует).
- **Не коммичено:** `experiments/run02_hermes_leaf/...` (M), `best_solutions.json`, `coverage.*`, `solutions_by_paper.json` (M) — это рабочие файлы, не трогать без проверки.

## Быстрый старт для нового чата
> "Продолжи icml-2026-repro. Ground truth готов: corpus_all.json (label-GT), process_gt_*.json (структура+иерархия), selected_for_runs.json (45 статей для прогона). Нужно запустить кандидат-модели на 45 статьях и оценить через process-GT. hy3:free доступен через hermes-router localhost:8319 (бесплатно). Браузер не привязан — используй web_extract/urllib для HF."
