# HANDOFF — ICML-2026-Repro: candidate auto-research agent (open-source, local)

**Дата:** 2026-07-31 | **Ветка:** `hermes-agentic-research` | **Статус:** сменили постановку, ищем real open-source агента

> См. также `groundtruth-handoff.md` (там — слой ground truth + selected_for_runs.json, уже готово).
> Этот файл — про КАНДИДАТ-АГЕНТА, который гоним на 45 статьях и оцениваем через process-GT.

## 0. Что УЖЕ готово (не трогать)
- `groundtruth/out/corpus_all.json` (label-GT), `process_gt_{all,medical}.json` (структура+иерархия),
  `selected_for_runs.json` (**45 статей** для прогона: ALL, CPU-feasible, disputed, 7 areas).
- `process_gt.py` (структура) + `process_gt_levelB.py` (hy3 validity). hy3:free → `localhost:8319/v1`, бесплатно.
- ПРОБЕЛ (из старого handoff, не блокер): `process_gt_all.json` покрывает только 11 статей (парсили codex-ветку).
  Для новой постановки это не критично — структуру будем извлекать из логбуков кандидат-агента.

## 1. Что ВЫЯСНИЛОСЬ в сеансе 2026-07-31 (критично — не повторяй ошибок)
1. Задача НЕ «прогнать наши run01/run02 на 45». run01=plain Codex (заблокирован, нужен интерактивный
   `codex login` от Кейт — Кейт решила НЕ проверять), run02=Hermes leaf/hy3 (наш собственный агент).
2. **alphaXiv/OpenResearch (`orx`) отпадает как кандидат.** `orx` — клиент над ИХ облаком
   (OpenResearch API), runs идут на их compute (managed GPU/CPU, Modal, HF, k8s). Требует аккаунт
   (`orx login` → браузер OAuth) + compute-кредиты. У Кейт кредитов НЕТ → на их серверах ничего не
   запустить. Токен `opr_pat_...` дан, но нужен только для API-доступа, не для бесплатного compute.
   Локально «бесплатно через hy3:free» orx НЕ работает — агент живёт в их облаке, не на нашей машине.
3. **Скиллы Hermes (`auto-research-benchmark-design`, `paper-claim-reproduction`,
   `icml-repro-groundtruth`) — это МЕТОДОЛОГИЯ бенчмарка/воспроизведения, НЕ кандидат-агент.**
   Кейт прямо: «то что ты нашел скилы не является приемлимым бейзлайном статьи для публикации».
   Скиллы используем для ДИЗАЙНА ОЦЕНКИ, но сам candidate должен быть реальным агентом.

## 2. НОВАЯ ПОСТАНОВКА (от Кейт)
- НЕ «воспроизведение anchored claim'а статьи», а **готовая идея/гипотеза из статьи, которую надо
  доказать или опровергнуть ПОД ОДИН CLAIM**. Агент сам формулирует проверяемое утверждение и гонит
  его через mutation-тест (дисциплина из `paper-claim-reproduction`: вердикт
  `verified`/`falsified`/`inconclusive`, обязательно mutation-тест, Evidence boundary).
- Кандидат-агент для ПУБЛИКАЦИИ = **реальный open-source auto-research агент с кодом** (из arXiv/GitHub),
  который мы реально прогоняем локально на 45 статьях и оцениваем через process-GT.

## 3. Требования к кандидат-агенту
- open-source, есть код на GitHub;
- **локально-запускаем** (не только облако вендора);
- позволяет указать OpenAI-compatible `base_url` → наш hermes-router `http://localhost:8319/v1`,
  модель `hy3` (free через Nous) — чтобы прогоны были БЕСПЛАТНЫ;
- пишет воспроизводимые артефакты (скрипты + логбук) в формате, адаптируемом к `extract_signals` process-GT.

## 4. Кандидаты на проверку (из обзора `ai-scientist` + arXiv) — статус recon
- `SamuelSchmidgall/AgentLaboratory` — staged multi-agent lab; обычно под Claude/OpenAI, но может
  принимать vLLM/Ollama (OpenAI-compatible). НУЖНО проверить локальный запуск с кастомным base_url.
- `SakanaAI/AI-Scientist-v2` — experiment-tree search; под их модели, но есть run-скрипты.
- `aiming-lab/AutoResearchClaw` — self-healing multi-agent; локальный запуск, под свою модель.
- (более лёгкие open-source агенты — см. разведку; возможно проще привязать к hy3:free).
> ВАЖНО: design-критика V6 (`01 Design and critique.md`) предупреждает — системы типа AI Scientist-v2 /
> Agent Laboratory заточены под «написать статью», не под reproduction конкретного claim, и низкий score
> может быть mismatch формата, а не слабость архитектуры. Нужен adapter-протокол (input bundle +
> output schema), и report «adapter effort» как отдельная величина.

## 4b. МЕТРИКА: токены / скорость / промпты (как считать для публикации)
Решение Кейт: гоним 3 open-source агента + наш Hermes-leaf (со скиллами как методология) как
сравнение. Нужно считать на КАЖДОГО: токены, wall time, $, промпты.

**Точка учёта = наш hermes-router (`localhost:8319`).** Все 3 агента + leaf стучатся в него
(OpenAI-compatible, model=hy3, free через Nous). Значит usage считаем ЦЕНТРАЛИЗОВАННО в роутере.

1. **Токены + латентность:** дописать в `router.py` (Flask, 5682 строки) middleware-лог
   `/v1/chat/completions` → append JSONL `usage_log.jsonl`:
   `{ts, model, prompt_tokens, completion_tokens, latency_ms, candidate_tag, orid}`.
   `candidate_tag` передаём через кастомный заголовок `X-Candidate: <name>` (агент шлёт его при
   конфиге base_url, либо мы добавляем в обёртку). `orid` — из контекста промпта (regex по
   openreview id) или отдельный заголовок `X-Orid`.
2. **$ (стоимость):** у нас hy3:free → $0.00. Честно пишем $0, НО считаем «если бы платили» по
   тарифу hy3/Nous для score-per-dollar (design-критика V2/V3). Ренат-контракт: free если
   live-tested.
3. **Промпты (сохранение):** каждый кандидат пишет ПОЛНЫЙ текст system+user промптов в
   `experiments/<candidate>/<orid>/prompts/` (это часть методологии, цитируем в статье).
   Плюс зеркалим в usage-лог первые N символов prompt для сверки.
4. **Артефакты → process-GT:** logbook/скрипты каждого агента адаптируем в `extract_signals`
   (mutation/seeds/cost) + hy3 validity (`process_gt_levelB`).
5. **Структура experiments/:**
   ```
   experiments/
     run01_plain_codex/        # (старое, заблокирован)
     run02_hermes_leaf/        # (старое, наш агент, 1 статья Xf9hJMGwDd)
     arc/                      # AutoResearchClaw  (candidate 1)
     aisv2/                    # AI-Scientist-v2  (candidate 2)
     agentlab/                 # AgentLaboratory  (candidate 3)
     hermes_leaf_skills/       # наш leaf + скиллы (baseline-сравнение)
     <candidate>/<orid>/
        input_bundle.json
        prompts/               # сохранённые промпты
        logbook/ (или agent-specific artifacts)
        results/
        metrics.json           # tokens, wall_s, $, validity, verdict
   ```
6. **Билл-гейт ОБЯЗАТЕЛЕН** перед запуском любого чужого репо (чужой код локально). Запущен
   2026-07-31 на 3 репозитория.

## 5. Блокеры / открытые вопросы
- **Который из open-source агентов реально поднимается локально с hy3:free через роутер?** Нужен
  recon (проверить README на base_url/local model) + пилот (1 агент × 1 статья) ДО масштабирования.
- **Масштаб:** 45 статей × ~60-90 мин CPU = десятки часов. i7-8550U (WSL) упрётся в swap —
  см. скилл `scientific-agent-workflow` (Hermes=dispatcher, swap-монитор, guarded launch).
  Пилот сначала (3-5 статей), потом батч в фоне.
- **Билл-гейт:** установка стороннего агента = исполнение чужого кода на нашей машине. По правилам
  Билл должен дать read-only security-взгляд на репо агента (prompt-injection / exfil / сетевые вызовы)
  ДО запуска. Это НЕ скилл (community-skill-porting не применим напрямую), но принцип тот же:
  security-аудит чужого кода перед исполнением.

## 6. Быстрый старт для нового чата
> «Продолжи icml-2026-repro. Ground truth готов (corpus_all.json, process_gt_*.json,
> selected_for_runs.json — 45 статей). Нужно прогнать РЕАЛЬНЫЙ open-source auto-research агент
> (с кодом, из arXiv/GitHub) на 45 статьях как кандидата для публикации, оценить через process-GT.
> Постановка: НЕ воспроизведение anchored claim'а, а гипотеза из статьи под один claim
> (доказать/опровергнуть, mutation-тест, Evidence boundary). orx/OpenResearch отпал (нет кредитов на
> compute). Codex заблокирован. hy3:free через localhost:8319 бесплатно. Скиллы Hermes — методология
> оценки, НЕ бейзлайн. Начни с recon open-source агентов + пилота (1 агент × 1 статья). Билл-гейт перед
> запуском чужого кода обязателен.»
