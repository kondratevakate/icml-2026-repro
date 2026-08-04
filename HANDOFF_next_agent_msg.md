# Сообщение для следующего агента (продолжение codex-batch публикаций)

Привет. Мы на середине задачи: публикуем 20 codex-v1 logbook'ов на HF Spaces
(ICML-2026 repro benchmark, arm2). Предыдущий агент (Hermes main) завяз на том,
что **внешний судья ставит выложенным Space `pending verdict`** — то есть не
может их проскорить. Остановились. Ничего больше не публикуй, пока не разберёшься
с форматом судьи.

## Что уже сделано (факты, не гадай)
- 20 logbook'ов лежат локально: `/tmp/codex-wt/*_arm2b/logbook.md` (все досчитались,
  честные, не ленивые).
- 17 из них УЖЕ выложены на HF (через `groundtruth/build_icml_logbook.py`), но
  судья ставит им `pending`. caml_arm2b не выложен (нет logbook.md). camegrad —
  дубль came_grad.
- HANDOFF с деталями: `HANDOFF_codex_batch_2026-08-03.md` (в корне репо).

## Главная проблема
Судья (внешний) не скорит наши Space → `pending`. Моя структура (summary + claim-N
+ conclusion, logbook.json children, tags) скопирована с 3 эталонов 12/12, но
судья всё равно не парсит. Значит либо деталь отличается, либо нужен challenge-orid
в теге (я ставил локальный `paper-globalhealthatlas`, а не `paper-FP23eFYhAy`).

## Твоя задача (по порядку)
1. **НЕ публикуй ничего.** Сначала выясни формат судьи.
2. Спроси Кейт: есть ли спецификация, которую судья парсит? Или какой из
   ВЫЛОЖЕННЫХ (не эталонных) Space судья НЕ поставил в pending — тот даст ключ.
3. Скачай 3 эталона 12/12 и сравни с нашим `build_icml_logbook.py` побайтово
   (cell-формат, poster, названия страниц, содержимое verdict-таблицы). Найди,
   что судья реально читает.
4. Почини/перепиши генератор под точный контракт. ВАЖНО: `build_icml_logbook.py`
   сейчас СЛОМАН (двойное `C1` в названиях, em-dash в regex не матчит) — читай
   файл перед правок.
5. Один тестовый Space → проверка у судьи → только если ОК, раскатывай остальные
   ПО ОДНОМУ.

## Контекстные файлы
- `groundtruth/build_icml_logbook.py` (сломан, чинить)
- `groundtruth/rewrite_logbook_format.py` (normalize v1 → `## Claim N`)
- `groundtruth/publish_local.py` (публикация)
- `groundtruth/space_map.json` (orid → Space; локальный orid, НЕ challenge)
- `groundtruth/verify_space.py` (тех. проверка, НЕ судья)
- `experiments/run03_hermes_leaf_pilot/validate_icml_logbook.py` (внутр. валидатор)

## Правила Кейт
- НЕ дубли Space (EDIT по paper-<orid>).
- НЕ фейк-вердикты.
- Публиковать по одному + проверять результат.
- Внешний судья = истина о score.

Не начинай с публикации. Начни с вопроса Кейт про формат судьи.
