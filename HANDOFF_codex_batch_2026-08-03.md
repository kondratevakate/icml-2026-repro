# HANDOFF — codex-batch publish (2026-08-03, Hermes main)

## TL;DR (честно)
Я выложил 17 codex-v1 logbook'ов на HF Spaces через самописный генератор
`build_icml_logbook.py`. **Внешний судья ставит им `pending verdict`** — то есть
не может их проскорить. Это хуже 0: моя структура не парсится судьёй.
**Нужно остановиться и выяснить формат судьи, прежде чем перепубликовывать.**

## Что сделано (факты)
- 20 codex-v1 logbook'ов лежат локально в `/tmp/codex-wt/*_arm2b/` (20 папок).
  Все досчитались агентами в этой сессии (50+ вердиктов по claim'ам, честные,
  НЕ ленивые — скан на `no access`/`toy` дал 0 ленивых паттернов).
- 17 из них выложены на HF (через старый `local_to_trackio.py`, потом через
  новый `build_icml_logbook.py` после фикса структуры).
  caml_arm2b НЕ выложен (нет logbook.md, только 5 json).
  camegrad_arm2b = дубль came_grad (обе → repro-came-grad-mathematical-audit).
- Текущий статус на HF: большинство `pending verdict` (судья не скорит).
  supgcl показал `refuted` (капслок-баг: третья колонка таблицы бралась как
  verdict целиком, а не слово).

## Почему судья ставит pending (гипотезы, НЕ доказано)
Судья (внешний, не наш) парсит Space по контракту, который я НЕ знаю точно.
Мои Space имеют структуру: `pages/summary`, `pages/claim-N-*`, `pages/conclusion`,
`logbook.json` root.children=[summary, claim-1..N, conclusion], tags=
["icml2026-repro", "paper-<orid>"]. Это скопировано с 3 эталонов 12/12:
  - repro-conditional-coverage-diagnostics-for-conformal-prediction
  - repro-linear-bandits-beyond-inner-product-spaces-the-case-of-bandi
  - repro-bridging-the-gap-between-average-and-discounted-td-learning
Но судья всё равно ставит pending → либо моя структура отличается от эталонов
в деталях, которые судья читает (напр. cell-формат, poster, точные названия
страниц, или содержимое verdict-таблицы), либо нужен challenge-orid в теге
(я ставил локальный `paper-globalhealthatlas`, а не `paper-FP23eFYhAy` и т.п.).

## Что НЕ сделано / заблокировано
- НЕ выяснен точный формат, который судья парсит (нет спеки challenge-judge).
- НЕ получен рейтинг от судьи для НИ ОДНОГО моего Space (все pending).
- `build_icml_logbook.py` в процессе правок был испорчен патчем:
  двойное `C1` в названиях claim'ов, em-dash в regex не матчит. Код сейчас
  в промежуточном (сломанном) состоянии — НЕ использовать как есть.

## Ключевые файлы
- `/home/kate/projects/02_academia/icml-2026-repro/groundtruth/build_icml_logbook.py`
  — самописный генератор (СЛОМАН, чинить или переписать).
- `.../groundtruth/rewrite_logbook_format.py` — normalize v1 → `## Claim N` секции.
- `.../groundtruth/local_to_trackio.py` — СТАРЫЙ генератор (неправильная структура).
- `.../groundtruth/publish_local.py` — публикация (ref-copy runtime + upload).
- `.../groundtruth/verify_space.py` — техническая проверка (файлы есть), НЕ судья.
- `.../groundtruth/space_map.json` — orid → Space (локальный orid, не challenge).
- `.../experiments/run03_hermes_leaf_pilot/validate_icml_logbook.py` — внутренний
  валидатор (НЕ судья; показывает 5 errors на моих Space).
- Локальные logbook'и: `/tmp/codex-wt/*_arm2b/logbook.md` (20 шт).

## Что делать дальше (приоритет)
1. **Выяснить формат судьи.** Спросить Кейт: есть ли спецификация, которую
   судья парсит? Или какой из ВЫЛОЖЕННЫХ (не эталонных) Space судья НЕ поставил
   в pending — тот, что прошёл, даст ключ к формату.
2. Не перепубликовывать, пока не понятен контракт.
3. Починить/переписать генератор под точный контракт.
4. Один тестовый Space → проверка у судьи → только если ОК, раскатывать.

## Важные правила Кейт (не нарушать)
- НЕ создавать дубли Space (EDIT существующий по paper-<orid>).
- НЕ фейк-вердикты (честно или inconclusive).
- Публиковать ПО ОДНОМУ, проверять результат (не циклом вслепую).
- Внешний судья — источник истины о score, не наша структура.
