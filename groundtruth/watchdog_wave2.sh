#!/bin/bash
# watchdog_wave2: wait for tRsnpaRO0m + LJdacnMXkr logbooks, score+publish both.
set +e
RUN=/home/kate/projects/02_academia/icml-2026-repro/experiments/run03_hermes_leaf_pilot
REPO=/home/kate/projects/02_academia/icml-2026-repro
PY=/home/kate/.venvs/neurofm_py311/bin/python
NEXT=$RUN/NEXT_WAVE.txt
echo "wave2 watchdog start $(date)" > "$NEXT"

for i in $(seq 1 120); do
  if [ -f "$RUN/tRsnpaRO0m_arm2b/logbook.md" ] && [ -f "$RUN/LJdacnMXkr_arm2b/logbook.md" ]; then
    echo "both logbooks at $(date)" >> "$NEXT"
    for o in tRsnpaRO0m LJdacnMXkr; do
      cd "$REPO" && $PY experiments/run03_hermes_leaf_pilot/score_run.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
      $PY groundtruth/local_to_trackio.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
      $PY groundtruth/publish_local.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
    done
    echo "WAVE2_DONE $(date)" >> "$NEXT"
    echo "NEXT: launch Wave 3 (KqMqJpSMnQ + omkG80XURl)" >> "$NEXT"
    exit 0
  fi
  sleep 60
done
echo "TIMEOUT wave2" >> "$NEXT"
exit 1
