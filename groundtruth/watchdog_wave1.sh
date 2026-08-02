#!/bin/bash
# watchdog_wave1: wait for TBSyYj4VV6 + l35QweVxgn logbooks, score both, publish both, write NEXT_WAVE.txt
set +e
RUN=/home/kate/projects/02_academia/icml-2026-repro/experiments/run03_hermes_leaf_pilot
REPO=/home/kate/projects/02_academia/icml-2026-repro
PY=/home/kate/.venvs/neurofm_py311/bin/python
NEXT=$RUN/NEXT_WAVE.txt
echo "wave1 watchdog start $(date)" > "$NEXT"

for i in $(seq 1 60); do
  if [ -f "$RUN/TBSyYj4VV6_arm2b/logbook.md" ] && [ -f "$RUN/l35QweVxgn_arm2b/logbook.md" ]; then
    echo "both logbooks at $(date)" >> "$NEXT"
    for o in TBSyYj4VV6 l35QweVxgn; do
      cd "$REPO" && $PY experiments/run03_hermes_leaf_pilot/score_run.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
      $PY groundtruth/local_to_trackio.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
      $PY groundtruth/publish_local.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
    done
    echo "WAVE1_DONE $(date)" >> "$NEXT"
    echo "NEXT: launch Wave 2 (tRsnpaRO0m + LJdacnMXkr) via delegate_task" >> "$NEXT"
    exit 0
  fi
  sleep 60
done
echo "TIMEOUT wave1" >> "$NEXT"
exit 1
