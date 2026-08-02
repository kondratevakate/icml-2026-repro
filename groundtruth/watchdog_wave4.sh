#!/bin/bash
# watchdog_wave4: wait for uiw8P2JGbW + ugjBMARbyt logbooks, score+publish both.
set +e
RUN=/home/kate/projects/02_academia/icml-2026-repro/experiments/run03_hermes_leaf_pilot
REPO=/home/kate/projects/02_academia/icml-2026-repro
PY=/home/kate/.venvs/neurofm_py311/bin/python
NEXT=$RUN/NEXT_WAVE.txt

for i in $(seq 1 120); do
  if [ -f "$RUN/uiw8P2JGbW_arm2b/logbook.md" ] && [ -f "$RUN/ugjBMARbyt_arm2b/logbook.md" ]; then
    echo "wave4 both logbooks at $(date)" >> "$NEXT"
    for o in uiw8P2JGbW ugjBMARbyt; do
      cd "$REPO" && $PY experiments/run03_hermes_leaf_pilot/score_run.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
      $PY groundtruth/local_to_trackio.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
      $PY groundtruth/publish_local.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
    done
    echo "WAVE4_DONE $(date)" >> "$NEXT"
    echo "NEXT: launch Wave 5 (rZTiFcDihH + vqxprtjuKH)" >> "$NEXT"
    exit 0
  fi
  sleep 60
done
echo "TIMEOUT wave4" >> "$NEXT"
exit 1
