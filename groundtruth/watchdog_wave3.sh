#!/bin/bash
# watchdog_wave3: wait for KqMqJpSMnQ + omkG80XURl logbooks, score+publish both.
set +e
RUN=/home/kate/projects/02_academia/icml-2026-repro/experiments/run03_hermes_leaf_pilot
REPO=/home/kate/projects/02_academia/icml-2026-repro
PY=/home/kate/.venvs/neurofm_py311/bin/python
NEXT=$RUN/NEXT_WAVE.txt

for i in $(seq 1 120); do
  if [ -f "$RUN/KqMqJpSMnQ_arm2b/logbook.md" ] && [ -f "$RUN/omkG80XURl_arm2b/logbook.md" ]; then
    echo "wave3 both logbooks at $(date)" >> "$NEXT"
    for o in KqMqJpSMnQ omkG80XURl; do
      cd "$REPO" && $PY experiments/run03_hermes_leaf_pilot/score_run.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
      $PY groundtruth/local_to_trackio.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
      $PY groundtruth/publish_local.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
    done
    echo "WAVE3_DONE $(date)" >> "$NEXT"
    echo "NEXT: launch Wave 4 (uiw8P2JGbW + ugjBMARbyt)" >> "$NEXT"
    exit 0
  fi
  sleep 60
done
echo "TIMEOUT wave3" >> "$NEXT"
exit 1
