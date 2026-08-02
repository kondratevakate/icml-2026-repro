#!/bin/bash
# watchdog_wave5: wait for rZTiFcDihH + vqxprtjuKH logbooks, score+publish both (FINAL wave).
set +e
RUN=/home/kate/projects/02_academia/icml-2026-repro/experiments/run03_hermes_leaf_pilot
REPO=/home/kate/projects/02_academia/icml-2026-repro
PY=/home/kate/.venvs/neurofm_py311/bin/python
NEXT=$RUN/NEXT_WAVE.txt

for i in $(seq 1 120); do
  if [ -f "$RUN/rZTiFcDihH_arm2b/logbook.md" ] && [ -f "$RUN/vqxprtjuKH_arm2b/logbook.md" ]; then
    echo "wave5 both logbooks at $(date)" >> "$NEXT"
    for o in rZTiFcDihH vqxprtjuKH; do
      cd "$REPO" && $PY experiments/run03_hermes_leaf_pilot/score_run.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
      $PY groundtruth/local_to_trackio.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
      $PY groundtruth/publish_local.py "$RUN/${o}_arm2b" >> "$NEXT" 2>&1
    done
    echo "WAVE5_DONE $(date)" >> "$NEXT"
    echo "BUNDLE2_COMPLETE: all 10 papers published. Generate final table (report_batch.py)." >> "$NEXT"
    exit 0
  fi
  sleep 60
done
echo "TIMEOUT wave5" >> "$NEXT"
exit 1
