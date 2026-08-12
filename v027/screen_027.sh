#!/usr/bin/env bash
# screen_027.sh — grind every 0.27 variable against the REAL trigger (load, then drain).
# Runs unattended. Per cell: boot, trigger-test, record, next. The fleet sentinel is paused
# for the duration so it does not fight the experiment; it is restarted at the end.
# Results: one JSON line per cell in $OUT.
set -uo pipefail
OUT=~/Desktop/GLM52-RESTORE-BUNDLE/window-20260810/screen027.jsonl
TRIG=~/Desktop/GLM52-RESTORE-BUNDLE/window-20260810/wedge_trigger.py
CELLS=("$@")
say(){ printf '%s %s\n' "$(date '+%H:%M:%S')" "$*"; }

ssh -o BatchMode=yes gx10-4 'sudo -n systemctl stop gx10-sentinel.service' 2>/dev/null
trap 'ssh -o BatchMode=yes gx10-4 "sudo -n systemctl start gx10-sentinel.service" 2>/dev/null' EXIT

for cell in "${CELLS[@]}"; do
  say "=== CELL $cell ==="
  for n in gx10-1 gx10-2 gx10-3 gx10-4; do
    ssh -o BatchMode=yes "$n" 'docker rm -f vllm_slot >/dev/null 2>&1; sudo -n /usr/local/sbin/gx10-rails.sh >/dev/null 2>&1; sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; true' 2>/dev/null
  done
  ssh -o BatchMode=yes gx10-1 "cd ~/glm-legacy-stack && sed -i 's|launch_gx10.sh|screen_${cell}.sh|' resolve_gid_and_launch.sh && ./resolve_gid_and_launch.sh >/dev/null 2>&1; sed -i 's|screen_${cell}.sh|launch_gx10.sh|' resolve_gid_and_launch.sh" 2>/dev/null

  booted=0
  for i in $(seq 1 30); do
    sleep 30
    curl -s --max-time 5 http://192.168.1.16:8210/v1/models 2>/dev/null | grep -q '"id"' && { booted=1; break; }
    ssh -o BatchMode=yes gx10-1 'docker logs vllm_slot 2>&1 | grep -qE "EngineCore failed|Unsupported architecture|AttributeError|ValueError" && echo X' 2>/dev/null | grep -q X && break
  done
  if [ "$booted" != 1 ]; then
    err=$(ssh -o BatchMode=yes gx10-1 'docker logs vllm_slot 2>&1 | grep -oE "[A-Za-z]+Error: .*" | sort -u | head -1 | cut -c1-110' 2>/dev/null)
    echo "{\"cell\":\"$cell\",\"verdict\":\"BOOT_FAIL\",\"err\":\"${err:-timeout}\"}" | tee -a "$OUT"
    continue
  fi
  say "  serving, running trigger"
  python3 "$TRIG" "$cell" 3 120 2>/dev/null | tail -1 | tee -a "$OUT"
done
say "SCREEN COMPLETE"
