#!/bin/bash
# Disk watchdog while the user is away (2026-10-14): every 5 minutes, if D: or E: has less than 4 GB free, delete
# obsolete checkpoints of this project only, one at a time, oldest-value first, and log each deletion.  Configs,
# reports, scores and results are never touched.  Candidates, in order:
#   1. E:/ib_checkpoints/m_main_cadapt (the change-1 treatment checkpoint), only after its probes are done;
#   2. D:/ib_checkpoints/mother_v2_discount (a finished mother-world run; its transcript and config stay);
#   3. D:/ib_checkpoints/d5b/c9.pt, then c10.pt, once d5b has finished (c11 / last stay);
#   4. E:/ib_checkpoints/m_main/best.pt;
#   5. the oldest every-10th m_main checkpoint (c10, c20, ...), never c1 and never the latest two.
# Stop it with: powershell Stop-Process on the bash running disk_watchdog.sh.
LOG=/e/fly_stream_overflow/disk_watchdog.log
free_gb() { df -BG "$1" | tail -1 | awk '{gsub("G","",$4); print $4}'; }
pick() {
  if grep -q "change 1 treatment done" /e/fly_stream_overflow/m_main/status 2>/dev/null && [ -f /e/ib_checkpoints/m_main_cadapt/last.pt ]; then
    echo /e/ib_checkpoints/m_main_cadapt/last.pt; return; fi
  [ -f /d/ib_checkpoints/mother_v2_discount/last.pt ] && { echo /d/ib_checkpoints/mother_v2_discount/last.pt; return; }
  if grep -q "d5b finished\|d5b done" /e/fly_stream_overflow/d5b/status /e/fly_stream_overflow/m_main/status 2>/dev/null; then
    for c in 9 10; do [ -f /d/ib_checkpoints/d5b/c$c.pt ] && { echo /d/ib_checkpoints/d5b/c$c.pt; return; }; done
  fi
  [ -f /e/ib_checkpoints/m_main/best.pt ] && { echo /e/ib_checkpoints/m_main/best.pt; return; }
  latest=$(ls /e/ib_checkpoints/m_main/c*.pt 2>/dev/null | sed 's/.*\/c\([0-9]*\)\.pt/\1/' | sort -n | tail -2 | head -1)
  for c in $(ls /e/ib_checkpoints/m_main/c*.pt 2>/dev/null | sed 's/.*\/c\([0-9]*\)\.pt/\1/' | sort -n); do
    [ "$c" -gt 1 ] && [ "$c" -lt "${latest:-0}" ] && { echo /e/ib_checkpoints/m_main/c$c.pt; return; }
  done
}
while true; do
  for drive in /d /e; do
    f=$(free_gb $drive)
    if [ "${f:-99}" -lt 4 ]; then
      victim=$(pick)
      if [ -n "$victim" ]; then
        rm -f "$victim"
        echo "$(date '+%m-%d %H:%M') $drive had ${f}G free: deleted $victim" >> $LOG
      else
        echo "$(date '+%m-%d %H:%M') $drive had ${f}G free: nothing obsolete left to delete" >> $LOG
      fi
    fi
  done
  sleep 300
done
