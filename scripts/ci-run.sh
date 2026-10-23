#!/usr/bin/env bash
# Runs a command; if it fails, repeats the last lines of its output as a GitHub annotation so the
# failure is readable from the run's summary page.
set -o pipefail
log=$(mktemp)
"$@" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  msg=$(tail -n 40 "$log" | sed -e 's/%/%25/g' | sed -e ':a;N;$!ba;s/\n/%0A/g')
  echo "::error title=$1 failed::${msg}"
fi
exit "$rc"
