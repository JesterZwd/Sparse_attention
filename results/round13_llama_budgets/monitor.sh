#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "$0")"
date -Is
if [[ -f status.json ]]; then python -m json.tool status.json; fi
printf '\nRecent log\n'
tail -n 8 run.log
printf '\nGPU utilization, memory, power\n'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
