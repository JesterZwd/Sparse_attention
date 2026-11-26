#!/usr/bin/env bash
set -euo pipefail

repo=${COMRAD_REPO:-/root/autodl-tmp/CoMRAD-round13}
run=${COMRAD_RUN:-$repo/.cache/comrad-round14}

snapshot() {
    date -Is
    printf '\nExperiment workers\n'
    for pidfile in "$run"/*.pid; do
        [[ -f "$pidfile" ]] || continue
        read -r pid < "$pidfile" || continue
        if kill -0 "$pid" 2>/dev/null && [[ $(ps -o stat= -p "$pid") != Z* ]]; then status=running; else status=exited; fi
        printf '%s  %s  pid=%s\n' "$(basename "$pidfile" .pid)" "$status" "$pid"
    done
    for stage in "$run"/*.status; do
        [[ -f "$stage" ]] || continue
        printf '%s  ' "$(basename "$stage" .status)"
        cat "$stage"
    done
    printf '\nSaved prompt bundles\n'
    for dir in "$run"/output/*/*/bundles; do
        [[ -d "$dir" ]] || continue
        printf '%-37s %s\n' "${dir#"$run/output/"}" "$(find "$dir" -maxdepth 1 -name '*.json.gz' | wc -l)"
    done
    printf '\nGPU\n'
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv || true
    for log in "$run"/*.log; do
        [[ -f "$log" ]] || continue
        printf '\n%s\n' "$(basename "$log")"
        tail -n 8 "$log"
    done
}

case ${1:---watch} in
    --once) snapshot ;;
    --watch)
        while true; do
            if [[ -t 1 && ${TERM:-dumb} != dumb ]]; then printf '\033[H\033[2J'; fi
            snapshot
            sleep "${COMRAD_MONITOR_INTERVAL:-5}"
        done ;;
    *) printf 'Usage: bash %s [--once|--watch]\n' "$0" >&2; exit 2 ;;
esac
