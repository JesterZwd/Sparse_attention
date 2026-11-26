#!/usr/bin/env bash
set -euo pipefail

repo=${COMRAD_REPO:-/root/autodl-tmp/CoMRAD-round13}
run=${COMRAD_RUN:-$repo/.cache/comrad-round14}
py="$repo/.cache/round13-cloud/image-venv/bin/python"

printf '%s\n' "$$" > "$run/package.pid"
printf '%s\n' waiting_for_gpu_queue > "$run/package.status"
trap 'printf "%s\n" failed > "$run/package.status"' ERR

read -r worker < "$run/mv32.pid"
while kill -0 "$worker" 2>/dev/null && [[ $(ps -o stat= -p "$worker") != Z* ]]; do sleep 5; done
[[ $(cat "$run/mv32.status") == 'Qwen3 32K MV complete' ]]
[[ $(cat "$run/fixed.status") == 'Six fixed panels complete' ]]
[[ $(cat "$run/clock.status") == complete ]]
printf '%s\n' verifying > "$run/package.status"

cd "$run"
"$py" report_calibration.py output
"$py" verify_vectors.py output --receipt vector_receipt.json
tar -czf round14_results.tar.gz.tmp output clock/clock_summary.csv clock/clock_step_timing.csv clock/receipt.json \
    calibration_summary.csv calibration_receipt.json vector_receipt.json \
    fixed_backend.py monitor.sh replay_clock.py report_calibration.py run_calibration.py run_qwen3_32k_mv.py \
    test_fixed_backend.py verify_vectors.py package_results.sh qwen3_32k_mv_prepare.json
mv round14_results.tar.gz.tmp round14_results.tar.gz
sha256sum round14_results.tar.gz > round14_results.tar.gz.sha256
printf '%s\n' complete > package.status
