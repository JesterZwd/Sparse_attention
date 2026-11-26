#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/CoMRAD-round13
experiment=results/round13_llama_budgets
exec >>"$experiment/run.log" 2>&1
exec 9>"$experiment/queue.lock"
flock -n 9
printf '\n%s Llama matrix queue started\n' "$(date -Is)"
while [[ ! -f "$experiment/data.ready" ]]; do
    if ! screen -ls | grep -q '[.]llama-budget-data[[:space:]]'; then
        [[ -f "$experiment/data.ready" ]] && break
        printf '%s Data preparation stopped. See prepare.log.\n' "$(date -Is)"
        exit 1
    fi
    sleep 5
done
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .cache/round13-cloud/image-venv/bin/python -u "$experiment/run_matrix.py"
