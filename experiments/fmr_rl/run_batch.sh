#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${HOME}/ns-3-dev/experiments/fmr_rl"
MODEL_PATH="${1:-$HOME/ns-3-dev/scratch/fmr_ai/models/model.zip}"

mkdir -p "${BASE_DIR}/runs"

declare -a SIM_TIMES=("1s" "2s" "5s")
declare -a TAGS=("smoke" "short" "medium")

for i in "${!SIM_TIMES[@]}"; do
  RUN_ID="fmr_rl_${TAGS[$i]}_$(date +%Y%m%d_%H%M%S)"
  "${BASE_DIR}/run_one.sh" "${RUN_ID}" "${SIM_TIMES[$i]}" "${MODEL_PATH}"
  sleep 2
done

echo "[INFO] batch finished"
