#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/ns3-fmr"

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

export BW_LIST="40"
export FMR_UE_COUNTS="3,5,7"
export FMR_SEEDS="$(grep -Eo '[0-9]+' seeds_100_master_20260618.txt | xargs)"
export FMR_MAX_WORKERS_CLASSIC=1
export SIM_TIME="30s"
export FMR_ENABLE_MOBILITY=1
export FMR_POSITION_MODE="fixed_profile"
export FMR_PHASE_DURATIONS="6,6,6,6,6"
export FMR_PHASE_LAMBDAS="200,350,550,250,650"
export FMR_PHASE_FLOWS="1,1,1,1,1"

python3 run_fmr.py \
  --run-id expB_fmr_40mhz_100seeds_30s_ues357_fixedmob \
  --only-mode fmr_rl \
  --skip-existing
