#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${HOME}/ns-3-dev/experiments/fmr_rl"
RUN_ID="${1:-run_$(date +%Y%m%d_%H%M%S)}"
SIM_TIME="${2:-1s}"
MODEL_PATH="${3:-$HOME/ns-3-dev/scratch/fmr_ai/models/model.zip}"

RUN_DIR="${BASE_DIR}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "[INFO] RUN_ID=${RUN_ID}"
echo "[INFO] RUN_DIR=${RUN_DIR}"

cd "${HOME}/ns-3-dev"

pkill -f "ns3.46-fmr-default|scratch/fmr_ai/agent.py|agent_probe.py" || true
sudo rm -f /dev/shm/ns3ai_fmr* /dev/shm/fmr_cpp2py* /dev/shm/fmr_py2cpp* /dev/shm/fmr_lock* || true

cat > "${RUN_DIR}/meta.txt" <<EOF
RUN_ID=${RUN_ID}
SIM_TIME=${SIM_TIME}
MODEL_PATH=${MODEL_PATH}
DATE=$(date)
EOF

(
  export NS_LOG="NrMacSchedulerOfdmaFmr=level_warn|prefix_time"
  ./build/scratch/ns3.46-fmr-default \
    --UseFmr=1 \
    --EnableNs3Ai=1 \
    --AiCppIsCreator=1 \
    --AiProtocol=0 \
    --AiVerbose=1 \
    --AiShmSize=4096 \
    --AiSegmentName=ns3ai_fmr \
    --AiCpp2PyName=fmr_cpp2py \
    --AiPy2CppName=fmr_py2cpp \
    --AiLockableName=fmr_lock \
    --EnableSlotCsv=1 \
    --SlotCsvPath="${RUN_DIR}/default-slot-log.csv" \
    --SlotCsvFlush=1 \
    --FmrTau=0.70 \
    --simTime="${SIM_TIME}" \
    --logging=1 \
    2>&1 | tee "${RUN_DIR}/ns3.log"
) &
NS3_PID=$!

sleep 1

(
  source "${HOME}/venvs/ns3ai_env/bin/activate"
  export PYTHONPATH="${HOME}/ns-3-dev/contrib/ai/model/gym-interface/py:${PYTHONPATH:-}"
  python3 -u scratch/fmr_ai/agent.py \
    --model "${MODEL_PATH}" \
    --segment ns3ai_fmr \
    --cpp2py fmr_cpp2py \
    --py2cpp fmr_py2cpp \
    --lock fmr_lock \
    --shm 4096 \
    --is-creator 0 \
    --b2 1 \
    --deterministic 1 \
    --print-every 10 \
    --wait-seconds 120 \
    --poll-interval 0.05 \
    --turn-timeout 10 \
    2>&1 | tee "${RUN_DIR}/agent.log"
) &
AGENT_PID=$!

wait "${NS3_PID}"
wait "${AGENT_PID}"

echo "[INFO] run finished"
