#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${HOME}/ns3-fmr"
BIN="${BASE_DIR}/build/scratch/ns3.46-fmr-compara-qos-default"

RUN_ID="${1:-compare_qos_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${BASE_DIR}/compare_runs/${RUN_ID}"

BW_LIST="${BW_LIST:-10}"

NUM_DL_FLOWS_PER_UE="${NUM_DL_FLOWS_PER_UE:-10}"

SIM_TIME="${SIM_TIME:-5s}"
RMIN="${RMIN:-20}"
RMAX="${RMAX:-150}"

MOBILITY="${MOBILITY:-1}"
VMIN="${VMIN:-0.5}"
VMAX="${VMAX:-1.5}"
BOUNDS="${BOUNDS:-200}"
STEPDIST="${STEPDIST:-5}"

LAMBDA="${LAMBDA:-600}"
UDP_PACKET_SIZE="${UDP_PACKET_SIZE:-3000}"

ENABLE_SLOT_CSV="${ENABLE_SLOT_CSV:-1}"
UE_SNAPSHOT_PERIOD="${UE_SNAPSHOT_PERIOD:-100ms}"

MODEL_DIR="${MODEL_DIR:-${BASE_DIR}/scratch/fmr_ai/models}"

mkdir -p "${RUN_DIR}"

cleanup_ai() {
  pkill -f "scratch/fmr_ai/agent.py" || true
  rm -f /dev/shm/ns3ai_fmr* /dev/shm/fmr_cpp2py* /dev/shm/fmr_py2cpp* /dev/shm/fmr_lock* || true
}

bw_to_hz() {
  local BW_MHZ="$1"
  echo "$((BW_MHZ * 1000000))"
}

tau_for_bw() {
  local BW_MHZ="$1"
  case "${BW_MHZ}" in
    10)  echo "${TAU_10:-0.65}" ;;
    20)  echo "${TAU_20:-0.65}" ;;
    50)  echo "${TAU_50:-0.65}" ;;
    100) echo "${TAU_100:-0.65}" ;;
    *)   echo "${TAU:-0.65}" ;;
  esac
}

run_mode() {
  local BW_MHZ="$1"
  local BW_HZ="$2"
  local MODE="$3"
  local TAU_BW="$4"

  if [[ "${MODE}" == "fmr_rl" ]]; then
    echo "[ERROR] fmr_rl deve ser executado via run_fmr_rl, não via run_mode."
    exit 1
  fi

  local MODE_DIR="${RUN_DIR}/bw${BW_MHZ}/${MODE}"
  mkdir -p "${MODE_DIR}"

  echo
  echo "=================================================="
  echo "[RUN] BW=${BW_MHZ}MHz | MODE=${MODE}"
  echo "=================================================="

  cd "${BASE_DIR}"

  "${BIN}" \
    --schedulerMode="${MODE}" \
    --EnableNs3Ai=0 \
    --EnableSlotCsv="${ENABLE_SLOT_CSV}" \
    --SlotCsvPath="${MODE_DIR}/slot_log_${MODE}.csv" \
    --SlotCsvAppend=0 \
    --SlotCsvFlush=1 \
    --FmrTau="${TAU_BW}" \
    --simTime="${SIM_TIME}" \
    --bandwidth="${BW_HZ}" \
    --scenarioRadiusMin="${RMIN}" \
    --scenarioRadiusMax="${RMAX}" \
    --enableMobility="${MOBILITY}" \
    --mobilitySpeedMin="${VMIN}" \
    --mobilitySpeedMax="${VMAX}" \
    --mobilityBounds="${BOUNDS}" \
    --mobilityDistance="${STEPDIST}" \
    --lambda="${LAMBDA}" \
    --udpPacketSize="${UDP_PACKET_SIZE}" \
    --numDlFlowsPerUe="${NUM_DL_FLOWS_PER_UE}" \
    --EnableUeSnapshotCsv=1 \
    --UeSnapshotCsvPath="${MODE_DIR}/ue_snapshot_${MODE}.csv" \
    --UeSnapshotPeriod="${UE_SNAPSHOT_PERIOD}" \
    --EnableFlowSummaryCsv=1 \
    --FlowSummaryCsvPath="${MODE_DIR}/flow_summary_${MODE}.csv" \
    --simTag="summary.txt" \
    --outputDir="${MODE_DIR}" \
    --logging=1 \
    2>&1 | tee "${MODE_DIR}/ns3.log"
}

run_fmr_rl() {
  local BW_MHZ="$1"
  local BW_HZ="$2"
  local TAU_BW="$3"

  local MODE="fmr_rl"
  local MODE_DIR="${RUN_DIR}/bw${BW_MHZ}/${MODE}"
  mkdir -p "${MODE_DIR}"

  local MODEL_PATH="${MODEL_DIR}/model_${BW_MHZ}.zip"

  if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "[ERROR] Modelo não encontrado: ${MODEL_PATH}"
    exit 1
  fi

  local SEGMENT="ns3ai_fmr_bw${BW_MHZ}"
  local CPP2PY="fmr_cpp2py_bw${BW_MHZ}"
  local PY2CPP="fmr_py2cpp_bw${BW_MHZ}"
  local LOCK="fmr_lock_bw${BW_MHZ}"

  cleanup_ai

  echo
  echo "=================================================="
  echo "[AGENT] BW=${BW_MHZ}MHz"
  echo "[MODEL] ${MODEL_PATH}"
  echo "[TAU] ${TAU_BW}"
  echo "=================================================="

  cd "${BASE_DIR}"
  source "${BASE_DIR}/.venv/bin/activate"
  export PYTHONPATH="${BASE_DIR}/contrib/ai/model/gym-interface/py:${PYTHONPATH:-}"

  python3 -u scratch/fmr_ai/agent.py \
    --model "${MODEL_PATH}" \
    --segment "${SEGMENT}" \
    --cpp2py "${CPP2PY}" \
    --py2cpp "${PY2CPP}" \
    --lock "${LOCK}" \
    --shm 4096 \
    --is-creator 0 \
    --b2 1 \
    --deterministic 1 \
    --print-every 50 \
    --wait-seconds 120 \
    2>&1 | tee "${MODE_DIR}/agent_${MODE}.log" &

  local AGENT_PID=$!
  echo "[AGENT] PID=${AGENT_PID}"

  sleep 2

  echo
  echo "=================================================="
  echo "[RUN] BW=${BW_MHZ}MHz | MODE=fmr_rl"
  echo "=================================================="

  "${BIN}" \
    --schedulerMode="fmr_rl" \
    --EnableNs3Ai=1 \
    --AiCppIsCreator=1 \
    --AiVerbose=1 \
    --AiShmSize=4096 \
    --AiSegmentName="${SEGMENT}" \
    --AiCpp2PyName="${CPP2PY}" \
    --AiPy2CppName="${PY2CPP}" \
    --AiLockableName="${LOCK}" \
    --EnableSlotCsv="${ENABLE_SLOT_CSV}" \
    --SlotCsvPath="${MODE_DIR}/slot_log_${MODE}.csv" \
    --SlotCsvFlush=1 \
    --FmrTau="${TAU_BW}" \
    --simTime="${SIM_TIME}" \
    --bandwidth="${BW_HZ}" \
    --scenarioRadiusMin="${RMIN}" \
    --scenarioRadiusMax="${RMAX}" \
    --enableMobility="${MOBILITY}" \
    --mobilitySpeedMin="${VMIN}" \
    --mobilitySpeedMax="${VMAX}" \
    --mobilityBounds="${BOUNDS}" \
    --mobilityDistance="${STEPDIST}" \
    --lambda="${LAMBDA}" \
    --udpPacketSize="${UDP_PACKET_SIZE}" \
    --numDlFlowsPerUe="${NUM_DL_FLOWS_PER_UE}" \
    --EnableUeSnapshotCsv=1 \
    --UeSnapshotCsvPath="${MODE_DIR}/ue_snapshot_${MODE}.csv" \
    --UeSnapshotPeriod="${UE_SNAPSHOT_PERIOD}" \
    --EnableFlowSummaryCsv=1 \
    --FlowSummaryCsvPath="${MODE_DIR}/flow_summary_${MODE}.csv" \
    --simTag="summary.txt" \
    --outputDir="${MODE_DIR}" \
    --logging=1 \
    2>&1 | tee "${MODE_DIR}/ns3.log"

  echo "[AGENT] Stopping PID=${AGENT_PID}"
  kill "${AGENT_PID}" 2>/dev/null || true
  wait "${AGENT_PID}" 2>/dev/null || true
  cleanup_ai
}

main() {
  if [[ ! -x "${BIN}" ]]; then
    echo "[ERROR] Binário não encontrado ou não executável:"
    echo "        ${BIN}"
    echo "Rode: ./ns3 build"
    exit 1
  fi

  cleanup_ai

  echo "[INFO] RUN_ID=${RUN_ID}"
  echo "[INFO] RUN_DIR=${RUN_DIR}"
  echo "[INFO] BW_LIST=${BW_LIST}"
  echo "[INFO] NUM_DL_FLOWS_PER_UE=${NUM_DL_FLOWS_PER_UE}"
  echo "[INFO] SIM_TIME=${SIM_TIME}"
  echo "[INFO] LAMBDA=${LAMBDA}"
  echo "[INFO] UDP_PACKET_SIZE=${UDP_PACKET_SIZE}"

  for BW_MHZ in ${BW_LIST}; do
    BW_HZ="$(bw_to_hz "${BW_MHZ}")"
    TAU_BW="$(tau_for_bw "${BW_MHZ}")"

    echo
    echo "##################################################"
    echo "[BANDWIDTH] ${BW_MHZ} MHz | ${BW_HZ} Hz | TAU=${TAU_BW}"
    echo "##################################################"

    run_mode "${BW_MHZ}" "${BW_HZ}" rr "${TAU_BW}"
    run_mode "${BW_MHZ}" "${BW_HZ}" pf "${TAU_BW}"
    run_mode "${BW_MHZ}" "${BW_HZ}" mr "${TAU_BW}"
    run_fmr_rl "${BW_MHZ}" "${BW_HZ}" "${TAU_BW}"
  done

  echo
  echo "[DONE] Compare finished."
  echo "[DONE] Results saved in: ${RUN_DIR}"
  echo
  find "${RUN_DIR}" -maxdepth 3 -type f | sort
}

main "$@"
