This repository provides a fully reproducible simulation environment based on **ns-3.46**, extended with:

- 5G NR module (5G-LENA)
- ns3-ai (C++ ↔ Python integration)
- FMR scheduler (Fair Max Rate)
- Reinforcement Learning agent (external Python process)

The system enables **slot-level resource allocation in 5G NR downlink (OFDMA)** using a learning-based decision policy.

The integration allows:
- Real-time interaction between ns-3 and a Python agent
- Dynamic control of fairness-efficiency trade-off via parameter α
- Export of detailed per-slot logs for analysis

---

## Architecture

The system follows a **hybrid simulation architecture**:


ns-3 (C++) <--shared memory--> Python Agent (RL / heuristic)


### Flow per slot:
1. ns-3 scheduler collects UE state:
   - MCS
   - buffer size
   - UE identifiers

2. State is sent to Python via ns3-ai

3. Python agent computes:
   - RBG allocation per UE
   - α (fairness control)

4. ns-3 applies allocation and logs results

---

## Key Features

- Downlink OFDMA scheduling
- External decision agent (Python)
- Support for:
  - Fixed α
  - Trace-based α
  - RL-based control (in_action)
- Slot-level logging (CSV)
- Deterministic reproducibility

---

## Repository Structure


ns-3-dev/
├── contrib/
│ ├── nr/ # 5G NR (modified)
│ └── ai/ # ns3-ai interface
├── scratch/
│ ├── fmr.cc # main simulation
│ ├── fmr_ai/ # Python agent
│ │ ├── agent.py
│ │ └── models/ # (ignored)
├── experiments/
│ └── fmr_rl/
│ ├── run_one.sh
│ └── run_batch.sh


---

## Main Components

### 1. FMR Scheduler (C++)

contrib/nr/model/nr-mac-scheduler-ofdma-fmr.cc
contrib/nr/model/nr-mac-scheduler-ofdma-fmr.h


- Implements slot-based scheduling
- Computes efficiency + fairness score
- Supports α modes:
  - Fixed
  - Trace (CSV-driven)
- Integrates with ns3-ai

---

### 2. AI Interface

contrib/nr/model/nr-fmr-ai-msg.h
contrib/ai/


- Defines shared memory messages
- Handles synchronization between C++ and Python

---

### 3. Simulation Entry Point

scratch/fmr.cc


- Configures scenario
- Enables FMR scheduler
- Handles trace generation
- Controls simulation loop

---

### 4. Python Agent

scratch/fmr_ai/agent.py


- Receives UE state
- Outputs allocation + α
- Can be:
  - heuristic
  - RL (PPO)
  - trace-driven

---

## Requirements

### System
- Ubuntu 20.04+ (recommended)
- Python 3.8+
- GCC 9+
- CMake

### Python packages
pip install numpy pandas torch stable-baselines3

### Installation
1. Clone repository
git clone git@github.com:DiegoCanizio/ns3-fmr.git
cd ns3-fmr

2. Configure ns-3
./ns3 configure --enable-examples --enable-tests

3. Build
./ns3 build

### Running the Simulation
Terminal 1 — Python agent
cd scratch/fmr_ai
python3 agent.py

Terminal 2 — ns-3 simulation
./ns3 run scratch/fmr

### Output
Slot-level CSV (if enabled)
Fields include:
- time_s
- slot
- beam_id
- rnti
- dl_mcs
- buf_req
- target_rbg
- alloc_rbg
- alpha


### Alpha Control Modes

| Mode  | Description       |
| ----- | ----------------- |
| Fixed | constant α        |
| Trace | read from CSV     |
| RL    | computed by agent |

### Reproducibility
- Deterministic execution with fixed seeds
- Same binary + same input → identical output

### Troubleshooting
Deadlock (ns3-ai)
- Ensure agent is running before ns-3
- Check blocking calls:
  - CppSendBegin / PyRecvBegin

No output
- Verify scheduler selection
- Check logs:
  - ns3.log
  - agent.log

### Build errors
- Clean build:
  rm -rf build/
  ./ns3 configure
  ./ns3 build

### Notes
Repository includes full ns-3 + NR + ns3-ai for reproducibility
Generated files and models are excluded via .gitignore

### Future Work
Uplink support
Multi-beam scheduling
Dynamic fairness models
O-RAN integration

Author

Diego Canizio Lopes
PhD Candidate — Computer Science
UFF (Universidade Federal Fluminense)
