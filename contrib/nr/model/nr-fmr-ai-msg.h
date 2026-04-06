#pragma once
#include <cstdint>

namespace ns3
{

static constexpr uint32_t FMR_AI_MAX_UES = 9;
static constexpr uint32_t FMR_AI_MAGIC  = 0xF0A11u;
static constexpr uint32_t FMR_AI_VER    = 1;

struct FmrAiObs
{
    // Header
    uint32_t magic;
    uint32_t version;

    // Context
    uint64_t slot;
    uint32_t beam_hash;
    uint32_t num_ues;
    uint32_t total_rbg;

    // Per-UE (active first, then zero-padded)
    uint16_t rnti[FMR_AI_MAX_UES];
    uint16_t dl_mcs[FMR_AI_MAX_UES];
    uint32_t buf_req[FMR_AI_MAX_UES];
};

struct FmrAiAct
{
    // Header
    uint32_t magic;
    uint32_t version;

    // Decision
    float alpha_next;
    uint32_t num_ues;

    uint16_t rnti[FMR_AI_MAX_UES];
    uint16_t alloc_rbg[FMR_AI_MAX_UES];
};

} // namespace ns3