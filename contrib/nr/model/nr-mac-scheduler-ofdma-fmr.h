#pragma once

#include "nr-mac-scheduler-ofdma-rr.h"
#include "nr-mac-scheduler-ue-info-fmr.h"

#include "ns3/nstime.h"
#include "ns3/enum.h"

#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// ns3-ai
#include "ns3/ns3-ai-msg-interface.h"
#include "nr-fmr-ai-msg.h"

namespace ns3
{

class NrMacSchedulerOfdmaFmr : public NrMacSchedulerOfdmaRR
{
public:
    enum AlphaMode
    {
        ALPHA_FIXED = 0,
        ALPHA_TRACE = 1
    };

    enum AiProtocol
    {
        AI_SEND_THEN_RECV = 0, // C++: send obs then recv act ; Python: recv obs then send act
        AI_RECV_THEN_SEND = 1  // C++: recv act then send obs (não recomendado aqui)
    };

    static TypeId GetTypeId();

    NrMacSchedulerOfdmaFmr();
    ~NrMacSchedulerOfdmaFmr() override;

protected:
    std::shared_ptr<NrMacSchedulerUeInfo> CreateUeRepresentation(
        const NrMacCschedSapProvider::CschedUeConfigReqParameters& params) const override;

    std::function<bool(const NrMacSchedulerNs3::UePtrAndBufferReq& lhs,
                       const NrMacSchedulerNs3::UePtrAndBufferReq& rhs)>
    GetUeCompareDlFn() const override;

    std::function<bool(const NrMacSchedulerNs3::UePtrAndBufferReq& lhs,
                       const NrMacSchedulerNs3::UePtrAndBufferReq& rhs)>
    GetUeCompareUlFn() const override
    {
        return NrMacSchedulerOfdmaRR::GetUeCompareUlFn();
    }

    BeamSymbolMap AssignDLRBG(uint32_t symAvail, const ActiveUeMap& activeDl) const override;

    void DoDispose() override;

private:
    // ------------------- FMR local (fallback) -------------------
    double m_alphaFixed{0.7};
    double m_tau{1.0};

    AlphaMode m_alphaMode{ALPHA_FIXED};
    std::string m_alphaTracePath{""};
    bool m_alphaTraceLoop{true};
    double m_alphaTraceDefault{0.7};

    // ------------------- ns3-ai -------------------
    bool m_enableNs3Ai{false};
    bool m_aiHandleFinish{true};
    uint32_t m_aiShmSize{4096};

    std::string m_aiSegmentName{"ns3ai_fmr"};
    std::string m_aiCpp2PyName{"fmr_cpp2py"};
    std::string m_aiPy2CppName{"fmr_py2cpp"};
    std::string m_aiLockableName{"fmr_lock"};

    bool m_aiCppIsCreator{false};          // IMPORTANT: allow Python as creator
    AiProtocol m_aiProtocol{AI_SEND_THEN_RECV};
    bool m_aiVerbose{false};

    uint32_t m_aiMaxConsecutiveFails{200};

    using AiIface = Ns3AiMsgInterfaceImpl<FmrAiObs, FmrAiAct>;
    mutable std::unique_ptr<AiIface> m_aiInterface;
    mutable uint32_t m_aiConsecutiveFails{0};

    // ------------------- Slot CSV -------------------
    bool m_enableSlotCsv{false};
    std::string m_slotCsvPath{""};
    bool m_slotCsvAppend{false};
    bool m_slotCsvFlush{false};

    // ------------------- State -------------------
    mutable uint64_t m_slotCounter{0};

    mutable bool m_alphaTraceLoaded{false};
    mutable std::vector<double> m_alphaTrace;

    mutable std::ofstream m_slotCsv;
    mutable bool m_slotCsvHeaderWritten{false};

private:
    void EnsureAiInterface() const;
    void MaybeDropAiInterface(const std::string& reason) const;

    void ComputeDlTargetsForBeam(std::vector<UePtrAndBufferReq>& ueVector,
                                 uint32_t totalRbgThisBeam,
                                 double alpha) const;

    double GetAlphaForThisSlot() const;
    void EnsureAlphaTraceLoaded() const;

    void MaybeOpenSlotCsv() const;
    void WriteSlotCsv(const BeamId& beamId,
                      double alphaThisSlot,
                      const std::vector<UePtrAndBufferReq>& ueVectorSnapshot) const;

    bool TryApplyAiDecisionForBeam(const BeamId& beamId,
                                   std::vector<UePtrAndBufferReq>& ueVector,
                                   uint32_t totalRbgThisBeam,
                                   double& outAlphaThisSlot) const;
};

} // namespace ns3