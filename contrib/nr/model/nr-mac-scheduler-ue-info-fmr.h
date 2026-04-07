// nr-mac-scheduler-ue-info-fmr.h
//
// MINHA: UE info para o scheduler FMR (DL apenas).
// Guarda contadores por slot: targetDlRbg e allocDlRbg.

#pragma once

#include "nr-mac-scheduler-ns3.h" // para BeamId, etc.

namespace ns3
{

/**
 * @ingroup scheduler
 * @brief UE representation for FMR (DL only)
 *
 * Campos:
 * - m_targetDlRbg: alvo de RBG no beam para o slot atual (inteiro)
 * - m_allocDlRbg: quantos RBG já recebeu no slot atual
 *
 * Observação:
 * A base NrMacSchedulerUeInfo já guarda MCS, RNTI, RBGs alocados, etc.
 */
class NrMacSchedulerUeInfoFmr : public NrMacSchedulerUeInfo
{
public:
    NrMacSchedulerUeInfoFmr(uint16_t rnti, BeamId beamId, const GetRbPerRbgFn& fn)
        : NrMacSchedulerUeInfo(rnti, beamId, fn)
    {
    }

    ~NrMacSchedulerUeInfoFmr() override = default;

    // MINHA: reset do slot (DL). Útil se você quiser chamar explicitamente.
    void ResetDlSlot()
    {
        m_allocDlRbg = 0;
        m_targetDlRbg = 0;
    }

public:
    // MINHA: alvo e alocado no slot atual
    uint32_t m_targetDlRbg{0};
    uint32_t m_allocDlRbg{0};

    // MINHA (opcional): espaço para histórico futuro
    double m_lastDlAllocFrac{0.0};
    double m_tpShort{0.0};
};

} // namespace ns3
