#include "nr-mac-scheduler-ofdma-fmr.h"
#include "nr-mac-scheduler-ue-info-fmr.h"

#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/enum.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("NrMacSchedulerOfdmaFmr");
NS_OBJECT_ENSURE_REGISTERED(NrMacSchedulerOfdmaFmr);

namespace
{

static inline NrMacSchedulerUeInfoFmr*
AsFmrUeInfoPtr(const std::shared_ptr<NrMacSchedulerUeInfo>& p)
{
    return dynamic_cast<NrMacSchedulerUeInfoFmr*>(p.get());
}

static inline double
Clamp(double v, double lo, double hi)
{
    return std::max(lo, std::min(v, hi));
}

static std::vector<double>
Softmax(const std::vector<double>& x, double tau)
{
    std::vector<double> y(x.size(), 0.0);
    if (x.empty())
    {
        return y;
    }

    const double t = std::max(tau, 1e-12);
    const double invT = 1.0 / t;

    const double maxv = *std::max_element(x.begin(), x.end());
    double sum = 0.0;

    for (size_t i = 0; i < x.size(); ++i)
    {
        y[i] = std::exp((x[i] - maxv) * invT);
        sum += y[i];
    }

    if (sum <= 0.0)
    {
        const double u = 1.0 / static_cast<double>(x.size());
        return std::vector<double>(x.size(), u);
    }

    for (auto& v : y)
    {
        v /= sum;
    }
    return y;
}

static std::vector<uint32_t>
LargestRemainder(const std::vector<double>& shares, uint32_t total)
{
    std::vector<uint32_t> alloc(shares.size(), 0);
    if (shares.empty() || total == 0)
    {
        return alloc;
    }

    std::vector<double> frac(shares.size(), 0.0);
    uint32_t used = 0;

    for (size_t i = 0; i < shares.size(); ++i)
    {
        const double exact = shares[i] * static_cast<double>(total);
        const uint32_t base = static_cast<uint32_t>(std::floor(exact));
        alloc[i] = base;
        used += base;
        frac[i] = exact - static_cast<double>(base);
    }

    const uint32_t remaining = (used <= total) ? (total - used) : 0;

    std::vector<size_t> idx(shares.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return frac[a] > frac[b]; });

    for (uint32_t k = 0; k < remaining; ++k)
    {
        alloc[idx[k % idx.size()]] += 1;
    }
    return alloc;
}

static bool
ParseAlphaLine(const std::string& line, double& outAlpha)
{
    std::string s = line;
    auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };

    while (!s.empty() && isSpace(static_cast<unsigned char>(s.front())))
    {
        s.erase(s.begin());
    }
    while (!s.empty() && isSpace(static_cast<unsigned char>(s.back())))
    {
        s.pop_back();
    }
    if (s.empty() || s[0] == '#')
    {
        return false;
    }

    std::stringstream ss(s);
    double a = 0.0;
    ss >> a;
    if (ss.fail())
    {
        return false;
    }

    outAlpha = Clamp(a, 0.0, 1.0);
    return true;
}

static inline uint32_t
BeamHash32(const BeamId& b)
{
    std::ostringstream oss;
    oss << b;
    const std::string s = oss.str();
    uint32_t h = 2166136261u;
    for (unsigned char c : s)
    {
        h ^= static_cast<uint32_t>(c);
        h *= 16777619u;
    }
    return h;
}

static inline std::string
SanitizeAiName(std::string s)
{
    auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!s.empty() && isSpace(static_cast<unsigned char>(s.front())))
    {
        s.erase(s.begin());
    }
    while (!s.empty() && isSpace(static_cast<unsigned char>(s.back())))
    {
        s.pop_back();
    }
    while (!s.empty() && s.front() == '/')
    {
        s.erase(s.begin());
    }
    for (auto& ch : s)
    {
        if (std::isspace(static_cast<unsigned char>(ch)))
        {
            ch = '_';
        }
    }
    if (s.empty())
    {
        s = "ns3ai_fmr";
    }
    return s;
}

} // namespace

TypeId
NrMacSchedulerOfdmaFmr::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::NrMacSchedulerOfdmaFmr")
            .SetParent<NrMacSchedulerOfdmaRR>()
            .SetGroupName("nr")
            .AddConstructor<NrMacSchedulerOfdmaFmr>()

            .AddAttribute("AlphaFixed",
                          "Fixed alpha used when AlphaMode=Fixed.",
                          DoubleValue(0.7),
                          MakeDoubleAccessor(&NrMacSchedulerOfdmaFmr::m_alphaFixed),
                          MakeDoubleChecker<double>(0.0, 1.0))
            .AddAttribute("Tau",
                          "Softmax temperature used in local fallback.",
                          DoubleValue(1.0),
                          MakeDoubleAccessor(&NrMacSchedulerOfdmaFmr::m_tau),
                          MakeDoubleChecker<double>(1e-12, 1e9))

            .AddAttribute("AlphaMode",
                          "Alpha mode: Fixed or Trace.",
                          EnumValue(NrMacSchedulerOfdmaFmr::ALPHA_FIXED),
                          MakeEnumAccessor<NrMacSchedulerOfdmaFmr::AlphaMode>(&NrMacSchedulerOfdmaFmr::m_alphaMode),
                          MakeEnumChecker(NrMacSchedulerOfdmaFmr::ALPHA_FIXED, "Fixed",
                                          NrMacSchedulerOfdmaFmr::ALPHA_TRACE, "Trace"))
            .AddAttribute("AlphaTracePath",
                          "Path to alpha trace file (one alpha per line).",
                          StringValue(""),
                          MakeStringAccessor(&NrMacSchedulerOfdmaFmr::m_alphaTracePath),
                          MakeStringChecker())
            .AddAttribute("AlphaTraceLoop",
                          "Loop alpha trace when reaches EOF.",
                          BooleanValue(true),
                          MakeBooleanAccessor(&NrMacSchedulerOfdmaFmr::m_alphaTraceLoop),
                          MakeBooleanChecker())
            .AddAttribute("AlphaTraceDefault",
                          "Default alpha if trace file is empty/unreadable.",
                          DoubleValue(0.7),
                          MakeDoubleAccessor(&NrMacSchedulerOfdmaFmr::m_alphaTraceDefault),
                          MakeDoubleChecker<double>(0.0, 1.0))

            // ns3-ai
            .AddAttribute("EnableNs3Ai",
                          "Enable ns3-ai message interface to get per-beam decisions from Python.",
                          BooleanValue(false),
                          MakeBooleanAccessor(&NrMacSchedulerOfdmaFmr::m_enableNs3Ai),
                          MakeBooleanChecker())
            .AddAttribute("AiHandleFinish",
                          "If true, C++ notifies finish when disposed.",
                          BooleanValue(true),
                          MakeBooleanAccessor(&NrMacSchedulerOfdmaFmr::m_aiHandleFinish),
                          MakeBooleanChecker())
            .AddAttribute("AiShmSize",
                          "Shared memory segment size (bytes).",
                          UintegerValue(4096),
                          MakeUintegerAccessor(&NrMacSchedulerOfdmaFmr::m_aiShmSize),
                          MakeUintegerChecker<uint32_t>(1024, 1u << 26))
            .AddAttribute("AiSegmentName",
                          "Shared memory segment name.",
                          StringValue("ns3ai_fmr"),
                          MakeStringAccessor(&NrMacSchedulerOfdmaFmr::m_aiSegmentName),
                          MakeStringChecker())
            .AddAttribute("AiCpp2PyName",
                          "Named object for Cpp->Py message.",
                          StringValue("fmr_cpp2py"),
                          MakeStringAccessor(&NrMacSchedulerOfdmaFmr::m_aiCpp2PyName),
                          MakeStringChecker())
            .AddAttribute("AiPy2CppName",
                          "Named object for Py->Cpp message.",
                          StringValue("fmr_py2cpp"),
                          MakeStringAccessor(&NrMacSchedulerOfdmaFmr::m_aiPy2CppName),
                          MakeStringChecker())
            .AddAttribute("AiLockableName",
                          "Named object for the lock/semaphore structure.",
                          StringValue("fmr_lock"),
                          MakeStringAccessor(&NrMacSchedulerOfdmaFmr::m_aiLockableName),
                          MakeStringChecker())
            .AddAttribute("AiCppIsCreator",
                          "If true, C++ creates shm; if false, Python creates shm.",
                          BooleanValue(false),
                          MakeBooleanAccessor(&NrMacSchedulerOfdmaFmr::m_aiCppIsCreator),
                          MakeBooleanChecker())
            .AddAttribute("AiProtocol",
                          "0=SendThenRecv (recommended), 1=RecvThenSend.",
                          EnumValue(NrMacSchedulerOfdmaFmr::AI_SEND_THEN_RECV),
                          MakeEnumAccessor<NrMacSchedulerOfdmaFmr::AiProtocol>(&NrMacSchedulerOfdmaFmr::m_aiProtocol),
                          MakeEnumChecker(NrMacSchedulerOfdmaFmr::AI_SEND_THEN_RECV, "SendThenRecv",
                                          NrMacSchedulerOfdmaFmr::AI_RECV_THEN_SEND, "RecvThenSend"))
            .AddAttribute("AiVerbose",
                          "Extra ns3-ai logs.",
                          BooleanValue(false),
                          MakeBooleanAccessor(&NrMacSchedulerOfdmaFmr::m_aiVerbose),
                          MakeBooleanChecker())
            .AddAttribute("AiMaxConsecutiveFails",
                          "Drop interface after this many consecutive failures (but keep EnableNs3Ai=true).",
                          UintegerValue(200),
                          MakeUintegerAccessor(&NrMacSchedulerOfdmaFmr::m_aiMaxConsecutiveFails),
                          MakeUintegerChecker<uint32_t>(1, 100000))

            // Slot CSV
            .AddAttribute("EnableSlotCsv",
                          "Enable per-slot CSV logging.",
                          BooleanValue(false),
                          MakeBooleanAccessor(&NrMacSchedulerOfdmaFmr::m_enableSlotCsv),
                          MakeBooleanChecker())
            .AddAttribute("SlotCsvPath",
                          "CSV path for per-slot scheduler log.",
                          StringValue(""),
                          MakeStringAccessor(&NrMacSchedulerOfdmaFmr::m_slotCsvPath),
                          MakeStringChecker())
            .AddAttribute("SlotCsvAppend",
                          "Append scheduler CSV if true; otherwise truncate.",
                          BooleanValue(false),
                          MakeBooleanAccessor(&NrMacSchedulerOfdmaFmr::m_slotCsvAppend),
                          MakeBooleanChecker())
            .AddAttribute("SlotCsvFlush",
                          "Flush CSV at every write.",
                          BooleanValue(false),
                          MakeBooleanAccessor(&NrMacSchedulerOfdmaFmr::m_slotCsvFlush),
                          MakeBooleanChecker());

    return tid;
}

NrMacSchedulerOfdmaFmr::NrMacSchedulerOfdmaFmr() = default;
NrMacSchedulerOfdmaFmr::~NrMacSchedulerOfdmaFmr() = default;

void
NrMacSchedulerOfdmaFmr::DoDispose()
{
    if (m_aiInterface)
    {
        if (m_aiHandleFinish)
        {
            try
            {
                m_aiInterface->CppSetFinished();
            }
            catch (const std::exception& e)
            {
                NS_LOG_WARN("[AI] CppSetFinished exception in DoDispose: " << e.what());
            }
            catch (...)
            {
                NS_LOG_WARN("[AI] CppSetFinished unknown exception in DoDispose");
            }
        }

        m_aiInterface.reset();
    }

    if (m_slotCsv.is_open())
    {
        m_slotCsv.close();
    }

    NrMacSchedulerOfdmaRR::DoDispose();
}

std::shared_ptr<NrMacSchedulerUeInfo>
NrMacSchedulerOfdmaFmr::CreateUeRepresentation(
    const NrMacCschedSapProvider::CschedUeConfigReqParameters& params) const
{
    return std::make_shared<NrMacSchedulerUeInfoFmr>(
        params.m_rnti,
        params.m_beamId,
        std::bind(&NrMacSchedulerOfdmaFmr::GetNumRbPerRbg, this));
}

std::function<bool(const NrMacSchedulerNs3::UePtrAndBufferReq& lhs,
                   const NrMacSchedulerNs3::UePtrAndBufferReq& rhs)>
NrMacSchedulerOfdmaFmr::GetUeCompareDlFn() const
{
    return NrMacSchedulerOfdmaRR::GetUeCompareDlFn();
}

void
NrMacSchedulerOfdmaFmr::EnsureAlphaTraceLoaded() const
{
    if (m_alphaTraceLoaded)
    {
        return;
    }
    m_alphaTraceLoaded = true;
    m_alphaTrace.clear();

    if (m_alphaTracePath.empty())
    {
        return;
    }

    std::ifstream in(m_alphaTracePath.c_str());
    if (!in.is_open())
    {
        NS_LOG_WARN("AlphaTracePath cannot be opened: " << m_alphaTracePath);
        return;
    }

    std::string line;
    while (std::getline(in, line))
    {
        double a = 0.0;
        if (ParseAlphaLine(line, a))
        {
            m_alphaTrace.push_back(a);
        }
    }
    in.close();

    if (m_alphaTrace.empty())
    {
        NS_LOG_WARN("Alpha trace is empty after parsing: " << m_alphaTracePath);
    }
}

double
NrMacSchedulerOfdmaFmr::GetAlphaForThisSlot() const
{
    if (m_alphaMode == ALPHA_FIXED)
    {
        return Clamp(m_alphaFixed, 0.0, 1.0);
    }

    EnsureAlphaTraceLoaded();

    if (m_alphaTrace.empty())
    {
        return Clamp(m_alphaTraceDefault, 0.0, 1.0);
    }

    const uint64_t k = m_slotCounter;
    if (k < m_alphaTrace.size())
    {
        return Clamp(m_alphaTrace[static_cast<size_t>(k)], 0.0, 1.0);
    }

    if (m_alphaTraceLoop)
    {
        const size_t idx = static_cast<size_t>(k % m_alphaTrace.size());
        return Clamp(m_alphaTrace[idx], 0.0, 1.0);
    }

    return Clamp(m_alphaTrace.back(), 0.0, 1.0);
}

void
NrMacSchedulerOfdmaFmr::ComputeDlTargetsForBeam(std::vector<UePtrAndBufferReq>& ueVector,
                                                uint32_t totalRbgThisBeam,
                                                double alpha) const
{
    std::vector<double> scores;
    scores.reserve(ueVector.size());

    for (auto& u : ueVector)
    {
        const auto ueInfo = u.first;
        const uint8_t mcs = ueInfo->m_dlMcs;
        const double eff = static_cast<double>(mcs);
        const double fair = 1.0;
        scores.push_back((1.0 - alpha) * eff + alpha * fair);
    }

    const auto shares = Softmax(scores, m_tau);
    const auto targets = LargestRemainder(shares, totalRbgThisBeam);

    for (size_t i = 0; i < ueVector.size(); ++i)
    {
        if (auto fmr = AsFmrUeInfoPtr(ueVector[i].first))
        {
            fmr->m_targetDlRbg = targets[i];
        }
    }
}

void
NrMacSchedulerOfdmaFmr::MaybeOpenSlotCsv() const
{
    if (!m_enableSlotCsv || m_slotCsv.is_open() || m_slotCsvPath.empty())
    {
        return;
    }

    std::ios_base::openmode mode = std::ofstream::out;
    mode |= (m_slotCsvAppend ? std::ofstream::app : std::ofstream::trunc);
    m_slotCsv.open(m_slotCsvPath.c_str(), mode);

    if (!m_slotCsv.is_open())
    {
        NS_LOG_WARN("Could not open SlotCsvPath: " << m_slotCsvPath);
        return;
    }

    if (!m_slotCsvAppend)
    {
        m_slotCsvHeaderWritten = false;
    }
}

void
NrMacSchedulerOfdmaFmr::WriteSlotCsv(const BeamId& beamId,
                                     double alphaThisSlot,
                                     const std::vector<UePtrAndBufferReq>& ueVectorSnapshot) const
{
    if (!m_enableSlotCsv || m_slotCsvPath.empty())
    {
        return;
    }

    MaybeOpenSlotCsv();
    if (!m_slotCsv.is_open())
    {
        return;
    }

    if (!m_slotCsvHeaderWritten)
    {
        m_slotCsv << "time_s,slot,beam_id,rnti,dl_mcs,buf_req,target_rbg,alloc_rbg,alpha\n";
        m_slotCsvHeaderWritten = true;
    }

    const double t = Simulator::Now().GetSeconds();

    for (const auto& u : ueVectorSnapshot)
    {
        const auto ueInfo = u.first;

        uint32_t target = 0;
        uint32_t alloc = 0;
        if (auto fmr = AsFmrUeInfoPtr(u.first))
        {
            target = fmr->m_targetDlRbg;
            alloc = fmr->m_allocDlRbg;
        }

        m_slotCsv << std::fixed << std::setprecision(6) << t << ","
                  << m_slotCounter << ","
                  << beamId << ","
                  << ueInfo->m_rnti << ","
                  << static_cast<uint32_t>(ueInfo->m_dlMcs) << ","
                  << u.second << ","
                  << target << ","
                  << alloc << ","
                  << std::setprecision(6) << alphaThisSlot << "\n";
    }

    if (m_slotCsvFlush)
    {
        m_slotCsv.flush();
    }
}

void
NrMacSchedulerOfdmaFmr::MaybeDropAiInterface(const std::string& reason) const
{
    m_aiConsecutiveFails++;
    if (m_aiVerbose || (m_aiConsecutiveFails % 50 == 1))
    {
        NS_LOG_WARN("[AI] fail=" << m_aiConsecutiveFails << "/" << m_aiMaxConsecutiveFails
                                 << " reason=" << reason);
    }
    if (m_aiConsecutiveFails >= m_aiMaxConsecutiveFails)
    {
        NS_LOG_WARN("[AI] dropping interface after too many fails (but keeping EnableNs3Ai=true)");
        m_aiInterface.reset();
        m_aiConsecutiveFails = 0;
    }
}

void
NrMacSchedulerOfdmaFmr::EnsureAiInterface() const
{
    if (!m_enableNs3Ai || m_aiInterface)
    {
        return;
    }

    const std::string seg = SanitizeAiName(m_aiSegmentName);
    const std::string c2p = SanitizeAiName(m_aiCpp2PyName);
    const std::string p2c = SanitizeAiName(m_aiPy2CppName);
    const std::string lk  = SanitizeAiName(m_aiLockableName);

    try
    {
        // IMPORTANT: do NOT disable ns3-ai if Python is not up yet.
        // Just keep interface null and retry next slot.
        m_aiInterface = std::make_unique<AiIface>(m_aiCppIsCreator,
                                          false, // use_vector
                                          m_aiHandleFinish,
                                          static_cast<uint32_t>(m_aiShmSize),
                                          seg.c_str(),
                                          c2p.c_str(),
                                          p2c.c_str(),
                                          lk.c_str());

        m_aiConsecutiveFails = 0;

        NS_LOG_WARN("[AI] interface created: creator=" << (m_aiCppIsCreator ? 1 : 0)
                                                       << " protocol="
                                                       << ((m_aiProtocol == AI_SEND_THEN_RECV) ? "SendThenRecv" : "RecvThenSend")
                                                       << " segment=" << seg
                                                       << " cpp2py=" << c2p
                                                       << " py2cpp=" << p2c
                                                       << " lock=" << lk
                                                       << " shm=" << m_aiShmSize);
    }
    catch (const std::exception& e)
    {
        // no DisableAi here
        MaybeDropAiInterface(std::string("create/open exception: ") + e.what());
        m_aiInterface.reset();
    }
    catch (...)
    {
        MaybeDropAiInterface("create/open unknown exception");
        m_aiInterface.reset();
    }
}

bool
NrMacSchedulerOfdmaFmr::TryApplyAiDecisionForBeam(const BeamId& beamId,
                                                  std::vector<UePtrAndBufferReq>& ueVector,
                                                  uint32_t totalRbgThisBeam,
                                                  double& outAlphaThisSlot) const
{
    if (!m_enableNs3Ai || ueVector.empty() || totalRbgThisBeam == 0)
    {
        return false;
    }

    EnsureAiInterface();
    if (!m_aiInterface)
    {
        return false;
    }

    auto* obs = m_aiInterface->GetCpp2PyStruct();
    auto* act = m_aiInterface->GetPy2CppStruct();
    if (!obs || !act)
    {
        MaybeDropAiInterface("null obs/act pointers");
        m_aiInterface.reset();
        return false;
    }

    const uint32_t n = std::min<uint32_t>(static_cast<uint32_t>(ueVector.size()), FMR_AI_MAX_UES);

    try
    {
        if (m_aiProtocol == AI_SEND_THEN_RECV)
        {
            if (m_aiVerbose)
            {
                NS_LOG_WARN("[AI] slot=" << m_slotCounter << " beam=" << BeamHash32(beamId) << " CppSendBegin()");
            }
            m_aiInterface->CppSendBegin();

            // Fill obs INSIDE the send critical section
            obs->magic = FMR_AI_MAGIC;
            obs->version = FMR_AI_VER;

            obs->slot = m_slotCounter;
            obs->beam_hash = BeamHash32(beamId);
            obs->num_ues = n;
            obs->total_rbg = totalRbgThisBeam;

            for (uint32_t i = 0; i < n; ++i)
            {
                const auto ue = ueVector[i].first;
                obs->rnti[i] = ue->m_rnti;
                obs->dl_mcs[i] = static_cast<uint16_t>(ue->m_dlMcs);
                obs->buf_req[i] = static_cast<uint32_t>(ueVector[i].second);
            }
            for (uint32_t i = n; i < FMR_AI_MAX_UES; ++i)
            {
                obs->rnti[i] = 0;
                obs->dl_mcs[i] = 0;
                obs->buf_req[i] = 0;
            }

            m_aiInterface->CppSendEnd();

            if (m_aiVerbose)
            {
                NS_LOG_WARN("[AI] slot=" << m_slotCounter << " beam=" << obs->beam_hash << " CppRecvBegin()");
            }
            m_aiInterface->CppRecvBegin();
            m_aiInterface->CppRecvEnd();
        }
        else
        {
            // not recommended, but kept for completeness
            m_aiInterface->CppRecvBegin();
            m_aiInterface->CppRecvEnd();
            m_aiInterface->CppSendBegin();
            m_aiInterface->CppSendEnd();
        }

        // Validate act header
        if (act->magic != FMR_AI_MAGIC || act->version != FMR_AI_VER)
        {
            MaybeDropAiInterface("act magic/version mismatch");
            return false;
        }

        m_aiConsecutiveFails = 0;
    }
    catch (const std::exception& e)
    {
        MaybeDropAiInterface(std::string("exchange exception: ") + e.what());
        m_aiInterface.reset();
        return false;
    }
    catch (...)
    {
        MaybeDropAiInterface("exchange unknown exception");
        m_aiInterface.reset();
        return false;
    }

    // Apply act
    const uint32_t actN = std::min<uint32_t>(act->num_ues, n);
    const double alphaNext = Clamp(static_cast<double>(act->alpha_next), 0.0, 1.0);

    std::unordered_map<uint16_t, uint32_t> allocByRnti;
    allocByRnti.reserve(actN);
    for (uint32_t i = 0; i < actN && i < FMR_AI_MAX_UES; ++i)
    {
        allocByRnti[act->rnti[i]] = static_cast<uint32_t>(act->alloc_rbg[i]);
    }

    std::vector<uint32_t> alloc(n, 0);
    uint32_t sum = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        const uint16_t rnti = ueVector[i].first->m_rnti;
        auto it = allocByRnti.find(rnti);
        uint32_t a = (it != allocByRnti.end()) ? it->second : 0;
        if (a > totalRbgThisBeam)
        {
            a = totalRbgThisBeam;
        }
        alloc[i] = a;
        sum += a;
    }

    // Fix sum to exactly totalRbgThisBeam
    if (sum > totalRbgThisBeam)
    {
        uint32_t excess = sum - totalRbgThisBeam;
        for (uint32_t i = 0; i < n && excess > 0; ++i)
        {
            const uint32_t dec = std::min<uint32_t>(alloc[i], excess);
            alloc[i] -= dec;
            excess -= dec;
        }
    }
    else if (sum < totalRbgThisBeam)
    {
        uint32_t left = totalRbgThisBeam - sum;
        for (uint32_t i = 0; i < n && left > 0; ++i)
        {
            alloc[i] += 1;
            left -= 1;
        }
    }

    for (uint32_t i = 0; i < n; ++i)
    {
        if (auto fmr = AsFmrUeInfoPtr(ueVector[i].first))
        {
            fmr->m_targetDlRbg = alloc[i];
        }
    }

    outAlphaThisSlot = alphaNext;
    return true;
}

NrMacSchedulerOfdma::BeamSymbolMap
NrMacSchedulerOfdmaFmr::AssignDLRBG(uint32_t symAvail, const ActiveUeMap& activeDl) const
{
    NS_LOG_FUNCTION(this);

    GetFirst GetBeamId;
    GetSecond GetUeVector;

    BeamSymbolMap symPerBeam = GetSymPerBeam(symAvail, activeDl);
    const double alphaBaseThisSlot = GetAlphaForThisSlot();

    for (const auto& el : activeDl)
    {
        const BeamId beamId = GetBeamId(el);
        const uint32_t beamSym = symPerBeam.at(beamId);

        std::vector<UePtrAndBufferReq> ueVector;
        FTResources assignedResources(0, 0);

        std::vector<bool> availableRbgs = GetDlBitmask();
        std::set<uint32_t> remainingRbgSet;
        for (size_t i = 0; i < availableRbgs.size(); ++i)
        {
            if (availableRbgs.at(i))
            {
                remainingRbgSet.emplace(static_cast<uint32_t>(i));
            }
        }
        NS_ASSERT(!remainingRbgSet.empty());

        for (const auto& ue : GetUeVector(el))
        {
            ueVector.emplace_back(ue);
            BeforeDlSched(ueVector.back(), FTResources(beamSym, beamSym));
        }

        const uint32_t totalRbgThisBeam = static_cast<uint32_t>(remainingRbgSet.size());
        double alphaThisBeam = alphaBaseThisSlot;

        bool aiApplied = false;
        if (m_enableNs3Ai)
        {
            aiApplied = TryApplyAiDecisionForBeam(beamId, ueVector, totalRbgThisBeam, alphaThisBeam);
        }

        if (!aiApplied)
        {
            ComputeDlTargetsForBeam(ueVector, totalRbgThisBeam, alphaThisBeam);
        }

        for (auto& u : ueVector)
        {
            if (auto* f = AsFmrUeInfoPtr(u.first))
            {
                f->m_allocDlRbg = 0;
            }
        }

        bool reapingResources = true;
        while (reapingResources)
        {
            while (!remainingRbgSet.empty())
            {
                const auto prevRemaining = remainingRbgSet.size();

                SortUeVector(&ueVector, std::bind(&NrMacSchedulerOfdmaFmr::GetUeCompareDlFn, this));

                auto schedInfoIt = ueVector.begin();

                while (AdvanceToNextUeToSchedule(schedInfoIt, ueVector.end(), beamSym))
                {
                    if (!AttemptAllocationOfCurrentResourceToUe(schedInfoIt,
                                                               remainingRbgSet,
                                                               beamSym,
                                                               assignedResources,
                                                               availableRbgs))
                    {
                        std::advance(schedInfoIt, 1);
                        continue;
                    }

                    if (auto* u = AsFmrUeInfoPtr(schedInfoIt->first))
                    {
                        u->m_allocDlRbg += 1;
                    }

                    GetFirst GetUe;
                    for (auto& ue : ueVector)
                    {
                        if (GetUe(ue)->m_rnti != GetUe(*schedInfoIt)->m_rnti)
                        {
                            NotAssignedDlResources(ue, FTResources(beamSym, beamSym), assignedResources);
                        }
                    }
                    break;
                }

                if (prevRemaining == remainingRbgSet.size())
                {
                    break;
                }
            }

            std::sort(ueVector.begin(), ueVector.end(), [](auto a, auto b) {
                GetFirst GetUe;
                return GetUe(a)->m_dlTbSize > GetUe(b)->m_dlTbSize;
            });

            if (!ueVector.empty() && ueVector.back().first->m_dlTbSize < 10)
            {
                auto& ue = ueVector.back();
                while (!ue.first->m_dlRBG.empty())
                {
                    auto reapedRbg = ue.first->m_dlRBG.back();
                    DeallocateCurrentResourceFromUe(ue.first,
                                                    reapedRbg,
                                                    beamSym,
                                                    assignedResources,
                                                    availableRbgs);
                    remainingRbgSet.emplace(reapedRbg);

                    if (auto* u = AsFmrUeInfoPtr(ue.first))
                    {
                        if (u->m_allocDlRbg > 0)
                        {
                            u->m_allocDlRbg -= 1;
                        }
                    }
                }

                AssignedDlResources(ue, FTResources(beamSym, beamSym), assignedResources);

                for (auto& uev : ueVector)
                {
                    NotAssignedDlResources(uev, FTResources(beamSym, beamSym), assignedResources);
                }

                ueVector.pop_back();
                continue;
            }

            reapingResources = false;
        }

        WriteSlotCsv(beamId, alphaThisBeam, ueVector);
    }

    ++m_slotCounter;
    return symPerBeam;
}

} // namespace ns3

##Versão de ajuste de alocação