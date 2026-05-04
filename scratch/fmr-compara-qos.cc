#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/buildings-module.h"
#include "ns3/config-store-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FmrComparaQos");

static void
SetRandomPositions(const NodeContainer& ueNodes,
                   Ptr<Node> gnbNode,
                   double minDist,
                   double maxDist,
                   double ueHeight,
                   int64_t stream)
{
    Ptr<MobilityModel> gnbMob = gnbNode->GetObject<MobilityModel>();
    Vector gnbPos(0.0, 0.0, ueHeight);
    if (gnbMob)
    {
        gnbPos = gnbMob->GetPosition();
    }

    Ptr<UniformRandomVariable> rvR = CreateObject<UniformRandomVariable>();
    Ptr<UniformRandomVariable> rvA = CreateObject<UniformRandomVariable>();

    rvR->SetAttribute("Min", DoubleValue(minDist));
    rvR->SetAttribute("Max", DoubleValue(maxDist));
    rvA->SetAttribute("Min", DoubleValue(0.0));
    rvA->SetAttribute("Max", DoubleValue(2.0 * M_PI));

    rvR->SetStream(stream);
    rvA->SetStream(stream + 1);

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        Ptr<MobilityModel> mm = ueNodes.Get(i)->GetObject<MobilityModel>();
        if (!mm)
        {
            continue;
        }

        const double r = rvR->GetValue();
        const double a = rvA->GetValue();
        Vector pos(gnbPos.x + r * std::cos(a), gnbPos.y + r * std::sin(a), ueHeight);
        mm->SetPosition(pos);

        NS_LOG_INFO("UE " << i << " random position = (" << pos.x << ", " << pos.y << ", "
                          << pos.z << "), r=" << r << ", theta=" << a);
    }
}

static void
SetFixedProfilePositions(const NodeContainer& ueNodes, Ptr<Node> gnbNode, double ueHeight)
{
    Ptr<MobilityModel> gnbMob = gnbNode->GetObject<MobilityModel>();
    Vector gnbPos(0.0, 0.0, ueHeight);
    if (gnbMob)
    {
        gnbPos = gnbMob->GetPosition();
    }

    const std::vector<double> radii = {20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 150.0};
    const std::vector<double> angles = {
        0.0,
        M_PI / 6.0,
        M_PI / 3.0,
        M_PI / 2.0,
        2.0 * M_PI / 3.0,
        5.0 * M_PI / 6.0,
        M_PI,
        4.0 * M_PI / 3.0,
        5.0 * M_PI / 3.0};

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        Ptr<MobilityModel> mm = ueNodes.Get(i)->GetObject<MobilityModel>();
        if (!mm)
        {
            continue;
        }

        const double r = radii[std::min<uint32_t>(i, radii.size() - 1)];
        const double a = angles[std::min<uint32_t>(i, angles.size() - 1)];

        Vector pos(gnbPos.x + r * std::cos(a), gnbPos.y + r * std::sin(a), ueHeight);
        mm->SetPosition(pos);

        NS_LOG_INFO("UE " << i << " fixed-profile position = (" << pos.x << ", " << pos.y
                          << ", " << pos.z << "), r=" << r << ", theta=" << a);
    }
}

static void
EnableRandomWalkMobility(const NodeContainer& ueNodes,
                         double speedMin,
                         double speedMax,
                         double bounds,
                         double stepDistance)
{
    Rectangle rect(-bounds, bounds, -bounds, bounds);

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        Ptr<MobilityModel> mm = ueNodes.Get(i)->GetObject<MobilityModel>();
        Vector pos(0.0, 0.0, 1.5);
        if (mm)
        {
            pos = mm->GetPosition();
        }

        Ptr<RandomWalk2dMobilityModel> rw = CreateObject<RandomWalk2dMobilityModel>();
        rw->SetAttribute("Mode", EnumValue(RandomWalk2dMobilityModel::MODE_DISTANCE));
        rw->SetAttribute("Distance", DoubleValue(stepDistance));
        rw->SetAttribute("Bounds", RectangleValue(rect));

        std::ostringstream speedStr;
        speedStr << "ns3::UniformRandomVariable[Min=" << speedMin << "|Max=" << speedMax << "]";
        rw->SetAttribute("Speed", StringValue(speedStr.str()));

        ueNodes.Get(i)->AggregateObject(rw);
        rw->SetPosition(pos);
    }
}

static bool
UsesFmr(const std::string& schedulerMode)
{
    return schedulerMode == "fmr_rl";
}

static std::string
SchedulerTypeFromMode(const std::string& schedulerMode)
{
    if (schedulerMode == "rr")
    {
        return "ns3::NrMacSchedulerOfdmaRR";
    }
    if (schedulerMode == "pf")
    {
        return "ns3::NrMacSchedulerOfdmaPF";
    }
    if (schedulerMode == "mr")
    {
        return "ns3::NrMacSchedulerOfdmaMR";
    }
    if (schedulerMode == "fmr_rl")
    {
        return "ns3::NrMacSchedulerOfdmaFmr";
    }

    NS_FATAL_ERROR("Invalid schedulerMode=" << schedulerMode << ". Use rr, pf, mr or fmr_rl.");
    return "";
}

static void
WriteUeSnapshotHeader(std::ofstream& out)
{
    out << "time_s,scheduler_mode,ue_idx,ue_x,ue_y,ue_z,gnb_x,gnb_y,gnb_z,dist_to_gnb_m\n";
}

static void
WriteUeSnapshotRow(std::ofstream& out,
                   double timeS,
                   const std::string& schedulerMode,
                   uint32_t ueIdx,
                   const Vector& uePos,
                   const Vector& gnbPos)
{
    const double dx = uePos.x - gnbPos.x;
    const double dy = uePos.y - gnbPos.y;
    const double dz = uePos.z - gnbPos.z;
    const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    out << std::fixed << std::setprecision(6)
        << timeS << ","
        << schedulerMode << ","
        << ueIdx << ","
        << uePos.x << ","
        << uePos.y << ","
        << uePos.z << ","
        << gnbPos.x << ","
        << gnbPos.y << ","
        << gnbPos.z << ","
        << dist << "\n";
}

int
main(int argc, char* argv[])
{
    uint16_t gNbNum = 1;
    uint16_t ueNumPergNb = 9;
    bool logging = true;

    Time simTime = MilliSeconds(2000);
    Time udpAppStartTime = MilliSeconds(400);

    uint16_t numerology = 1;
    double centralFrequency = 4e9;
    double bandwidth = 100e6;
    double totalTxPower = 43.0;
    uint16_t mcsTable = 2;

    uint32_t udpPacketSize = 3000;
    uint32_t lambda = 200;
    uint16_t dlPort = 1234;

    uint16_t numDlFlowsPerUe = 1;

    std::string schedulerMode = "rr";
    std::string positionMode = "random"; // random | fixed_profile

    double scenarioRadiusMin = 20.0;
    double scenarioRadiusMax = 150.0;
    double bsHeight = 1.5;
    double ueHeight = 1.5;

    bool enableMobility = false;
    double mobilitySpeedMin = 0.5;
    double mobilitySpeedMax = 1.5;
    double mobilityBounds = 200.0;
    double mobilityDistance = 5.0;

    bool enableUeSnapshotCsv = true;
    std::string ueSnapshotCsvPath = "ue_snapshot.csv";
    Time ueSnapshotPeriod = MilliSeconds(100);

    bool enableFlowSummaryCsv = true;
    std::string flowSummaryCsvPath = "flow_summary.csv";

    // FMR RL
    bool enableNs3Ai = true;
    uint32_t aiShmSize = 4096;
    std::string aiSegmentName = "ns3ai_fmr";
    std::string aiCpp2PyName = "fmr_cpp2py";
    std::string aiPy2CppName = "fmr_py2cpp";
    std::string aiLockableName = "fmr_lock";
    bool aiCppIsCreator = true;
    bool aiVerbose = true;
    double fmrTau = 0.70;
    bool enableSlotCsv = true;
    std::string slotCsvPath = "slot_log.csv";
    bool slotCsvAppend = false;
    bool slotCsvFlush = true;

    std::string simTag = "summary.txt";
    std::string outputDir = "./";

    CommandLine cmd;
    cmd.AddValue("gNbNum", "The number of gNbs", gNbNum);
    cmd.AddValue("ueNumPergNb", "The number of UE per gNb", ueNumPergNb);
    cmd.AddValue("logging", "Enable logging", logging);
    cmd.AddValue("simTime", "Simulation time", simTime);
    cmd.AddValue("numerology", "The numerology to be used", numerology);
    cmd.AddValue("centralFrequency", "The system frequency to be used", centralFrequency);
    cmd.AddValue("bandwidth", "The system bandwidth to be used", bandwidth);
    cmd.AddValue("totalTxPower", "The total tx power", totalTxPower);
    cmd.AddValue("mcsTable", "MCS table index", mcsTable);

    cmd.AddValue("schedulerMode", "rr | pf | mr | fmr_rl", schedulerMode);
    cmd.AddValue("positionMode", "random | fixed_profile", positionMode);

    cmd.AddValue("udpPacketSize", "UDP packet size in bytes", udpPacketSize);
    cmd.AddValue("lambda", "UDP packet rate (packets/s)", lambda);
    cmd.AddValue("dlPort", "DL base port", dlPort);
    
    cmd.AddValue("numDlFlowsPerUe", "Number of DL UDP flows per UE", numDlFlowsPerUe);

    cmd.AddValue("scenarioRadiusMin", "Minimum UE distance from gNB (m)", scenarioRadiusMin);
    cmd.AddValue("scenarioRadiusMax", "Maximum UE distance from gNB (m)", scenarioRadiusMax);
    cmd.AddValue("bsHeight", "gNB height", bsHeight);
    cmd.AddValue("ueHeight", "UE height", ueHeight);

    cmd.AddValue("enableMobility", "Enable UE mobility", enableMobility);
    cmd.AddValue("mobilitySpeedMin", "Minimum UE speed", mobilitySpeedMin);
    cmd.AddValue("mobilitySpeedMax", "Maximum UE speed", mobilitySpeedMax);
    cmd.AddValue("mobilityBounds", "Mobility bounds", mobilityBounds);
    cmd.AddValue("mobilityDistance", "Random walk distance", mobilityDistance);

    cmd.AddValue("EnableUeSnapshotCsv", "Enable UE snapshot CSV", enableUeSnapshotCsv);
    cmd.AddValue("UeSnapshotCsvPath", "UE snapshot CSV path", ueSnapshotCsvPath);
    cmd.AddValue("UeSnapshotPeriod", "UE snapshot period", ueSnapshotPeriod);

    cmd.AddValue("EnableFlowSummaryCsv", "Enable flow summary CSV", enableFlowSummaryCsv);
    cmd.AddValue("FlowSummaryCsvPath", "Flow summary CSV path", flowSummaryCsvPath);

    cmd.AddValue("EnableNs3Ai", "Enable ns3-ai for fmr_rl", enableNs3Ai);
    cmd.AddValue("AiShmSize", "Shared memory size", aiShmSize);
    cmd.AddValue("AiSegmentName", "Shared memory segment name", aiSegmentName);
    cmd.AddValue("AiCpp2PyName", "Cpp->Py name", aiCpp2PyName);
    cmd.AddValue("AiPy2CppName", "Py->Cpp name", aiPy2CppName);
    cmd.AddValue("AiLockableName", "Lock name", aiLockableName);
    cmd.AddValue("AiCppIsCreator", "C++ creates shm", aiCppIsCreator);
    cmd.AddValue("AiVerbose", "Verbose AI logs", aiVerbose);
    cmd.AddValue("FmrTau", "FMR tau", fmrTau);
    cmd.AddValue("EnableSlotCsv", "Enable FMR slot CSV", enableSlotCsv);
    cmd.AddValue("SlotCsvPath", "FMR slot CSV path", slotCsvPath);
    cmd.AddValue("SlotCsvAppend", "Append slot CSV", slotCsvAppend);
    cmd.AddValue("SlotCsvFlush", "Flush slot CSV", slotCsvFlush);

    cmd.AddValue("simTag", "Output summary filename", simTag);
    cmd.AddValue("outputDir", "Output directory", outputDir);

    
   

    cmd.Parse(argc, argv);

    std::filesystem::create_directories(outputDir);

    if (ueSnapshotCsvPath == "ue_snapshot.csv")
    {
        ueSnapshotCsvPath = outputDir + "/ue_snapshot_" + schedulerMode + ".csv";
    }
    if (flowSummaryCsvPath == "flow_summary.csv")
    {
        flowSummaryCsvPath = outputDir + "/flow_summary_" + schedulerMode + ".csv";
    }
    if (slotCsvPath == "slot_log.csv")
    {
        slotCsvPath = outputDir + "/slot_log_" + schedulerMode + ".csv";
    }

    if (logging)
    {
        auto logLevel =
            (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME | LOG_PREFIX_NODE | LOG_LEVEL_INFO);
        LogComponentEnable("FmrComparaQos", logLevel);
        LogComponentEnable("NrMacSchedulerNs3", logLevel);
        LogComponentEnable("NrMacSchedulerTdma", logLevel);
        LogComponentEnable("NrMacSchedulerOfdmaFmr", logLevel);
    }

    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));

    int64_t randomStream = 1;

    GridScenarioHelper gridScenario;
    gridScenario.SetRows(1);
    gridScenario.SetColumns(gNbNum);
    gridScenario.SetHorizontalBsDistance(5.0);
    gridScenario.SetVerticalBsDistance(5.0);
    gridScenario.SetBsHeight(bsHeight);
    gridScenario.SetUtHeight(ueHeight);
    gridScenario.SetSectorization(GridScenarioHelper::SINGLE);
    gridScenario.SetBsNumber(gNbNum);
    gridScenario.SetUtNumber(ueNumPergNb * gNbNum);
    gridScenario.SetScenarioHeight(2 * mobilityBounds);
    gridScenario.SetScenarioLength(2 * mobilityBounds);
    randomStream += gridScenario.AssignStreams(randomStream);
    gridScenario.CreateScenario();

    Ptr<Node> gnb0 = gridScenario.GetBaseStations().Get(0);

    if (positionMode == "random")
    {
        SetRandomPositions(gridScenario.GetUserTerminals(),
                           gnb0,
                           scenarioRadiusMin,
                           scenarioRadiusMax,
                           ueHeight,
                           randomStream);
    }
    else if (positionMode == "fixed_profile")
    {
        SetFixedProfilePositions(gridScenario.GetUserTerminals(), gnb0, ueHeight);
    }
    else
    {
        NS_FATAL_ERROR("Invalid positionMode=" << positionMode << ". Use random or fixed_profile.");
    }

    if (enableMobility)
    {
        EnableRandomWalkMobility(gridScenario.GetUserTerminals(),
                                 mobilitySpeedMin,
                                 mobilitySpeedMax,
                                 mobilityBounds,
                                 mobilityDistance);
    }

    Ptr<NrPointToPointEpcHelper> nrEpcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();

    nrHelper->SetBeamformingHelper(idealBeamformingHelper);
    nrHelper->SetEpcHelper(nrEpcHelper);
    nrEpcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));

    std::string schedulerType = SchedulerTypeFromMode(schedulerMode);
    std::cout << "SchedulerType: " << schedulerType << std::endl;
    nrHelper->SetSchedulerTypeId(TypeId::LookupByName(schedulerType));

    /*
     * Slot allocation CSV.
     *
     * For native OFDMA schedulers (RR, PF, MR), the logger is the common
     * logger added to NrMacSchedulerOfdma.
     *
     * For FMR, use the FMR-specific logger because it also records
     * target_rbg and alpha.
     */
    if (UsesFmr(schedulerMode))
    {
        nrHelper->SetSchedulerAttribute("Tau", DoubleValue(fmrTau));

        nrHelper->SetSchedulerAttribute("EnableSlotCsv", BooleanValue(enableSlotCsv));
        nrHelper->SetSchedulerAttribute("SlotCsvPath", StringValue(slotCsvPath));
        nrHelper->SetSchedulerAttribute("SlotCsvAppend", BooleanValue(slotCsvAppend));
        nrHelper->SetSchedulerAttribute("SlotCsvFlush", BooleanValue(slotCsvFlush));

        nrHelper->SetSchedulerAttribute("EnableNs3Ai", BooleanValue(enableNs3Ai));
        nrHelper->SetSchedulerAttribute("AiShmSize", UintegerValue(aiShmSize));
        nrHelper->SetSchedulerAttribute("AiSegmentName", StringValue(aiSegmentName));
        nrHelper->SetSchedulerAttribute("AiCpp2PyName", StringValue(aiCpp2PyName));
        nrHelper->SetSchedulerAttribute("AiPy2CppName", StringValue(aiPy2CppName));
        nrHelper->SetSchedulerAttribute("AiLockableName", StringValue(aiLockableName));
        nrHelper->SetSchedulerAttribute("AiCppIsCreator", BooleanValue(aiCppIsCreator));
        nrHelper->SetSchedulerAttribute("AiVerbose", BooleanValue(aiVerbose));
    }
    else
    {
        nrHelper->SetSchedulerAttribute("EnableCommonSlotCsv", BooleanValue(enableSlotCsv));
        nrHelper->SetSchedulerAttribute("CommonSlotCsvPath", StringValue(slotCsvPath));
        nrHelper->SetSchedulerAttribute("CommonSlotCsvAppend", BooleanValue(slotCsvAppend));
        nrHelper->SetSchedulerAttribute("CommonSlotCsvFlush", BooleanValue(slotCsvFlush));
    }

    std::string errorModel = "ns3::NrEesmIrT" + std::to_string(mcsTable);
    nrHelper->SetDlErrorModel(errorModel);
    nrHelper->SetUlErrorModel(errorModel);
    nrHelper->SetGnbDlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));
    nrHelper->SetGnbUlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));

    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                    TypeIdValue(DirectPathBeamforming::GetTypeId()));
    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));

    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(1));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(1));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));

    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    OperationBandInfo band;
    const uint8_t numOfCcs = 1;

    CcBwpCreator::SimpleOperationBandConf bandConf(centralFrequency, bandwidth, numOfCcs);
    bandConf.m_numBwp = 1;

    band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMi", "Default", "ThreeGpp");
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));
    Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue(MilliSeconds(0)));
    channelHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    channelHelper->AssignChannelsToBands({band});
    allBwps = CcBwpCreator::GetAllBwps({band});

    double x = std::pow(10.0, totalTxPower / 10.0);

    Packet::EnableChecking();
    Packet::EnablePrinting();

    uint32_t bwpId = 0;
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB", UintegerValue(bwpId));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB", UintegerValue(bwpId));

    NetDeviceContainer gnbNetDev =
        nrHelper->InstallGnbDevice(gridScenario.GetBaseStations(), allBwps);
    NetDeviceContainer ueNetDev =
        nrHelper->InstallUeDevice(gridScenario.GetUserTerminals(), allBwps);

    randomStream += nrHelper->AssignStreams(gnbNetDev, randomStream);
    randomStream += nrHelper->AssignStreams(ueNetDev, randomStream);

    NrHelper::GetGnbPhy(gnbNetDev.Get(0), 0)->SetAttribute("Numerology", UintegerValue(numerology));
    NrHelper::GetGnbPhy(gnbNetDev.Get(0), 0)->SetAttribute("TxPower",
                                                           DoubleValue(10.0 * std::log10(x)));

    auto [remoteHost, remoteHostIpv4Address] =
        nrEpcHelper->SetupRemoteHost("100Gb/s", 2500, Seconds(0.000));

    InternetStackHelper internet;
    internet.Install(gridScenario.GetUserTerminals());

    Ipv4InterfaceContainer ueIpIface =
        nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(ueNetDev));

    nrHelper->AttachToClosestGnb(ueNetDev, gnbNetDev);

    ApplicationContainer serverApps;
    ApplicationContainer clientApps;

    UdpClientHelper dlClient;
    dlClient.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    dlClient.SetAttribute("PacketSize", UintegerValue(udpPacketSize));
    dlClient.SetAttribute("Interval", TimeValue(Seconds(1.0 / lambda)));

    NrEpsBearer bearer(NrEpsBearer::NGBR_LOW_LAT_EMBB);

    for (uint32_t i = 0; i < gridScenario.GetUserTerminals().GetN(); ++i)
    {
        Address ueAddress = ueIpIface.GetAddress(i);

        Ptr<NrEpcTft> tft = Create<NrEpcTft>();

        for (uint16_t f = 0; f < numDlFlowsPerUe; ++f)
        {
            uint16_t flowPort = dlPort + static_cast<uint16_t>(i * numDlFlowsPerUe + f);

            UdpServerHelper dlPacketSink(flowPort);
            serverApps.Add(dlPacketSink.Install(gridScenario.GetUserTerminals().Get(i)));

            dlClient.SetAttribute(
                "Remote",
                AddressValue(addressUtils::ConvertToSocketAddress(ueAddress, flowPort)));
            clientApps.Add(dlClient.Install(remoteHost));

            NrEpcTft::PacketFilter dlpf;
            dlpf.localPortStart = flowPort;
            dlpf.localPortEnd = flowPort;
            tft->Add(dlpf);
        }

        Ptr<NetDevice> ueDevice = ueNetDev.Get(i);
        nrHelper->ActivateDedicatedEpsBearer(ueDevice, bearer, tft);
    }

    serverApps.Start(udpAppStartTime);
    clientApps.Start(udpAppStartTime);
    serverApps.Stop(simTime);
    clientApps.Stop(simTime);

    std::ofstream ueSnapshotOut;
    if (enableUeSnapshotCsv)
    {
        ueSnapshotOut.open(ueSnapshotCsvPath, std::ofstream::out | std::ofstream::trunc);
        if (!ueSnapshotOut.is_open())
        {
            std::cerr << "Can't open UE snapshot CSV: " << ueSnapshotCsvPath << std::endl;
            return 1;
        }

        WriteUeSnapshotHeader(ueSnapshotOut);

        Ptr<MobilityModel> gnbMob = gnb0->GetObject<MobilityModel>();
        auto snapshotFn = [&gridScenario, &ueSnapshotOut, &schedulerMode, gnbMob]() {
            Vector gnbPos(0.0, 0.0, 0.0);
            if (gnbMob)
            {
                gnbPos = gnbMob->GetPosition();
            }

            const NodeContainer& ueNodes = gridScenario.GetUserTerminals();
            const double now = Simulator::Now().GetSeconds();

            for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
            {
                Ptr<MobilityModel> mm = ueNodes.Get(i)->GetObject<MobilityModel>();
                if (!mm)
                {
                    continue;
                }
                WriteUeSnapshotRow(ueSnapshotOut, now, schedulerMode, i, mm->GetPosition(), gnbPos);
            }
        };

        for (Time t = Seconds(0.0); t <= simTime; t += ueSnapshotPeriod)
        {
            Simulator::Schedule(t, snapshotFn);
        }
    }

    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add(remoteHost);
    endpointNodes.Add(gridScenario.GetUserTerminals());

    Ptr<FlowMonitor> monitor = flowmonHelper.Install(endpointNodes);
    monitor->SetAttribute("DelayBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("JitterBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("PacketSizeBinWidth", DoubleValue(20));

    Simulator::Stop(simTime);
    Simulator::Run();

    if (ueSnapshotOut.is_open())
    {
        ueSnapshotOut.close();
    }

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();

    std::ofstream outFile;
    std::string summaryPath = outputDir + "/" + simTag;
    outFile.open(summaryPath.c_str(), std::ofstream::out | std::ofstream::trunc);
    if (!outFile.is_open())
    {
        std::cerr << "Can't open file " << summaryPath << std::endl;
        return 1;
    }

    std::ofstream flowCsvOut;
    if (enableFlowSummaryCsv)
    {
        flowCsvOut.open(flowSummaryCsvPath, std::ofstream::out | std::ofstream::trunc);
        if (!flowCsvOut.is_open())
        {
            std::cerr << "Can't open flow summary CSV: " << flowSummaryCsvPath << std::endl;
            return 1;
        }

        flowCsvOut
            << "scheduler_mode,flow_id,source_ip,source_port,dest_ip,dest_port,protocol,"
            << "tx_packets,tx_bytes,tx_offered_mbps,rx_packets,rx_bytes,throughput_mbps,"
            << "mean_delay_ms,mean_jitter_ms,loss_ratio\n";
    }

    outFile.setf(std::ios_base::fixed);

    double sumFlowThroughput = 0.0;
    double sumFlowDelay = 0.0;
    uint32_t rxFlows = 0;

    double flowDuration = (simTime - udpAppStartTime).GetSeconds();

    for (auto i = stats.begin(); i != stats.end(); ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
        std::stringstream protoStream;
        protoStream << static_cast<uint16_t>(t.protocol);
        if (t.protocol == 6)
        {
            protoStream.str("TCP");
        }
        if (t.protocol == 17)
        {
            protoStream.str("UDP");
        }

        double txOffered = i->second.txBytes * 8.0 / flowDuration / 1e6;
        double thr = 0.0;
        double dly = 0.0;
        double jit = 0.0;
        double lossRatio = 0.0;

        if (i->second.txPackets > 0)
        {
            lossRatio = static_cast<double>(i->second.txPackets - i->second.rxPackets) /
                        static_cast<double>(i->second.txPackets);
        }

        if (i->second.rxPackets > 0)
        {
            thr = i->second.rxBytes * 8.0 / flowDuration / 1e6;
            dly = 1000.0 * i->second.delaySum.GetSeconds() / i->second.rxPackets;
            jit = 1000.0 * i->second.jitterSum.GetSeconds() / i->second.rxPackets;

            sumFlowThroughput += thr;
            sumFlowDelay += dly;
            rxFlows++;
        }

        outFile << "Flow " << i->first << " (" << t.sourceAddress << ":" << t.sourcePort << " -> "
                << t.destinationAddress << ":" << t.destinationPort << ") proto "
                << protoStream.str() << "\n";
        outFile << "  Tx Packets: " << i->second.txPackets << "\n";
        outFile << "  Tx Bytes:   " << i->second.txBytes << "\n";
        outFile << "  TxOffered:  " << txOffered << " Mbps\n";
        outFile << "  Rx Bytes:   " << i->second.rxBytes << "\n";
        outFile << "  Rx Packets: " << i->second.rxPackets << "\n";
        outFile << "  Throughput: " << thr << " Mbps\n";
        outFile << "  Mean delay: " << dly << " ms\n";
        outFile << "  Mean jitter:" << jit << " ms\n";
        outFile << "  Loss ratio: " << lossRatio << "\n\n";

        if (flowCsvOut.is_open())
        {
            flowCsvOut << schedulerMode << ","
                       << i->first << ","
                       << t.sourceAddress << ","
                       << t.sourcePort << ","
                       << t.destinationAddress << ","
                       << t.destinationPort << ","
                       << protoStream.str() << ","
                       << i->second.txPackets << ","
                       << i->second.txBytes << ","
                       << txOffered << ","
                       << i->second.rxPackets << ","
                       << i->second.rxBytes << ","
                       << thr << ","
                       << dly << ","
                       << jit << ","
                       << lossRatio << "\n";
        }
    }

    double aggregateThroughputMbps = sumFlowThroughput;
    double meanFlowThroughputMbps = (rxFlows > 0) ? (sumFlowThroughput / rxFlows) : 0.0;
    double meanFlowDelayMs = (rxFlows > 0) ? (sumFlowDelay / rxFlows) : 0.0;

    outFile << "Aggregate throughput: " << aggregateThroughputMbps << " Mbps\n";
    outFile << "Mean flow throughput: " << meanFlowThroughputMbps << " Mbps\n";
    outFile << "Mean flow delay: " << meanFlowDelayMs << " ms\n";
    outFile.close();

    if (flowCsvOut.is_open())
    {
        flowCsvOut.close();
    }

    std::ifstream f(summaryPath.c_str());
    if (f.is_open())
    {
        std::cout << f.rdbuf();
    }

    std::cout << "[RESULT] schedulerMode=" << schedulerMode
              << " aggregate_throughput_mbps=" << std::fixed << std::setprecision(2)
              << aggregateThroughputMbps
              << " mean_flow_throughput_mbps=" << meanFlowThroughputMbps
              << " positionMode=" << positionMode
              << " EnableNs3Ai=" << (UsesFmr(schedulerMode) ? enableNs3Ai : false)
              << " UeSnapshotCsvPath=" << ueSnapshotCsvPath
              << " FlowSummaryCsvPath=" << flowSummaryCsvPath;

    if (UsesFmr(schedulerMode))
    {
        std::cout << " SlotCsvPath=" << slotCsvPath;
    }
    std::cout << std::endl;

    Simulator::Destroy();
    return 0;
}
