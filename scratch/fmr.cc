// fmr.cc

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/buildings-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/enum.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FmrSingleFlowExample");

static inline uint32_t
ClampProtocol01(uint32_t v)
{
    return (v == 1) ? 1u : 0u;
}

int
main(int argc, char* argv[])
{
    uint16_t gNbNum = 1;
    uint16_t ueNumPergNb = 9;
    bool logging = true;

    Time simTime = Seconds(5.0);
    Time udpAppStartTime = MilliSeconds(400);

    double centralFrequency = 4e9;
    double bandwidth = 100e6;
    double totalTxPowerDbm = 43.0;

    // FMR
    bool useFmr = true;
    double fmrAlphaFixed = 0.7;
    double fmrTau = 0.70;

    // ns3-ai
    bool enableNs3Ai = false;
    uint32_t aiShmSize = 4096;
    std::string aiSegmentName = "ns3ai_fmr";
    std::string aiCpp2PyName = "fmr_cpp2py";
    std::string aiPy2CppName = "fmr_py2cpp";
    std::string aiLockableName = "fmr_lock";

    bool aiCppIsCreator = true;
    uint32_t aiProtocol = 0; // 0=SendThenRecv, 1=RecvThenSend
    bool aiVerbose = true;

    // Scheduler slot CSV
    bool enableSlotCsv = false;
    std::string slotCsvPath = "default-slot-log.csv";
    bool slotCsvAppend = false;
    bool slotCsvFlush = false;

    // Traffic
    uint32_t udpPacketSize = 3000;
    uint32_t lambda = 1000;
    uint16_t dlPortBase = 1234;

    uint32_t seed = 1;
    uint32_t run = 1;

    std::string simTag = "default";
    std::string outputDir = "./";

    CommandLine cmd;
    cmd.AddValue("gNbNum", "The number of gNbs", gNbNum);
    cmd.AddValue("ueNumPergNb", "The number of UE per gNb", ueNumPergNb);
    cmd.AddValue("logging", "Enable logging", logging);

    cmd.AddValue("simTime", "Simulation time", simTime);
    cmd.AddValue("udpAppStartTime", "UDP start time", udpAppStartTime);

    cmd.AddValue("centralFrequency", "Central frequency (Hz)", centralFrequency);
    cmd.AddValue("bandwidth", "Bandwidth (Hz)", bandwidth);
    cmd.AddValue("totalTxPower", "Total TX power (dBm)", totalTxPowerDbm);

    cmd.AddValue("UseFmr", "Use OFDMA FMR scheduler (DL only).", useFmr);
    cmd.AddValue("FmrAlphaFixed", "FMR fixed alpha (AlphaMode=Fixed).", fmrAlphaFixed);
    cmd.AddValue("FmrTau", "FMR tau.", fmrTau);

    cmd.AddValue("EnableNs3Ai", "Enable ns3-ai decisions (Python).", enableNs3Ai);
    cmd.AddValue("AiShmSize", "Shared memory size.", aiShmSize);
    cmd.AddValue("AiSegmentName", "Shared memory segment name.", aiSegmentName);
    cmd.AddValue("AiCpp2PyName", "Cpp->Py msg name.", aiCpp2PyName);
    cmd.AddValue("AiPy2CppName", "Py->Cpp msg name.", aiPy2CppName);
    cmd.AddValue("AiLockableName", "Lockable name.", aiLockableName);

    cmd.AddValue("AiCppIsCreator", "C++ creates shm (1) or Python creates shm (0).", aiCppIsCreator);
    cmd.AddValue("AiProtocol", "0=SendThenRecv, 1=RecvThenSend.", aiProtocol);
    cmd.AddValue("AiVerbose", "Verbose AI logs.", aiVerbose);

    cmd.AddValue("EnableSlotCsv", "Enable per-slot CSV in scheduler (FMR only).", enableSlotCsv);
    cmd.AddValue("SlotCsvPath", "CSV path for per-slot scheduler log.", slotCsvPath);
    cmd.AddValue("SlotCsvAppend", "Append scheduler CSV.", slotCsvAppend);
    cmd.AddValue("SlotCsvFlush", "Flush scheduler CSV at every write.", slotCsvFlush);

    cmd.AddValue("udpPacketSize", "UDP packet size (bytes)", udpPacketSize);
    cmd.AddValue("lambda", "UDP packet rate (packets/s)", lambda);
    cmd.AddValue("dlPort", "Downlink base port used by UEs (ports are dlPort+i)", dlPortBase);

    cmd.AddValue("seed", "RNG seed", seed);
    cmd.AddValue("run", "RNG run number", run);

    cmd.AddValue("simTag", "Tag appended to output files", simTag);
    cmd.AddValue("outputDir", "Directory to store results", outputDir);

    cmd.Parse(argc, argv);

    // clamp protocol (avoid invalid values)
    aiProtocol = ClampProtocol01(aiProtocol);

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);

    if (logging)
    {
        LogComponentEnable("FmrSingleFlowExample", LOG_LEVEL_INFO);
        LogComponentEnable("NrMacSchedulerOfdmaFmr", LOG_LEVEL_ALL);
    }

    NodeContainer gnbNodes;
    gnbNodes.Create(gNbNum);

    NodeContainer ueNodes;
    ueNodes.Create(ueNumPergNb);

    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    mobility.Install(gnbNodes);
    gnbNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(0.0, 0.0, 10.0));

    mobility.Install(ueNodes);
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        const double x = 10.0 + 10.0 * static_cast<double>(i);
        const double y = 5.0;
        ueNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(x, y, 1.5));
    }

    Ptr<NrPointToPointEpcHelper> nrEpcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetBeamformingHelper(idealBeamformingHelper);
    nrHelper->SetEpcHelper(nrEpcHelper);

    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;

    CcBwpCreator::SimpleOperationBandConf bandConf(centralFrequency, bandwidth, numCcPerBand);
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMa", "Default", "ThreeGpp");
    channelHelper->AssignChannelsToBands({band});
    allBwps = CcBwpCreator::GetAllBwps({band});

    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                         TypeIdValue(DirectPathBeamforming::GetTypeId()));

    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));

    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));

    if (useFmr)
    {
        nrHelper->SetSchedulerTypeId(TypeId::LookupByName("ns3::NrMacSchedulerOfdmaFmr"));

        nrHelper->SetSchedulerAttribute("AlphaFixed", DoubleValue(fmrAlphaFixed));
        nrHelper->SetSchedulerAttribute("Tau", DoubleValue(fmrTau));

        nrHelper->SetSchedulerAttribute("EnableNs3Ai", BooleanValue(enableNs3Ai));
        nrHelper->SetSchedulerAttribute("AiShmSize", UintegerValue(aiShmSize));
        nrHelper->SetSchedulerAttribute("AiSegmentName", StringValue(aiSegmentName));
        nrHelper->SetSchedulerAttribute("AiCpp2PyName", StringValue(aiCpp2PyName));
        nrHelper->SetSchedulerAttribute("AiPy2CppName", StringValue(aiPy2CppName));
        nrHelper->SetSchedulerAttribute("AiLockableName", StringValue(aiLockableName));

        // ✅ FIX DEFINITIVO:
        if (aiProtocol == 1)
        {
            nrHelper->SetSchedulerAttribute("AiProtocol", StringValue("RecvThenSend"));
        }
        else
        {
            nrHelper->SetSchedulerAttribute("AiProtocol", StringValue("SendThenRecv"));
        }

        nrHelper->SetSchedulerAttribute("AiVerbose", BooleanValue(aiVerbose));
        nrHelper->SetSchedulerAttribute("AiCppIsCreator", BooleanValue(aiCppIsCreator));

        nrHelper->SetSchedulerAttribute("EnableSlotCsv", BooleanValue(enableSlotCsv));
        nrHelper->SetSchedulerAttribute("SlotCsvPath", StringValue(slotCsvPath));
        nrHelper->SetSchedulerAttribute("SlotCsvAppend", BooleanValue(slotCsvAppend));
        nrHelper->SetSchedulerAttribute("SlotCsvFlush", BooleanValue(slotCsvFlush));
    }

    NetDeviceContainer gnbDevs = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueDevs = nrHelper->InstallUeDevice(ueNodes, allBwps);

    NrHelper::GetGnbPhy(gnbDevs.Get(0), 0)->SetTxPower(totalTxPowerDbm);

    auto [remoteHost, remoteHostIpv4Address] =
        nrEpcHelper->SetupRemoteHost("100Gb/s", 2500, Seconds(0.010));

    InternetStackHelper internet;
    internet.Install(ueNodes);

    Ipv4InterfaceContainer ueIpIface = nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));

    for (uint32_t i = 0; i < ueDevs.GetN(); ++i)
    {
        nrHelper->AttachToGnb(ueDevs.Get(i), gnbDevs.Get(0));
    }

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNodes.Get(i)->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(nrEpcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    ApplicationContainer serverApps;
    ApplicationContainer clientApps;
    std::vector<Ptr<UdpServer>> servers;
    servers.reserve(ueNodes.GetN());

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        uint16_t port = dlPortBase + static_cast<uint16_t>(i);

        UdpServerHelper server(port);
        ApplicationContainer s = server.Install(ueNodes.Get(i));
        serverApps.Add(s);
        servers.push_back(DynamicCast<UdpServer>(s.Get(0)));

        UdpClientHelper client(ueIpIface.GetAddress(i), port);
        client.SetAttribute("PacketSize", UintegerValue(udpPacketSize));
        client.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
        client.SetAttribute("Interval", TimeValue(Seconds(1.0 / static_cast<double>(lambda))));
        clientApps.Add(client.Install(remoteHost));
    }

    serverApps.Start(udpAppStartTime);
    clientApps.Start(udpAppStartTime);
    serverApps.Stop(simTime);
    clientApps.Stop(simTime);

    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll();

    Simulator::Stop(simTime);
    Simulator::Run();

    const double activeSeconds = (simTime - udpAppStartTime).GetSeconds();
    double aggMbps = 0.0;

    for (uint32_t i = 0; i < servers.size(); ++i)
    {
        uint64_t bytes =
            servers[i] ? servers[i]->GetReceived() * static_cast<uint64_t>(udpPacketSize) : 0;
        double mbps =
            (activeSeconds > 0.0) ? (8.0 * static_cast<double>(bytes) / activeSeconds / 1e6) : 0.0;
        aggMbps += mbps;
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[RESULT] DL aggregate throughput ~= " << aggMbps << " Mbps"
              << " simTag=" << simTag
              << " EnableSlotCsv=" << (enableSlotCsv ? 1 : 0)
              << " SlotCsvPath=" << slotCsvPath
              << " EnableNs3Ai=" << (enableNs3Ai ? 1 : 0)
              << " AiProtocol=" << aiProtocol
              << "\n";

    if (monitor)
    {
        monitor->CheckForLostPackets();
    }

    Simulator::Destroy();
    return 0;
}