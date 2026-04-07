#include "ns3/core-module.h"
#include "ns3/nr-module.h"

using namespace ns3;

int main(int argc, char** argv)
{
    TypeId tid;
    bool ok = TypeId::LookupByNameFailSafe("ns3::NrMacSchedulerOfdmaFmr", &tid);

    std::cout << "LookupByNameFailSafe(ns3::NrMacSchedulerOfdmaFmr) = " << ok << std::endl;
    if (ok)
    {
        std::cout << "TypeId uid=" << tid.GetUid() << " name=" << tid.GetName() << std::endl;
    }
    return ok ? 0 : 1;
}
