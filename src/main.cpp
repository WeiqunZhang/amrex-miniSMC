#include "MiniSMC.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>

int main(int argc, char* argv[])
{
    if (argc > 1) {
        minismc::MiniSMC::SetInputFilePath(argv[1]);
    }
    amrex::Initialize(argc, argv);
    {
        minismc::MiniSMC driver;
        driver.Evolve();
    }
    amrex::Finalize();
    return 0;
}
