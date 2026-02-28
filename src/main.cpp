#include "MiniSMC.H"

#include <AMReX.H>

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        minismc::MiniSMC driver;
        driver.Evolve();
    }
    amrex::Finalize();
    return 0;
}
