#include "Simulation.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

using namespace modern;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " [-C|-A|-R] [-O|-R|-B] input_file1 [input_file2 ...]" << std::endl;
        std::cerr << "  -C|-A|-R: Geometry (Chain, Arbitrary, Ring)" << std::endl;
        std::cerr << "  -O|-R: Tensor (Oseen, Rotne-Prager, Blake)" << std::endl;
        return -1;
    }

    GeometryType geometry = GeometryType::RING;
    if (std::strncmp(argv[1], "-C", 2) == 0) {
        geometry = GeometryType::CHAIN;
    } else if (std::strncmp(argv[1], "-A", 2) == 0) {
        geometry = GeometryType::ARBITRARY;
    } else if (std::strncmp(argv[1], "-R", 2) == 0) {
        geometry = GeometryType::RING;
    }

    MobilityType mobility = MobilityType::ROTNE_PRAGER;
    if (std::strncmp(argv[2], "-O", 2) == 0) {
        mobility = MobilityType::OSEEN;
    } else if (std::strcmp(argv[2], "-B") == 0) {
        mobility = MobilityType::BLAKE;
    } else if (std::strcmp(argv[2], "-R") != 0) {
        std::cerr << "Unknown mobility flag\n"; return 1;
    }
    int status = 0;

    for (int i = 3; i < argc; ++i) {
        try {
            std::string input_file = argv[i];
            Simulation sim(input_file, geometry, mobility);
            sim.run();
        } catch (const std::exception& e) {
            status = 1;
            std::cerr << "Error running simulation for " << argv[i] << ": " << e.what() << std::endl;
        }
    }

    return status;
}
