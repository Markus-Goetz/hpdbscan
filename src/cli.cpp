/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: Command-line tool for HPDBSCAN
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "cxxopts.h"
#include "hpdbscan.h"

int main(int argc, char** argv) {
    #ifdef WITH_MPI
    int error, provided;
    error = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (error != MPI_SUCCESS or provided != MPI_THREAD_FUNNELED) {
        std::cerr << "Could not initialize MPI with threading support." << std::endl;
        std::exit(1);
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif
    Clusters clusters;

    cxxopts::Options parser("HPDBSCAN", "Highly parallel DBSCAN clustering algorithm");
    parser.add_options()
        ("h, help", "this help message")
        ("m, minPoints", "density threshold", cxxopts::value<size_t>()->default_value("2"))
        ("e, epsilon", "spatial search radius", cxxopts::value<float>()->default_value("0.1"))
        ("t, threads", "utilized threads", cxxopts::value<int>()->default_value(std::to_string(omp_get_max_threads())))
        ("i, input", "input file", cxxopts::value<std::string>()->default_value("data.h5"))
        ("o, output", "output file", cxxopts::value<std::string>()->default_value("data.h5"))
        ("input-dataset", "input dataset name", cxxopts::value<std::string>()->default_value("DATA"))
        ("output-dataset", "output dataset name", cxxopts::value<std::string>()->default_value("CLUSTERS"))
    ;

    // parse the command-line arguments
    cxxopts::ParseResult arguments = [&]() {
        try {
            return parser.parse(argc, argv);
        } catch (cxxopts::OptionException &e) {
            # ifdef WITH_MPI
            if (rank == 0) {
            #endif
            std::cerr << "Could not parse command line:" << std::endl;
            std::cerr << e.what() << std::endl;
            #ifdef WITH_MPI
            }
            MPI_Finalize();
            #endif
            std::exit(0);
        }
    } ();

    if (arguments.count("help") > 0) {
        # ifdef WITH_MPI
        if (rank == 0) {
            std::cout << parser.help() << std::endl;
        }
        # else
        std::cout << parser.help() << std::endl;
        #endif
        goto finalize;
    }

    // run the clustering algorithm
    try {
        HPDBSCAN hpdbscan(arguments["epsilon"].as<float>(), arguments["minPoints"].as<size_t>());
        clusters = hpdbscan.cluster(
            arguments["input"].as<std::string>(),
            arguments["input-dataset"].as<std::string>(),
            arguments["threads"].as<int>()
        );
    } catch (cxxopts::OptionParseException& e) {
        # ifdef WITH_MPI
        if (rank == 0) {
        #endif
        std::cerr << e.what() << std::endl;
        #ifdef WITH_MPI
        }
        #endif
        goto finalize;
    } catch (std::exception& e) {
        # ifdef WITH_MPI
        if (rank == 0) {
        #endif
        std::cerr << e.what() << std::endl;
        #ifdef WITH_MPI
        }
        #endif
        goto finalize;
    }

    try {
        IO::write_hdf5(arguments["output"].as<std::string>(), arguments["output-dataset"].as<std::string>(), clusters);
    } catch (cxxopts::OptionParseException& e) {
        # ifdef WITH_MPI
        if (rank == 0) {
        #endif
        std::cerr << e.what() << std::endl;
        #ifdef WITH_MPI
        }
        #endif
        goto finalize;
    } catch (std::exception& e) {
        # ifdef WITH_MPI
        if (rank == 0) {
        #endif
        std::cerr << e.what() << std::endl;
        #ifdef WITH_MPI
        }
        #endif
        goto finalize;
    }

finalize:
    #ifdef WITH_MPI
    MPI_Finalize();
    #endif
    return 0;
}
