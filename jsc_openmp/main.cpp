#include "hpdbscan.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <string>
#include <stdexcept>
#include <unistd.h>

struct Parameters
{
    std::string file;
    std::string out;
    size_t      threads;
    size_t      minPoints;
    float       epsilon;
    
    Parameters() : 
        file("data.csv"), 
        out("out.csv"),
        threads(omp_get_max_threads()),
        minPoints(0),
        epsilon(0)
    {}
};

void usage(char* program)
{
    std::cout << "Usage: " << program << " -m minPoints -e epsilon [OPTIONS] [FILE=data.csv]" << std::endl
              << "    Format : One data point per line, whereby each line contains the space-seperated values for each dimension '<dim 1> <dim 2> ... <dim n>'" << std::endl
              << "    -m minPoints : DBSCAN parameter, minimum number of points required to form a cluster, postive integer, required" << std::endl
              << "    -e epsilon   : DBSCAN parameter, maximum neighborhood search radius for cluster, positive floating point, required" << std::endl
              << "    -o output    : Processing parameter, path to output file, string, defaults to out.csv" << std::endl
              << "    -t threads   : Processing parameter, the number of threads to use, positive integer, defaults to number of cores" << std::endl
              << "    -h help      : Show this help message" << std::endl
              << "    Output : A copy of the input data points plus an additional column containing the cluster id, the id 0 denotes noise" << std::endl;
}

inline ssize_t stoll(const char* optarg)
{
    size_t  length;
    ssize_t parsed;
    
    try
    {
        parsed = std::stoll(optarg, &length);
    }
    catch (const std::logic_error& e)
    {
        parsed = 0L;
    }
    
    return length < strlen(optarg) ? 0L : parsed;
}

inline float stof(const char* optarg)
{
    size_t length;
    float  parsed;
    
    try
    {
        parsed = std::stof(optarg, &length);
    }
    catch (const std::logic_error& e)
    {
        parsed = 0.0f;
    }
    
    return length < strlen(optarg) ? 0.0f : parsed;
}

Parameters parse(int argc, char* const* argv)
{
    Parameters parameters;
    char option;
    int errors = 0;
    
    while ((option = getopt(argc, argv, "hm:e:o:t:")) != -1)
    {
        switch (option)
        {
            case 'm':
            {
                ssize_t minPoints = stoll(optarg);
                if (minPoints <= 0L)
                {
                    parameters.minPoints = 1L;
                    std::cerr << "minPoints needs to be a positive integer number, but was " << optarg << std::endl;
                    ++errors;
                }
                else
                {
                    parameters.minPoints = (size_t) minPoints;
                }
                break;
            }
            case 'e':
            {
                float epsilon = stof(optarg);
                if (epsilon <= 0.0f)
                {
                    parameters.epsilon = 1.0f;
                    std::cerr << "epsilon needs to be a positive floating point number, but was " << optarg << std::endl;
                    ++errors;
                }
                else
                {
                    parameters.epsilon = epsilon;
                }
                break;
            }
            case 't':
            {
                ssize_t threads = stoll(optarg);
                if (threads <= 0L)
                {
                    parameters.threads = 1L;
                    std::cerr << "thread count needs to be a positive integer number, but was " << optarg << std::endl;
                    ++errors;
                }
                else
                {
                    parameters.threads = (size_t) threads;
                }
                break;
            }
            case 'o':
            {
                parameters.out = optarg;
                break;
            }
            case 'h':
            {
                usage(argv[0]);
                std::exit(EXIT_SUCCESS);
                break;
            }
            case '?':
            {
                std::cerr << "Unknown option: " << (char) optopt << std::endl;
                ++errors;
                break;
            }
        }
    }
    
    if (argc - optind <= 0)
    {
        parameters.file = "data.csv";
    }
    else if (argc - optind > 1)
    {
        std::cerr << "Please provide only one data file" << std::endl;
        ++errors;
    }
    else
    {
        parameters.file = argv[optind];
    }
    
    if (!parameters.minPoints || !parameters.epsilon)
    {
        std::cerr << "minPoints or epsilon were not provided" << std::endl;
        ++errors;
    }
    
    if (errors)
    {
        std::cerr << std::endl;
        usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }
    
    return parameters;
}

extern "C"
void hpdbscan_f(const char* file, float epsilon, size_t minPoints)
{
    HPDBSCAN scanner(file);
    scanner.scan(epsilon, minPoints);
}

extern "C"
void hpdbscan_a(float* array, size_t size, size_t dimensions, float epsilon, size_t minPoints)
{
    HPDBSCAN scanner(array, size, dimensions);
    scanner.scan(epsilon, minPoints);
}

int main(int argc, char* const* argv)
{
    double start = omp_get_wtime();
    Parameters parameters = parse(argc, argv);
    
    omp_set_num_threads(parameters.threads);
    HPDBSCAN scanner(parameters.file);
    scanner.scan(parameters.epsilon, parameters.minPoints);
    scanner.writeFile(parameters.out);
    
    std::cout << "Took: " << (omp_get_wtime() - start) << "s" << std::endl;
    
    return 0;
}
