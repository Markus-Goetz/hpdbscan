#include "hpdbscan.h"
#include "util.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <math.h>
#include <omp.h>
#include <sstream>
#include <unordered_set>

/**
 * Internal Operations
 */
void HPDBSCAN::applyRules(const Rules& rules)
{
    #pragma omp parallel for
    for (size_t i = 0; i < this->m_points.size(); ++i)
    {
        const bool core    = this->m_points.corePoint(i);
        ssize_t    cluster = this->m_points.cluster(i);
        ssize_t    found   = rules.rule(cluster);
        
        while (found < NOISE)
        {
            cluster = found;
            found = rules.rule(found);
        }
        this->m_points.overrideCluster(i, cluster, core);
    }
}

Pointz HPDBSCAN::readArray(float* array, size_t size, size_t dimensions)
{
    Pointz points(size, dimensions, true);
    for (size_t i = 0; i < size; ++i)
    {
        wrap(array + i * dimensions, dimensions, points[i]);
    }
    return points;
}

Pointz HPDBSCAN::readFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cout << "Could not open file " << filename << std::endl;
        exit(1);
    }

    std::cout << "Reading " << filename << "... " << std::endl;
    std::string line, buf;
    std::stringstream ss;
    size_t dimensions = 0;

    // get the first line and get the dimensions
    getline(file, line);
    ss.clear();
    ss << line;
    while (ss >> buf) // get the corordinate of the points
    {
        ++dimensions;
    }

    size_t size = 0;
    // get point count
    file.clear();
    file.seekg (0, std::ios::beg);
    while (!file.eof())
    {
        getline(file, line);
        if(!line.length())
        {
            continue;
        }
        ++size;
    }
    
    size_t i = 0;
    Pointz points(size, dimensions);
    // read in points
    file.clear();
    file.seekg (0, std::ios::beg);
    while (!file.eof())
    {
        getline(file, line);
        if (!line.length())
        {
            continue;
        }
        ss.clear();
        ss << line;

        size_t j = 0;
        auto& point = points[i];
        while(ss >> buf && j < dimensions) // get the corordinate of the points
        {
            point[j] = atof(buf.c_str());
            ++j;
        }
        ++i;
    }
    file.close();
    
    std::cout << "\t" << dimensions << "\tDimensions " << std::endl;
    std::cout << "\t" << size << "\tPoints " << std::endl;

    return points;
}

void HPDBSCAN::writeFile(const std::string& outputPath)
{
    std::cout << "Writing Output File ...";
    std::ofstream outputFile(outputPath);
    std::ostream_iterator<float> output(outputFile, " ");
  
    for (size_t i = 0; i < this->m_points.size(); ++i)
    {
        const auto& coordinates = this->m_points[i];
        std::copy(coordinates.begin(), coordinates.end(), output);
        outputFile << this->m_points.cluster(i) << std::endl;
    }
    
    outputFile.close();
    std::cout << "[OK]" << std::endl;
}

Rules HPDBSCAN::localDBSCAN(const Space& space, const float epsilon, const size_t minPoints)
{
    const float EPS2 = std::pow(epsilon, 2);
    const CellIndex& cellIdx = space.cellIndex();
    
    std::cout << "Scanning Locally... " << std::endl; 
    // unpack keys
    std::vector<size_t> keys;
    keys.reserve(cellIdx.size());
    for (auto iter = cellIdx.begin(); iter != cellIdx.end(); ++iter)
    {
        keys.push_back(iter->first);
    }
    
    Rules rules;

    // local dbscan
    #pragma omp parallel for schedule(dynamic) reduction(merge: rules)
    for (size_t i = 0; i < keys.size(); ++i)
    {
        const auto& cell = keys[i];
        const auto& pointLocator = cellIdx.find(cell)->second;
        const std::vector<size_t> neighborPoints = space.getNeighbors(cell);

        for (size_t point = pointLocator.first; point < pointLocator.first + pointLocator.second; ++point)
        {
            std::vector<size_t> minPointsArea;
            ssize_t clusterId = space.regionQuery(point, cell, neighborPoints, EPS2, minPointsArea);
            
            if (minPointsArea.size() >= minPoints)
            {
                this->m_points.cluster(point, clusterId, true);
                
                for (size_t other : minPointsArea)
                {
                    ssize_t otherClusterId = this->m_points.cluster(other);
                    if (this->m_points.corePoint(other))
                    {
                        auto minmax = std::minmax(otherClusterId, clusterId);
                        rules.update(minmax.second, minmax.first);
                    }
                    this->m_points.cluster(other, clusterId, false);
                }
            }
            else if (this->m_points.cluster(point) == NOT_VISITED)
            {
                this->m_points.cluster(point, NOISE, false);
            }
        }
    }
    
    return rules;
}

void HPDBSCAN::summarize() const
{
    std::unordered_set<size_t> clusters;
    size_t clusterPoints = 0L;
    size_t noisePoints   = 0L;
    size_t corePoints    = 0L;

    for (size_t i = 0; i < this->m_points.size(); ++i)
    {
        size_t cluster = this->m_points.cluster(i);
        clusters.insert(cluster);
        
        if (cluster == 0)
        {
            ++noisePoints;
        }
        else
        {
            ++clusterPoints;
        }
        if (this->m_points.corePoint(i))
        {
            ++corePoints;
        }
    }
    std::cout << "\t" << (clusters.size() - (noisePoints > 0)) << "\tClusters" << std::endl
              << "\t" << clusterPoints << "\tCluster Points" << std::endl
              << "\t" << noisePoints << "\tNoise Points" << std::endl
              << "\t" << corePoints << "\tCore Points" << std::endl;
}

/**
 * Constructors
 */
HPDBSCAN::HPDBSCAN(const std::string &filename) : 
    m_points(readFile(filename)),
    m_array(false)
{}


HPDBSCAN::HPDBSCAN(float* data, size_t size, size_t dimensions) : 
    m_points(readArray(data, size, dimensions)),
    m_array(true)
{}

/**
 * Operations
 */
void HPDBSCAN::scan(float epsilon, size_t minPoints)
{
    Space space(this->m_points, epsilon);
    Rules rules = this->localDBSCAN(space, epsilon, minPoints);
    this->applyRules(rules);
    if (REORDER)
    {
        this->m_points.reorder(ceil(log10(this->m_points.size())));
    }
    this->summarize();
}

HPDBSCAN::~HPDBSCAN()
{
    if (this->m_array)
    {
        for (auto& point : this->m_points)
        {
            release(point);
        }
    }
}
