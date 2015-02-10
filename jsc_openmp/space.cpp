#include "space.h"
#include "util.h"

#include <fstream>
#include <limits>
#include <math.h>
#include <numeric>
#include <omp.h>

void mergeCells(CellCounter& omp_in, CellCounter& omp_out)
{
    for(auto pair: omp_in)
    {
        omp_out[pair.first] += pair.second;
    }
}

void vectorMin(std::vector<float>& omp_in, std::vector<float>& omp_out)
{
    size_t index = -1;
    for (auto& coordinate : omp_out)
    {
        coordinate = std::min(coordinate, omp_in[++index]);
    }
}

void vectorMax(std::vector<float>& omp_in, std::vector<float>& omp_out)
{
    size_t index = -1;
    for (auto& coordinate : omp_out)
    {
        coordinate = std::max(coordinate, omp_in[++index]);
    }
}

#pragma omp declare reduction(mergeCells: CellCounter: mergeCells(omp_in, omp_out)) initializer(omp_priv(CellCounter()))
#pragma omp declare reduction(vectorMax: std::vector<float>: vectorMax(omp_in, omp_out)) initializer(omp_priv(omp_orig))
#pragma omp declare reduction(vectorMin: std::vector<float>: vectorMin(omp_in, omp_out)) initializer(omp_priv(omp_orig))

Space::Space(Pointz &points, float epsilon) :
    m_cells(std::vector<size_t>(points.dimensions(), 0)),
    m_maximum(std::vector<float>(points.dimensions(), -std::numeric_limits<float>::max())),
    m_minimum(std::vector<float>(points.dimensions(),  std::numeric_limits<float>::max())),
    m_points(points),
    m_total(1)
{    
    std::cout << "Calculating Cell Space..." << std::endl;
    
    this->computeDimensions(epsilon);
    this->computeCells(epsilon);
    this->m_points.sortByCell(ceil(log10(this->m_total)));
}

/**
 * Initialization
 */
void Space::computeCells(float epsilon)
{
    CellCounter cellCounter;

    std::cout << "\tComputing Cells... " << std::flush;   
    #pragma omp parallel for reduction(mergeCells: cellCounter)
    for (size_t i = 0; i < this->m_points.size(); ++i)
    {
        size_t cell    = 0;
        size_t cellAcc = 1;

        for (size_t d = 0; d < this->m_points.dimensions(); ++d)
        {
            const float minimum = this->m_minimum[d];
            const float point   = this->m_points[i][d];

            size_t dim_index = (size_t) floor((point - minimum) / epsilon);
            cell    += dim_index * cellAcc;
            cellAcc *= this->m_cells[d];
        }
        
        this->m_points.cell(i, cell);
        cellCounter[cell] += 1;
    }
    
    const size_t numThreads = omp_get_max_threads();
    size_t total            = 0;
    size_t accumulator      = 0;
    
    for (auto& cell : cellCounter)
    {   
        auto& index  = m_cellIndex[cell.first];
        index.first  = accumulator;
        index.second = cell.second;
        accumulator += cell.second;
    }
    
    std::cout << "\t[OK]" << std::endl;
}

void Space::computeDimensions(float epsilon)
{
    const size_t dimensions = this->m_points.dimensions();
    auto& maximum = this->m_maximum;
    auto& minimum = this->m_minimum;
    
    std::cout << "\tComputing Dimensions... " << std::flush;
    #pragma omp parallel for reduction(vectorMin: minimum) reduction(vectorMax: maximum)
    for (size_t iter = 0; iter < this->m_points.size(); ++iter)
    {
        const auto& point = this->m_points[iter];
        for (size_t d = 0; d < dimensions; ++d)
        {
            const float& coordinate = point[d];
            minimum[d] = std::min(minimum[d], coordinate);
            maximum[d] = std::max(maximum[d], coordinate);
        }
    }

    for (size_t d = 0; d < this->m_cells.size(); ++d)
    {
        size_t cells     = (size_t) ceil((this->m_maximum[d] - this->m_minimum[d]) / epsilon);
        this->m_cells[d] = cells;
        this->m_total   *= cells;
    }
    
    std::cout << "[OK]" << std::endl;
}

/**
 * Operations
 */
std::vector<size_t> Space::getNeighbors(const size_t cellId) const
{    
    const CellIndex& cellIdx = this->m_cellIndex;
    
    std::vector<size_t> neighborCells;
    neighborCells.reserve(std::pow(3, m_points.dimensions()));
    neighborCells.push_back(cellId);

    size_t lowerSpace     = 1;
    size_t currentSpace   = 1;
    size_t numberOfPoints = cellIdx.find(cellId)->second.second;
    
    // here be dragons!
    for (size_t d = 0; d < this->m_points.dimensions(); ++d)
    {
        currentSpace *= this->m_cells[d];
        
        for (size_t i = 0, end = neighborCells.size(); i < end; ++i)
        {
            const size_t current = neighborCells[i];
            // check "left" neighbor - a.k.a the cell in the current dimension that has a lower number
            const long int left = current - lowerSpace;
            auto found    = cellIdx.find(left);
            if (current % currentSpace >= lowerSpace)
            {
                neighborCells.push_back(left);
                numberOfPoints += found != cellIdx.end() ? found->second.second : 0;
            }

            // check "right" neighbor - a.k.a the cell in the current dimension that has a higher number
            const long int right = current + lowerSpace;
            found = cellIdx.find(right);
            if (current % currentSpace < currentSpace - lowerSpace)
            {
                neighborCells.push_back(right);
                numberOfPoints += found != cellIdx.end() ? found->second.second : 0;
            }
        }
        
        lowerSpace = currentSpace;
    }

    std::vector<size_t> neighborPoints;
    neighborPoints.reserve(numberOfPoints);
    
    for (size_t neighborCell : neighborCells)
    {
        const auto found = cellIdx.find(neighborCell);
        if (found == cellIdx.end())
        {
            continue;
        }
        
        const std::pair<size_t, size_t>& locator = found->second;
        neighborPoints.resize(neighborPoints.size() + locator.second);
        auto end = neighborPoints.end();
        std::iota(end - locator.second, end, locator.first);
    }

    return neighborPoints;
}

size_t Space::regionQuery(const size_t pointIndex, const size_t cell, const std::vector<size_t>& neighborPoints, const float EPS2, std::vector<size_t>& minPointsArea) const
{
    const Point& point = this->m_points[pointIndex];
    size_t clusterId   = pointIndex + 1; // this MUST be a positive number so that atomicMin will result in correct result with set corePoint bit

    for (size_t neighbor: neighborPoints)
    {
        float offset            = 0.0f;
        const Point& otherPoint = this->m_points[neighbor];

        for (size_t d = 0; d < this->m_points.dimensions(); ++d)
        {
            offset += std::pow(otherPoint[d] - point[d], 2);
        }
        if (offset <= EPS2)
        {
            minPointsArea.push_back(neighbor);  
            size_t neighborCluster = this->m_points.cluster(neighbor);
            if (neighborCluster != NOT_VISITED && this->m_points.corePoint(neighbor))
            {
                clusterId = std::min(clusterId, neighborCluster);
            }
        }
    }

    return clusterId;
}

/**
 * Output 
 */
std::ostream& operator<<(std::ostream& os, const Space& space)
{
    os << "Space: "    << std::endl
       << "\tMin:\t"   << space.min()
       << "\tMax:\t"   << space.max()
       << "\tCell:\t"  << space.cells()
       << "\tTotal:\t" << space.total() << std::endl;
    
    return os;
}
