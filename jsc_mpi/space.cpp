#include "constants.h"
#include "space.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
#include <iomanip>
#include <math.h>
#include <mpi.h>
#include <numeric>
#include <omp.h>

void mergeCells(CellCounter& omp_in, CellCounter& omp_out)
{
    for(auto pair: omp_in)
    {
        omp_out[pair.first] += pair.second;
    }
}

void vectorMin(std::vector<Coord>& omp_in, std::vector<Coord>& omp_out)
{
    size_t index = -1;
    for (auto& coordinate : omp_out)
    {
        coordinate = std::min(coordinate, omp_in[++index]);
    }
}

void vectorMax(std::vector<Coord>& omp_in, std::vector<Coord>& omp_out)
{
    size_t index = -1;
    for (auto& coordinate : omp_out)
    {
        coordinate = std::max(coordinate, omp_in[++index]);
    }
}

#pragma omp declare reduction(mergeCells: CellCounter: mergeCells(omp_in, omp_out)) initializer(omp_priv(CellCounter()))
#pragma omp declare reduction(vectorMax: std::vector<Coord>: vectorMax(omp_in, omp_out)) initializer(omp_priv(omp_orig))
#pragma omp declare reduction(vectorMin: std::vector<Coord>: vectorMin(omp_in, omp_out)) initializer(omp_priv(omp_orig))

Space::Space(Pointz& points, float epsilon) :
    m_points(points),
    m_total(1),
    m_pointOffset(0),
    m_halo(1),
    m_lastCell(0),
    m_cells(std::vector<size_t>(points.dimensions(),  0)),
    m_maximum(std::vector<Coord>(points.dimensions(), -std::numeric_limits<Coord>::max())),
    m_minimum(std::vector<Coord>(points.dimensions(),  std::numeric_limits<Coord>::max())),
    m_swapDims(std::vector<size_t>(points.dimensions()))
{
    MPI_Comm_rank(MPI_COMM_WORLD, &this->m_mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->m_mpiSize);
    
    std::iota(this->m_swapDims.begin(), this->m_swapDims.end(), 0);
    
    if (!this->m_mpiRank)
    {
        std::cout << std::fixed << std::setw( 11 ) << std::setprecision( 6 ) 
          << std::setfill( ' ' );
        std::cout << "Calculating Cell Space..." << std::endl;
    }
    
    // Dimensions
    if (!this->m_mpiRank)
    {
        std::cout << "\tComputing Dimensions... " << std::flush;
    }
    double start = omp_get_wtime();
    this->computeDimensions(epsilon);
    if (!this->m_mpiRank)
    {
        std::cout << "[OK] in " << omp_get_wtime() - start << std::endl;
    }
    
    // Cell Business
    if (!this->m_mpiRank)
    {
        std::cout << "\tComputing Cells... " << std::flush;
        start = omp_get_wtime();   
    }
    CellCounter cellCounter = this->computeCells(epsilon);
    this->computeIndex(cellCounter);   
        this->m_cellBounds[0][0] = 0;
        this->m_cellBounds[0][1] = 0;
        this->m_cellBounds[0][2] = this->m_lastCell;
        this->m_cellBounds[0][2] = this->m_lastCell;
    if (!this->m_mpiRank)
    {
        std::cout << "\t[OK] in " << omp_get_wtime() - start << std::endl;
    }
    
    // Sorting
    if (!this->m_mpiRank)
    {
        std::cout << "\tSorting Points... " << std::flush;
        start = omp_get_wtime();
    }
    
    this->m_points.sortByCell(this->m_cellIndex);
    if (!this->m_mpiRank)
    {
        std::cout << "\t[OK] in " << omp_get_wtime() - start << std::endl;
    }
    
    // Redistribute
    if (!this->m_mpiRank)
    {
        std::cout << "\tDistributing Points... " << std::flush;
        start = omp_get_wtime();
    }
    if(this->m_mpiSize > 1)
    {
        this->computeGlobalCounter(cellCounter);
        this->splitAndRedistribute();
        cellCounter = this->computeCells(epsilon);
        this->m_cellIndex.clear();
        this->computeIndex(cellCounter);
        this->computePointOffset();
        this->m_points.sortByCell(this->m_cellIndex);
    } 
    if (!this->m_mpiRank)
    {
        std::cout << "\t[OK] in " << omp_get_wtime() - start << std::endl;
    }
}

/**
 * Initialization
 */
void Space::computeBounds(const CellCounter& counter)
{    
    size_t totalScore = 0;
    size_t i          = 0;
    std::vector<size_t> scores(counter.size(), 0);
    
    // compute score
    for (auto& pair : counter)
    {
        const size_t cell  = pair.first;
        const size_t score = this->computeScore(cell, counter);
        scores[i++]        = score;
        totalScore        += score;
    }
    
    // compute actual bounds
    const size_t chunkedScore = totalScore / this->m_mpiSize + 1;
    size_t rank               = 0;
    size_t start              = 0;
    size_t acc                = 0;
    size_t lowerSplit         = 0;
    size_t offScore           = 0;
    
    auto cells = counter.begin();
    for (size_t i = 0; i < scores.size(); ++ i)
    {
        const auto& pair   = cells++;
        const size_t score = scores[i];
        const size_t cell  = pair->first;
        acc               += score;
        
        while (acc > chunkedScore)
        {
            size_t pointsSplit = (acc - chunkedScore) / (score / pair->second);
            offScore = pointsSplit * score / pair->second;
            m_computeOffset[rank][0] = lowerSplit;
            m_computeOffset[rank][1] = pointsSplit;
            lowerSplit          = pointsSplit;
            const size_t offset = (start % this->m_halo) + this->m_halo;
            auto& bound         = this->m_cellBounds[rank];
            bound[0]            = offset > start ? 0 : start - offset;
            bound[1]            = start;
            bound[2]            = cell + 1;
            bound[3]            = std::min((bound[2] / this->m_halo) * this->m_halo + (this->m_halo * 2), this->m_lastCell);
            start               = bound[2];
            acc                 = offScore;
            ++rank;
        }
        
        if ((int) rank == this->m_mpiSize - 1  || i == counter.size() - 1)
        {
            auto& bound         = this->m_cellBounds[rank];
            m_computeOffset[rank][0] = lowerSplit;
            m_computeOffset[rank][1] = 0;
            const size_t offset = (start % this->m_halo) + this->m_halo;
            bound[0]            = offset > start ? 0 : start - offset;
            bound[1]            = start;
            bound[2]            = this->m_lastCell;
            bound[3]            = this->m_lastCell;
            break;
        }
    }
}

CellCounter Space::computeCells(float epsilon)
{
    CellCounter cellCounter;
    #pragma omp parallel for reduction(mergeCells: cellCounter)
    for (size_t i = 0; i < this->m_points.size(); ++i)
    {
        size_t cell    = 0;
        size_t cellAcc = 1;

        for (size_t d : this->m_swapDims)
        {
            const Coord minimum = this->m_minimum[d];
            const Coord point   = this->m_points[i][d];

            size_t dim_index = (size_t) floor((point - minimum) / epsilon);
            cell            += dim_index * cellAcc;
            cellAcc         *= this->m_cells[d];
        }
        
        this->m_points.cell(i, cell);
        cellCounter[cell] += 1;
    } 
    return cellCounter;
}
    
void Space::computeGlobalCounter(CellCounter& cellCounter)
{
    // fetch cell counter over network
    int counts[this->m_mpiSize];
    int displs[this->m_mpiSize];
    int rCounts[this->m_mpiSize];
    int rDispls[this->m_mpiSize];
    
    for (int i = 0; i < this->m_mpiSize; ++i)
    {
        counts[i] = cellCounter.size() * 2;
        displs[i] = 0;
    }
    
    MPI_Alltoall(counts, 1, MPI_INT, rCounts, 1, MPI_INT, MPI_COMM_WORLD);
    
    size_t totalCount = 0;
    for (int i = 0; i < this->m_mpiSize; ++i)
    {
        rDispls[i]  = totalCount;
        totalCount += rCounts[i];
    }
    
    size_t* sendBuffer    = new size_t[cellCounter.size() * 2];
    size_t* counterBuffer = new size_t[totalCount];
    size_t i              = 0;
    for (const auto& item : cellCounter)
    {
        sendBuffer[i++] = item.first;
        sendBuffer[i++] = item.second;
    }
    
    MPI_Alltoallv(sendBuffer, counts, displs, MPI_UNSIGNED_LONG, counterBuffer, rCounts, rDispls, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    CellCounter counter;
    for (size_t i = 0; i < totalCount; i+=2)
    {
        counter[counterBuffer[i]] += counterBuffer[i + 1];
    }
    delete[] sendBuffer;
    delete[] counterBuffer;
    
    this->m_lastCell = counter.rbegin()->first + 1;
    // compute global cell bounds
    this->computeBounds(counter);
}

void Space::computeDimensions(float epsilon)
{
    const size_t dimensions = this->m_points.dimensions();
    auto& maximum = this->m_maximum;
    auto& minimum = this->m_minimum;
    
    #pragma omp parallel for reduction(vectorMin: minimum) reduction(vectorMax: maximum)
    for (size_t iter = 0; iter < this->m_points.size(); ++iter)
    {
        const auto& point = this->m_points[iter];
        for (size_t d = 0; d < dimensions; ++d)
        {
            const Coord& coordinate = point[d];
            minimum[d] = std::min(minimum[d], coordinate);
            maximum[d] = std::max(maximum[d], coordinate);
        }
    }
    
    // Get the minimum/maximum of the other nodes
    Coord* buffer = new Coord[dimensions];
    MPI_Allreduce(this->m_minimum.data(), buffer, dimensions, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    std::copy(buffer, buffer + dimensions, this->m_minimum.begin());
    MPI_Allreduce(this->m_maximum.data(), buffer, dimensions, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    std::copy(buffer, buffer + dimensions, this->m_maximum.begin());

    // compute cell count
    for (size_t d = 0; d < this->m_cells.size(); ++d)
    {
        size_t cells     = (size_t) ceil((this->m_maximum[d] - this->m_minimum[d]) / epsilon) + 1;
        this->m_cells[d] = cells;
        this->m_total   *= cells;
    }
    
    delete[] buffer;
    this->swapDimensions();
    this->m_halo = this->m_total / this->m_cells[this->m_swapDims[dimensions-1]];
    this->m_cellBounds.resize(this->m_mpiSize);
    this->m_computeOffset.resize(this->m_mpiSize);
    this->m_lastCell = this->m_total;
}

void Space::computeIndex(CellCounter& cellCounter)
{
    // setup index
    size_t accumulator = 0;
    
    for (auto& cell : cellCounter)
    {
        auto& index  = m_cellIndex[cell.first];
        index.first  = accumulator;
        index.second = cell.second;
        accumulator += cell.second;
    }
    m_cellIndex[this->m_lastCell].first  = m_points.size();
    m_cellIndex[this->m_lastCell].second = 0;
}

void Space::computePointOffset()
{
    //Calculate number of points
    const size_t numberOfPoints = upperHaloBound() - lowerHaloBound(); 
    size_t pointCounts[this->m_mpiSize];
    
    std::fill(pointCounts, pointCounts + this->m_mpiSize, numberOfPoints);
    MPI_Alltoall(MPI_IN_PLACE, 1, MPI_UNSIGNED_LONG, pointCounts, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    this->m_pointOffset = 0;
    for (int i = 0; i < this->m_mpiRank; ++i)
    {
        this->m_pointOffset += pointCounts[i];
    }
}

void Space::splitAndRedistribute()
{
    const size_t dimensions = this->m_points.dimensions();
    
    int* counts  = new int[this->m_mpiSize];
    int* displs  = new int[this->m_mpiSize];
    int* rCounts = new int[this->m_mpiSize];
    int* rDispls = new int[this->m_mpiSize];
    
    //Calculate send count and displacement
    for (size_t i = 0; i < this->m_cellBounds.size(); ++i)
    {
        const auto& bound = this->m_cellBounds[i];
        size_t lower = this->m_cellIndex.lower_bound(bound[0])->second.first;
        size_t upper = this->m_cellIndex.lower_bound(bound[3])->second.first;
        displs[i] = lower * dimensions;
        counts[i] = upper * dimensions  - displs[i];
            
    }
    
    MPI_Alltoall(counts, 1, MPI_INT, rCounts, 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < this->m_mpiSize; ++i)
    {
        rDispls[i] = (i == 0) ? 0 : (rDispls[i - 1] + rCounts[i - 1]);
    }
    this->m_points.redistributePoints(counts, displs, rCounts, rDispls);
    
    delete[] rDispls;
    delete[] rCounts;
    delete[] displs;
    delete[] counts;
}

void Space::swapDimensions()
{
    const auto& dims = this->m_cells;
    std::sort(this->m_swapDims.begin(), this->m_swapDims.end(), [dims](size_t a, size_t b)
    {
        return dims[a] < dims[b];
    });   
}

/**
 * Operations
 */
Cuts Space::computeCuts() const
{
    Cuts cuts(this->m_mpiSize, std::pair<size_t, size_t>(0,0));
    
    for (int i = 0; i < this->m_mpiSize; ++i)
    {   
        if (i == this->m_mpiRank)
        {
            continue;
        }
        auto bound = this->m_cellBounds[i];
        auto cbound = this->m_computeOffset[i];
        
        // lower bound
        const auto& lowerCut = this->m_cellIndex.lower_bound(bound[1]);
        cuts[i].first        = lowerCut->second.first;
        if (lowerCut->first != this->m_lastCell ||  this->m_cellIndex.find(m_lastCell - 1) != this->m_cellIndex.end())
        {
            cuts[i].first   = cuts[i].first < cbound[0] ? 0 : cuts[i].first - cbound[0];
        }
        // upper bound
        const auto& upperCut = this->m_cellIndex.lower_bound(bound[2]);
        cuts[i].second       =  upperCut->second.first;        
        if (upperCut->first != this->m_lastCell || this->m_cellIndex.find(m_lastCell - 1) != this->m_cellIndex.end())
        {
            cuts[i].second   = cuts[i].second < cbound[1] ? 0 : cuts[i].second - cbound[1];
        }
    }
    return cuts;
}

size_t Space::computeScore(const size_t cellId, const CellCounter& cellCounter) const
{    
    std::vector<size_t> neighborCells;
    neighborCells.reserve(std::pow(3, m_points.dimensions()));
    neighborCells.push_back(cellId);

    size_t lowerSpace     = 1;
    size_t currentSpace   = 1;
    size_t pointsInCell   = cellCounter.find(cellId)->second;
    size_t numberOfPoints = pointsInCell;
    
    // here be dragons!
    for (size_t d : this->m_swapDims)
    {
        currentSpace *= this->m_cells[d];
        
        for (size_t i = 0, end = neighborCells.size(); i < end; ++i)
        {
            const size_t current = neighborCells[i];
            const long int left = current - lowerSpace;
            if (current % currentSpace >= lowerSpace)
            {
                const auto& locator = cellCounter.find(left);
                numberOfPoints += locator != cellCounter.end() ? locator->second : 0;
            }
            const long int right = current + lowerSpace;
            if (current % currentSpace < currentSpace - lowerSpace)
            {
                const auto& locator = cellCounter.find(right);
                numberOfPoints += locator != cellCounter.end() ? locator->second : 0;
            }
        }
        lowerSpace = currentSpace;
    }
    
    return pointsInCell * numberOfPoints;
}

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
    for (size_t d : this->m_swapDims)
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

size_t Space::regionQuery(const size_t pointIndex, const std::vector<size_t>& neighborPoints, const float EPS2, std::vector<size_t>& minPointsArea) const
{
    const Coord* point = this->m_points[pointIndex];
    // this MUST be a positive number so that atomicMin will result in correct result with set corePoint bit
    size_t clusterId   = this->m_pointOffset + pointIndex + 1;
    
    for (size_t neighbor: neighborPoints)
    {
        float offset            = 0.0f;
        const Coord* otherPoint = this->m_points[neighbor];

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

size_t Space::lowerHaloBound() const
{
    size_t lower = this->m_cellIndex.lower_bound(this->m_cellBounds[this->m_mpiRank][1])->second.first;  
    return lower - this->m_computeOffset[this->m_mpiRank][0];
}

size_t Space::upperHaloBound() const
{
    size_t upper = this->m_cellIndex.lower_bound(this->m_cellBounds[this->m_mpiRank][2])->second.first;
    return upper - this->m_computeOffset[this->m_mpiRank][1];
}
