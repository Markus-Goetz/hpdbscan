#include "points.h"
#include "space.h"

#include <algorithm>
#include <limits>
#include <omp.h>

const unsigned short DIGITS   = 10; /* digit count, radix sort buckets */
const unsigned int   POWERS[] = {
    1, 
    10, 
    100, 
    1000, 
    10000, 
    100000, 
    1000000, 
    10000000, 
    100000000, 
    1000000000
};

inline const unsigned int power(size_t exp)
{
    return POWERS[exp];
}

Pointz::Pointz(size_t size, size_t dimensions, bool array) :
    m_dimensions(dimensions),
    m_cells(std::vector<size_t>(size, 0)),
    m_points(std::vector<Point>(size, Point(array ? 0 : dimensions, 0.0f))),
    m_size(size)
{
    this->m_clusters.resize(size);
}

/**
 * Operations
 */
void Pointz::sortByCell(size_t maxDigits)
{
    std::cout << "\tSorting Points... " << std::flush;
    if (maxDigits > DIGITS)
    {
        throw std::invalid_argument("epsilon is too small relative to data space.");
    }
    
    // reset cluster ids
    std::fill(this->m_clusters.begin(), this->m_clusters.end(), std::numeric_limits<ssize_t>::max());
    
    std::vector<std::vector<size_t> > buckets(maxDigits, std::vector<size_t>(DIGITS));
    std::vector<size_t> cellBuffer(this->size(), 0);
    std::vector<Point> pointsBuffer(this->size(), Point(this->dimensions(),0));
    std::vector<size_t> reorderBuffer;
    if (REORDER)
    {
        this->m_reorder = std::vector<size_t>(this->size());
        reorderBuffer = std::vector<size_t>(this->size());
        std::iota (this->m_reorder.begin(), this->m_reorder.end(), 0);
    }
    
    for (size_t j = 0; j < maxDigits; ++j)
    {
        for(size_t f = 0; f < DIGITS; ++f)
        {
            buckets[j][f] = 0;
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->size(); ++i)
    {
        for (size_t j = 0; j < maxDigits; ++j)
        {
            const size_t base  = power(j);
            const size_t digit = this->m_cells[i] / base % DIGITS;
            #pragma omp atomic
            ++buckets[j][digit];
        }
    }

    #pragma omp parallel for shared(buckets)
    for (size_t j = 0; j < maxDigits; ++j)
    {
        for (size_t f = 1; f < DIGITS; ++f)
        {
            buckets[j][f] += buckets[j][f-1];
        }
    }

    for (size_t j = 0; j < maxDigits; ++j)
    {
        const size_t base = power(j);
        for (size_t i = this->size() - 1; i < std::numeric_limits<size_t>::max(); --i)
        {
            size_t unit  = this->m_cells[i] / base % DIGITS;
            size_t pos = --buckets[j][unit];

            for (size_t d = 0; d < this->dimensions(); ++d)
            {
                pointsBuffer[pos][d] = this->m_points[i][d];
            }
            cellBuffer[pos] = this->m_cells[i];
            if (REORDER)
            {
                reorderBuffer[pos] = this->m_reorder[i];
            }
        }
        std::copy(pointsBuffer.begin(), pointsBuffer.end(), this->m_points.begin());
        std::copy(cellBuffer.begin(), cellBuffer.end(), this->m_cells.begin());
        if (REORDER)
        {
            std::copy(reorderBuffer.begin(), reorderBuffer.end(), this->m_reorder.begin());
        }
    }
    
    std::cout << "\t[OK]" << std::endl;
}

void Pointz::reorder(size_t maxDigits)
{    
    std::cout << "\tSorting Points Back... " << std::flush;   
    std::vector<std::vector<size_t> > buckets(maxDigits, std::vector<size_t>(DIGITS));
    std::vector<Point> pointsBuffer(this->size(), Point(this->dimensions(),0));
    std::vector<size_t> cellBuffer(this->size(), 0);
    std::vector<size_t> clusterBuffer(this->size(), 0);
    std::vector<size_t> reorderBuffer(this->size(), 0);
    
    for (size_t j = 0; j < maxDigits; ++j)
    {
        for(size_t f = 0; f < DIGITS; ++f)
        {
            buckets[j][f] = 0;
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->size(); ++i)
    {
        for (size_t j = 0; j < maxDigits; ++j)
        {
            const size_t base  = power(j);
            const size_t digit = this->m_reorder[i] / base % DIGITS;
            #pragma omp atomic
            ++buckets[j][digit];
        }
    }

    #pragma omp parallel for shared(buckets)
    for (size_t j = 0; j < maxDigits; ++j)
    {
        for (size_t f = 1; f < DIGITS; ++f)
        {
            buckets[j][f] += buckets[j][f-1];
        }
    }

    for (size_t j = 0; j < maxDigits; ++j)
    {
        const size_t base = power(j);
        for (size_t i = this->size() - 1; i < std::numeric_limits<size_t>::max(); --i)
        {
            size_t unit  = this->m_reorder[i] / base % DIGITS;
            size_t pos = --buckets[j][unit];

            for (size_t d = 0; d < this->dimensions(); ++d)
            {
                pointsBuffer[pos][d] = this->m_points[i][d];
            }
            cellBuffer[pos] = this->m_cells[i];
            clusterBuffer[pos] = this->m_clusters[i];
            reorderBuffer[pos] = this->m_reorder[i];
        }
        std::copy(pointsBuffer.begin(), pointsBuffer.end(), this->m_points.begin());
        std::copy(cellBuffer.begin(), cellBuffer.end(), this->m_cells.begin());
        std::copy(clusterBuffer.begin(), clusterBuffer.end(), this->m_clusters.begin());
        std::copy(reorderBuffer.begin(), reorderBuffer.end(), this->m_reorder.begin());
    }
    
    std::cout << "\t[OK]" << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Pointz& points)
{
    os << "Points: " << std::endl;
    for (const auto& point : points)
    {
        os << "\t" << point << std::endl;
    }
    os << "Clusters: " << std::endl;
    for (size_t index = 0; index < points.size(); ++index)
    {
        os << "\t" << points.cluster(index) << " " << points.corePoint(index) << std::endl;
    }
    return os;
}
