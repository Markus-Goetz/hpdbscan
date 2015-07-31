#ifndef POINTS_H
#define	POINTS_H

#include "constants.h"
#include "rules.h"
#include "util.h"

#include <cmath>
#include <stddef.h>
#include <string>
#include <vector>

#define REORDER true
#define DATASET "DBSCAN"

class Pointz
{
    int      m_mpiRank;
    int      m_mpiSize;
    
    Cell*    m_cells;
    Cluster* m_clusters;
    Coord*   m_points;
    
    size_t   m_dimensions;
    size_t   m_size;
    size_t   m_totalSize;
    
    
    /**
     * Internal Operations
     */
    void mergeNeighborClusters(Cluster* halo, size_t count, size_t offset, Rules& rules);
    void readFile(const std::string& filename);

public:
    size_t* m_initialOrder;
    /**
     * Constructor
     */
    Pointz(const std::string& filename);
    
    /**
     * Access
     */
    inline const size_t cell(size_t index) const
    {
        return this->m_cells[index];
    }
    
    inline const ssize_t cluster(const size_t index) const
    {
        return std::abs(this->m_clusters[index]);
    }
    
    inline bool corePoint(const size_t index) const
    {
        return this->m_clusters[index] < 0;
    }
    
    inline const size_t dimensions() const
    {
        return this->m_dimensions;
    }
    
    inline Coord* operator[](size_t index) const
    {
        return this->m_points + index * this->m_dimensions;
    }
    
    inline const size_t size() const
    {
        return this->m_size;
    }
    inline const size_t totalSize() const
    {
        return this->m_totalSize;
    }
    
    /**
     * Modifiers
     */
    inline void cell(size_t index, size_t number)
    {
        this->m_cells[index] = number;
    }
    
    inline void cluster(const size_t index, ssize_t value, bool core)
    {
        atomicMin(&this->m_clusters[index], core ? -value : value);
    }
    
    inline void overrideCluster(const size_t index, ssize_t value, bool core)
    {
        this->m_clusters[index] = core ? -value : value;
    }
    
    /**
     * Operations
     */
    void   mergeNeighborHalos(const size_t lowerHaloBound, const size_t upperHaloBound, Rules& rules, const Cuts& cuts);
    size_t redistributePoints(int* counts, int* displs, int* rCounts, int* rDispls, bool incCluster = false);
    void   resetClusters();
    void   sortByCell(const CellIndex& index);
    void   sortByOrder(size_t maxDigits, size_t lowerBound, size_t upperBound);
    void   writeClusterToFile(const std::string& filename) const;
    void   recoverInitialOrder(size_t lowerBound, size_t upperBound);
    
    /**
     * Destructor
     */
    ~Pointz();
};

#endif	// POINTS_H
