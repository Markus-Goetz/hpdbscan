#ifndef POINTS_H
#define POINTS_H

#include "util.h"

#include <limits>
#include <map>
#include <math.h>
#include <iostream>
#include <vector>

#define REORDER true

typedef std::vector<float>          Point;

const ssize_t NOT_VISITED = std::numeric_limits<ssize_t>::max();
const ssize_t NOISE       = std::numeric_limits<ssize_t>::max() - 1;

class Pointz
{
    size_t m_size;
    size_t m_dimensions;
    
    std::vector<size_t>  m_cells;
    std::vector<ssize_t> m_clusters;
    std::vector<Point>   m_points;
    std::vector<size_t>  m_reorder;

public:
    Pointz(size_t size, size_t dimensions, bool array=false);
    
    /**
     *  Iteration 
     */
    inline std::vector<Point>::iterator begin()
    {
        return this->m_points.begin();
    }
    
    inline std::vector<Point>::const_iterator begin() const
    {
        return this->m_points.begin();
    }
    
    inline std::vector<Point>::iterator end()
    {
        return this->m_points.end();
    }
    
    inline std::vector<Point>::const_iterator end() const
    {
        return this->m_points.end();
    }
    
    /**
     *  Access
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
    
    inline Point& operator[](size_t index)
    {
        return this->m_points[index];
    }
    
    inline const size_t size() const
    {
        return this->m_size;
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
    void sortByCell(size_t maxDigits);
    void reorder(size_t maxDigits);
};

/**
 * Output
 */
std::ostream& operator<<(std::ostream& os, const Pointz& points);

#endif // POINTS_H
