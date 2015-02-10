#ifndef SPACE_H
#define SPACE_H

#include "points.h"

#include <iostream>
#include <map>
#include <vector>

typedef std::map< size_t, size_t >                    CellCounter;
typedef std::map< size_t, std::pair<size_t, size_t> > CellIndex;

class Space
{
    CellIndex           m_cellIndex;
    
    std::vector<size_t> m_cells;
    std::vector<float>  m_maximum; 
    std::vector<float>  m_minimum;
    
    Pointz&             m_points;
    size_t              m_total;
    
    /**
     * Initialization
     */
    void computeCells(float epsilon);
    void computeDimensions(float epsilon);
    
public:
    Space(Pointz& points, float epsilon);
    
    /**
     * Access 
     */
    inline const CellIndex& cellIndex() const
    {
        return this->m_cellIndex;
    }
    
    inline const std::vector<size_t>& cells() const
    {
        return this->m_cells;
    }
    
    inline size_t cells(size_t dimension) const
    {
        return this->m_cells[dimension];
    }
    
    inline const std::vector<float>& max() const
    {
        return this->m_maximum;
    }
    
    inline float max(size_t dimension) const
    { 
        return this->m_maximum[dimension];
    
    }
    
    inline const std::vector<float>& min() const
    {
        return this->m_minimum;
    }
    
    inline float min(size_t dimension) const
    {
        return this->m_minimum[dimension];
    }
    
    inline size_t total() const
    {
        return this->m_total;
    }
    
    /**
     * Operations
     */
    std::vector<size_t> getNeighbors(const size_t cellId) const;
    size_t regionQuery(const size_t pointIndex, const size_t cell, const std::vector<size_t>& neighborPoints, const float EPS2, std::vector<size_t>& minPointsArea) const;
};

/**
 * Output
 */
std::ostream& operator<<(std::ostream& os, const Space& space);

#endif // SPACE_H
