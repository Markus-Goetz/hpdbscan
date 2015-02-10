#ifndef HPDBSCAN_H
#define	HPDBSCAN_H

#include "points.h"
#include "rules.h"
#include "space.h"

#include <stddef.h>
#include <string>

class HPDBSCAN
{
    int         m_mpiRank;
    int         m_mpiSize;
    
    Pointz      m_points;
    std::string m_filename;
    
    /**
     * Internal Operation\
     */
    void applyRules(const Rules& rules);
    void distributeRules(Rules& rules);
    Rules localDBSCAN(const Space &space, float epsilon, size_t minPoints);
    void summarize() const;
    
public:
    HPDBSCAN(const std::string& filename);
    void scan(float epsilon, size_t minPoints);
};

#endif	// HPDBSCAN_H
