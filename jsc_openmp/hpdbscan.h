#ifndef HPDBSCAN_H
#define HPDBSCAN_H

#include "points.h"
#include "rules.h"
#include "space.h"

#include <string>

class HPDBSCAN
{
    Pointz m_points;
    bool m_array;
    
    /**
     * Internal Operations
     */
    void applyRules(const Rules& rules);
    Rules localDBSCAN(const Space &space, float epsilon, size_t minPoints);
    Pointz readArray(float* array, size_t size, size_t dimensions);
    Pointz readFile(const std::string& filename);
    void summarize() const;

public:
    HPDBSCAN(const std::string& filename);
    HPDBSCAN(float* data, size_t size, size_t dimensions);
    void scan(float epsilon, size_t minPoints);
    void writeFile(const std::string& outpuFilePath);
    ~HPDBSCAN();
};

#endif // HPDBSCAN_H
