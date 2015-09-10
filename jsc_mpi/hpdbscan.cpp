#include "hpdbscan.h"

#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <vector>
#include <unordered_set>
#include <set>
#include <iterator>
#include <omp.h>

/**
 * Constructors
 */
HPDBSCAN::HPDBSCAN(const std::string& filename) :
    m_points(filename),
    m_filename(filename)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &this->m_mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->m_mpiSize);
}

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
            found   = rules.rule(found);
        }
        this->m_points.overrideCluster(i, cluster, core);
    }
}

void HPDBSCAN::distributeRules(Rules& rules)
{
    const int numberOfRules = (const int) rules.size();
    
    int* counts  = new int[this->m_mpiSize];
    int* displs  = new int[this->m_mpiSize];
    int* rCounts = new int[this->m_mpiSize];
    int* rDispls = new int[this->m_mpiSize];
    
    for (int i = 0; i < this->m_mpiSize; ++i)
    {
        counts[i] = 2 * numberOfRules;
        displs[i] = 0;
    }
    MPI_Alltoall(counts, 1, MPI_INT, rCounts, 1, MPI_INT, MPI_COMM_WORLD);
    
    size_t total = 0;
    for (int i = 0; i < this->m_mpiSize; ++i)
    {
        rDispls[i]  = total;
        total      += rCounts[i];
    }
    
    Cluster* sendRules = new Cluster[counts[this->m_mpiRank]];
    size_t   index     = 0;
    for (const auto& rule : rules)
    {
        sendRules[index]     = rule.first;
        sendRules[index + 1] = rule.second;
        index               += 2;
    }
    Cluster* recvRules = new Cluster[total];
    MPI_Alltoallv(sendRules, counts, displs, MPI_LONG, recvRules, rCounts, rDispls, MPI_LONG, MPI_COMM_WORLD);
    
    for (size_t i = 0; i < total; i += 2)
    {
        rules.update(recvRules[i], recvRules[i + 1]);
    }
    
    delete[] recvRules;
    delete[] sendRules;
    delete[] rDispls;
    delete[] rCounts;
    delete[] displs;
    delete[] counts;
}

Rules HPDBSCAN::localDBSCAN(const Space& space, const float epsilon, const size_t minPoints)
{
    const float      EPS2    = std::pow(epsilon, 2);
    
    const size_t lower = space.lowerHaloBound();
    const size_t upper = space.upperHaloBound();
    
    Rules rules;
    // local dbscan
    
    size_t cell = NOT_VISITED;
    std::vector<size_t> neighborPoints;
    
    
    #pragma omp parallel for schedule(dynamic, 500) private(neighborPoints) firstprivate(cell) reduction(merge: rules)
    for (size_t point = lower; point < upper; ++point)
    {
        size_t pointCell = this->m_points.cell(point);
        if (pointCell != cell)
        {
            neighborPoints = space.getNeighbors(pointCell);
            cell = pointCell;
        }
        std::vector<size_t> minPointsArea;
        ssize_t clusterId = NOISE;
        if(neighborPoints.size() >= minPoints)
        {
            clusterId =space.regionQuery(point, neighborPoints, EPS2, minPointsArea);
        }

        if (minPointsArea.size() >= minPoints)
        {
            this->m_points.cluster(point, clusterId, true);

            for (size_t other : minPointsArea)
            {
                ssize_t otherClusterId = this->m_points.cluster(other);
                if (this->m_points.corePoint(other))
                {
                    const std::pair<Cluster, Cluster> minmax = std::minmax(otherClusterId, clusterId);
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
    
    
    return rules;
}

/**
 * Operations
 */
void HPDBSCAN::scan(float epsilon, size_t minPoints)
{
    
    this->m_points.resetClusters();
    Space space(this->m_points, epsilon);
    
    double start = omp_get_wtime();
    if (!this->m_mpiRank)
    {
        std::cout << "DBSCAN... " << std::endl
                  << "\tLocal Scan... " << std::flush;
    }
    Rules rules = this->localDBSCAN(space, epsilon, minPoints);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (!this->m_mpiRank)
    {
        std::cout << "\t\t[OK] in " << omp_get_wtime() - start << std::endl
                  << "\tMerging Neighbors... " << std::flush;
    }
    start = omp_get_wtime();
    
    if(this->m_mpiSize > 1)
    {
        this->m_points.mergeNeighborHalos(space.lowerHaloBound(), space.upperHaloBound(), rules, space.computeCuts());
        this->distributeRules(rules);
    }
    
    if (!this->m_mpiRank)
    {
        std::cout << "\t[OK] in " << omp_get_wtime() - start << std::endl
                  << "\tAdjust Labels ... " << std::flush;
    }
    
    start = omp_get_wtime();
    this->applyRules(rules);   
    if (!this->m_mpiRank)
    {
        std::cout << "\t[OK] in " << omp_get_wtime() - start << std::endl
                  << "\tRec. Init. Order ... " << std::flush;
    }
        
    start = omp_get_wtime();
    if(this->m_mpiSize > 1)
    {    
        this->m_points.recoverInitialOrder(space.lowerHaloBound(), space.upperHaloBound());
    }
    else
    {
        this->m_points.sortByOrder(ceil(log10(this->m_points.size())), space.lowerHaloBound(),space.upperHaloBound());
    }
    if (!this->m_mpiRank)
    {
        std::cout << "\t[OK] in " << omp_get_wtime() - start << std::endl
                  << "\tWriting File ... " << std::flush;
    }
    start = omp_get_wtime();
    this->m_points.writeClusterToFile(this->m_filename);
    if (!this->m_mpiRank)
    {
        std::cout << "\t[OK] in " << omp_get_wtime() - start << std::endl;
    }
    this->summarize();
}

/**
 * Output
 */
void HPDBSCAN::summarize() const
{
    std::unordered_set<size_t> clusters;
    size_t clusterPoints = 0L;
    size_t noisePoints   = 0L;
    size_t corePoints    = 0L;
    
    
    for (size_t i = 0; i < this->m_points.size(); ++i)
    {
        const size_t cluster = this->m_points.cluster(i);
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
    int      setSize     = (int) clusters.size();
    int*     setSizes    = NULL;
    int*     setDispls   = NULL;
    Cluster* allClusters = NULL;
    if (!this->m_mpiRank)
    {
        std::cout << "Result... " << std::endl;
        setSizes  = new int[this->m_mpiSize];
        setDispls = new int[this->m_mpiSize];
    }
    MPI_Gather(&setSize, 1, MPI_INT, setSizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    Cluster* clusterBuffer = new Cluster[setSize];
    std::copy(clusters.begin(), clusters.end(), clusterBuffer);
    
    size_t total = 0;
    if (!this->m_mpiRank)
    {
        for (int i = 0; i < this->m_mpiSize; ++i)
        {
            setDispls[i] = total;
            total += setSizes[i];
        }
        allClusters = new Cluster[total];
    }
    
    MPI_Gatherv(clusterBuffer, setSize, MPI_LONG, allClusters, setSizes, setDispls, MPI_LONG, 0, MPI_COMM_WORLD);
    size_t metrics[] = {clusterPoints, noisePoints, corePoints};
    MPI_Reduce(!this->m_mpiRank ? MPI_IN_PLACE : metrics, metrics, sizeof(metrics) / sizeof(size_t), MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (!this->m_mpiRank)
    {
        clusters.insert(allClusters, allClusters + total);
        std::cout << "\t" << (metrics[1] ? clusters.size() - 1 : clusters.size()) << "\tClusters" << std::endl
                  << "\t" << metrics[0] << "\tCluster Points" << std::endl
                  << "\t" << metrics[1] << "\tNoise Points" << std::endl
                  << "\t" << metrics[2] << "\tCore Points" << std::endl;
        
        delete[] allClusters;
        delete[] setDispls;
        delete[] setSizes;
    }
    
    delete[] clusterBuffer;
}
