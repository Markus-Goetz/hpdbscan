#include "points.h"
#include "rules.h"

#include <algorithm>
#include <atomic>
#include <hdf5.h>
#include <iostream>
#include <mpi.h>
#include <unordered_map>

/**
 * Radix-Sort helpers
 */
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

/**
 * Constructor
 */
Pointz::Pointz(const std::string& filename) :
    m_cells(NULL),
    m_clusters(NULL),
    m_points(NULL)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &this->m_mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->m_mpiSize);
    this->readFile(filename);
    
    this->m_clusters = new Cluster[this->m_size];
}
/**
 * Internal Operations
 */
void Pointz::readFile(const std::string& filename)
{   
    // Open the HDF5 file and the dataset DBSCAN in it    
    try 
    {  
        hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);        
        hid_t dset = H5Dopen1(file, DATASET);
        hid_t fileSpace= H5Dget_space(dset);
        
        // Read dataset size and calculate chunk size 
        hsize_t count[2];
        H5Sget_simple_extent_dims(fileSpace, count,NULL);
        this->m_totalSize = count[0];
        hsize_t chunkSize =(this->m_totalSize / this->m_mpiSize) + 1;
        hsize_t offset[2] = {this->m_mpiRank * chunkSize, 0};
        count[0] = std::min(chunkSize, this->m_totalSize - offset[0]);

        // Initialize members
        this->m_size         = count[0];
        this->m_dimensions   = count[1];
        this->m_cells        = new Cell[this->m_size];
        this->m_points       = new Coord[this->m_size * this->m_dimensions];
        this->m_initialOrder = new size_t[this->m_size];        
        std::iota(this->m_initialOrder, this->m_initialOrder + this->m_size, this->m_mpiRank * chunkSize);

        // Read data
        hid_t memSpace = H5Screate_simple(2, count, NULL);
        H5Sselect_hyperslab(fileSpace,H5S_SELECT_SET,offset, NULL, count, NULL);
        H5Dread(dset, H5T_IEEE_F32LE, memSpace, fileSpace,H5P_DEFAULT, m_points);

        // Check if there is an "Cluster" dataset in the file
        if (!this->m_mpiRank)
        {
            htri_t exists = H5Lexists(file, "Clusters", H5P_DEFAULT);
            if (!exists)
            {
                hsize_t dims[2] = {this->m_totalSize, 1};
                hid_t globalSpace = H5Screate_simple(1,dims,NULL);     
                hid_t clusterSet = H5Dcreate1(file, "Clusters", H5T_NATIVE_LONG ,globalSpace, H5P_DEFAULT);
                H5Fclose(clusterSet);
            }
        }

        // Close file and dataset
        H5Dclose(dset);
        H5Fclose(file);    
    }
    catch(herr_t error)
    {
        if (!this->m_mpiRank)
        {
            std::cerr << "Could not open file " << filename << std::endl;
        }
        exit(this->m_mpiRank ? EXIT_SUCCESS : EXIT_FAILURE); 
    }
}

void Pointz::writeClusterToFile(const std::string& filename) const
{
    try 
    {        
        // Open Dataset
        hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);        
        hid_t dset = H5Dopen1(file, "Clusters");
        
        // Create data space
        hsize_t chunkSize =(this->m_totalSize / this->m_mpiSize) + 1;
        hsize_t count[2] = {this->m_size, 1};
        hsize_t start[2] = {this->m_mpiRank * chunkSize , 0};
        hid_t memSpace = H5Screate_simple(1, count, NULL);  
        hid_t fileSpace = H5Dget_space(dset);
        
        // Select area to write
        H5Sselect_hyperslab(fileSpace,H5S_SELECT_SET, start, NULL, count, NULL);
        
        // Write
        H5Dwrite(dset, H5T_NATIVE_LONG, memSpace, fileSpace, H5P_DEFAULT, this->m_clusters);       
        
        // Close 
        H5Dclose(dset);
        H5Fclose(file);
       
    }
    catch(herr_t error)
    {
        if (!this->m_mpiRank)
        {
            std::cerr << "Could not open file " << filename << std::endl;
        }
        exit(this->m_mpiRank ? EXIT_SUCCESS : EXIT_FAILURE); 
    }
}

/**
 * Operations
 */
void Pointz::resetClusters()
{
    std::fill(this->m_clusters, this->m_clusters + this->m_size, NOT_VISITED);
}

void Pointz::mergeNeighborHalos(const size_t lowerHaloBound,const  size_t upperHaloBound, Rules& rules, const Cuts& cuts)
{
    int sendCounts[this->m_mpiSize];
    int recvCounts[this->m_mpiSize];
    
    for (size_t i = 0; i < cuts.size(); ++i)
    {
        sendCounts[i] = (int) (cuts[i].second - cuts[i].first);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Alltoall(sendCounts, 1, MPI_INT, recvCounts, 1, MPI_INT, MPI_COMM_WORLD);
    
    size_t recvTotal = 0;
    int sendDispls[this->m_mpiSize];
    int recvDispls[this->m_mpiSize];
    
    for (int i = 0; i < this->m_mpiSize; ++i)
    {
        sendDispls[i] = cuts[i].first;
        recvDispls[i] = recvTotal;
        
        recvTotal += (size_t) recvCounts[i];
    }
    
    Cluster recvCuts[recvTotal];
    MPI_Alltoallv(this->m_clusters, sendCounts, sendDispls, MPI_LONG, recvCuts, recvCounts, recvDispls, MPI_LONG, MPI_COMM_WORLD);
    for (int i = 0; i < this->m_mpiSize; ++i)
    {
        size_t offset = (i < this->m_mpiRank ? lowerHaloBound : upperHaloBound - recvCounts[i]);
        this->mergeNeighborClusters(recvCuts + recvDispls[i], recvCounts[i], offset, rules);
    }
}

void Pointz::mergeNeighborClusters(Cluster* halo, size_t count, size_t offset, Rules& rules)
{   
    for (size_t i = 0; i < count; ++i)
    {
        const size_t  index   = i + offset;
        const Cluster cluster = halo[i];
                
        if (this->corePoint(index))
        {
            const std::pair<Cluster, Cluster> minmax = std::minmax(this->cluster(index), cluster);
            rules.update(minmax.second, minmax.first);
        }
        else
        {
            this->cluster(index, cluster, false);
        }
    }
}

void Pointz::sortByOrder(size_t maxDigits, size_t lowerBound, size_t upperBound)
{    
    std::vector<std::vector<size_t> > buckets(maxDigits, std::vector<size_t>(DIGITS));
    size_t size            = upperBound - lowerBound;
    this->m_points        += lowerBound * this->m_dimensions;
    this->m_initialOrder  += lowerBound;
    this->m_clusters      += lowerBound;
    Cluster* clusterBuffer = new Cluster[size];
    size_t*  orderBuffer   = new size_t[size];
    Coord*   pointsBuffer  = new Coord[size * this->m_dimensions];
    
    for (size_t j = 0; j < maxDigits; ++j)
    {
        for(size_t f = 0; f < DIGITS; ++f)
        {
            buckets[j][f] = 0;
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < maxDigits; ++j)
        {
            const size_t base  = power(j);
            const size_t digit = this->m_initialOrder[i] / base % DIGITS;
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
        for (size_t i = size - 1; i < NOT_VISITED; --i)
        {
            size_t unit = this->m_initialOrder[i] / base % DIGITS;
            size_t pos  = --buckets[j][unit];

            for (size_t d = 0; d < this->dimensions(); ++d)
            {
                pointsBuffer[pos * this->m_dimensions + d] = this->operator [](i)[d];
            }
            orderBuffer[pos]   = this->m_initialOrder[i];
            clusterBuffer[pos] = this->m_clusters[i];
        }
        std::copy(orderBuffer   , orderBuffer   + size, this->m_initialOrder);
        std::copy(clusterBuffer , clusterBuffer + size, this->m_clusters );
        std::copy(pointsBuffer  , pointsBuffer  + size * this->m_dimensions, this->m_points);
    }
    
    
    this->m_points       -= lowerBound * this->m_dimensions;
    this->m_initialOrder -= lowerBound;
    this->m_clusters     -= lowerBound;
    
    delete[] pointsBuffer;
    delete[] orderBuffer;
    delete[] clusterBuffer;
}

 void Pointz::sortByCell(const CellIndex& index)
 {
     // Initialization
     Cell*   cellBuffer  = new Cell[this->m_size];
     size_t* orderBuffer = new size_t[this->m_size];
     Coord*  pointBuffer = new Coord[this->m_size * this->m_dimensions];
     
     std::unordered_map<size_t, std::atomic<size_t>> counter;
     for (auto pair : index)
     {
         counter[pair.first].store(0);
     }
     
     // Sorting Off-Place
     #pragma omp parallel for if(this->m_mpiSize == 1)
     for (size_t i = 0; i < m_size; ++i)
     {
         const auto& locator = index.find(this->m_cells[i]);
         size_t copyTo       = locator->second.first + (counter[locator->first]++);
         for (size_t d = 0; d < this->m_dimensions; ++d)
         {
             pointBuffer[copyTo * this->m_dimensions + d] = this->m_points[i * this->m_dimensions + d];
         }
         cellBuffer[copyTo]  = this->m_cells[i];
         orderBuffer[copyTo] = this->m_initialOrder[i];
     }         
     
     // Copy In-Place
     std::copy(cellBuffer,  cellBuffer  + this->m_size, this->m_cells);
     std::copy(orderBuffer, orderBuffer + this->m_size, this->m_initialOrder);
     std::copy(pointBuffer, pointBuffer + this->m_size * this->m_dimensions, this->m_points);
     
     delete[] cellBuffer;
     delete[] pointBuffer;
 }

size_t Pointz::redistributePoints(int* counts, int* displs, int* rCounts, int* rDispls, bool incCluster)
{
    int countsS[this->m_mpiSize];
    int displsS[this->m_mpiSize];
    int rCountsS[this->m_mpiSize];
    int rDisplsS[this->m_mpiSize];
    
    size_t total = 0;
    for (int i = 0; i < this->m_mpiSize; ++i)
    {
        total      += rCounts[i];
        countsS[i]  = counts[i]  / this->m_dimensions;
        displsS[i]  = displs[i]  / this->m_dimensions;
        rCountsS[i] = rCounts[i] / this->m_dimensions;
        rDisplsS[i] = rDispls[i] / this->m_dimensions;
    }
    Coord*  pointBuffer = new Coord[total];
    size_t* orderBuffer = new size_t[total / this->m_dimensions];
    
    MPI_Alltoallv(this->m_points, counts, displs, MPI_FLOAT, pointBuffer, rCounts, rDispls, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Alltoallv(this->m_initialOrder, countsS, displsS, MPI_UNSIGNED_LONG, orderBuffer, rCountsS, rDisplsS, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    
    delete[] this->m_points;
    delete[] this->m_cells;
    delete[] this->m_initialOrder;
    
    this->m_size         = total / this->m_dimensions;
    this->m_cells        = new Cell[this->m_size];
    this->m_initialOrder = new size_t[this->m_size];
    this->m_points       = pointBuffer;
    this->m_initialOrder = orderBuffer;
    
    if (incCluster)
    {
        Cluster* clusterBuffer = new Cluster[this->m_size];        
        MPI_Alltoallv(this->m_clusters, countsS, displsS, MPI_LONG, clusterBuffer, rCountsS, rDisplsS, MPI_LONG, MPI_COMM_WORLD);         
        delete[] this->m_clusters; 
        this->m_clusters = clusterBuffer;
    }
    else
    {      
        delete[] this->m_clusters; 
        this->m_clusters = new Cluster[this->m_size]; 
        std::fill(this->m_clusters, this->m_clusters + this->m_size, NOT_VISITED);
    }
    
    return total;    
}

void Pointz::recoverInitialOrder(size_t lowerBound, size_t upperBound)
{
    const float magnitude = ceil(log10(this->m_totalSize));
    this->sortByOrder(magnitude, lowerBound, upperBound);
    int counts[this->m_mpiSize];
    int offset[this->m_mpiSize];
    int rCounts[this->m_mpiSize];
    int rOffset[this->m_mpiSize];
    
    size_t* start      = this->m_initialOrder + lowerBound;
    size_t* end        = this->m_initialOrder + upperBound;
    size_t* lastOffset = start;
    size_t  chunkSize  = this->m_totalSize / this->m_mpiSize + 1;
    
    for (int i = 0; i < this->m_mpiSize; ++i)
    {
        size_t* index = std::lower_bound(start, end, chunkSize * (i + 1));
        counts[i]     = (int) (index - lastOffset) * this->m_dimensions;
        offset[i]     = (int) (lastOffset - this->m_initialOrder) * this->m_dimensions;
        lastOffset    = index;
    }
    
    MPI_Alltoall(counts, 1, MPI_INT, rCounts, 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < this->m_mpiSize; ++i)
    {
        rOffset[i] = (i == 0) ? 0 : (rOffset[i - 1] + rCounts[i - 1]);
    }
    this->redistributePoints(counts, offset, rCounts, rOffset, true);
    this->sortByOrder(magnitude, 0, this->m_size);
}

/**
 * Destructor
 */
Pointz::~Pointz()
{
    delete[] this->m_clusters;
    delete[] this->m_cells;
    delete[] this->m_points;
    delete[] this->m_initialOrder;
}
