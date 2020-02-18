/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: Indexes the features space to allow fast neighborhood queries
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#ifndef SPATIAL_INDEX_H
#define SPATIAL_INDEX_H

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <hdf5.h>
#include <limits>
#include <numeric>
#include <omp.h>
#include <parallel/algorithm>
#include <vector>

#ifdef WITH_OUTPUT
#include <iostream>
#endif

#include "constants.h"
#include "dataset.h"

#ifdef WITH_MPI
#include <mpi.h>
#include "mpi_util.h"
#endif

template <typename T>
class SpatialIndex {
    Dataset&                   m_data;
    const float                m_epsilon;

    std::vector<T>             m_minimums;
    std::vector<T>             m_maximums;

    std::vector<size_t>        m_cell_dimensions;
    size_t                     m_total_cells;
    Cell                       m_last_cell;
    Cells                      m_cells;
    CellHistogram              m_cell_histogram;
    CellIndex                  m_cell_index;

    std::vector<size_t>        m_swapped_dimensions;
    size_t                     m_halo;
    size_t                     m_global_point_offset;
    std::vector<size_t>        m_initial_order;

    #ifdef WITH_MPI
    int                        m_rank;
    int                        m_size;

    std::vector<CellBounds>    m_cell_bounds;
    std::vector<ComputeBounds> m_compute_bounds;
    #endif

public:
    // implementations of the custom omp reduction operations
    static void vector_min(std::vector<T>& omp_in, std::vector<T>& omp_out) {
        for (size_t index = 0; index < omp_out.size(); ++index) {
            omp_out[index] = std::min(omp_in[index], omp_out[index]);
        }
    }
    #pragma omp declare reduction(vector_min: std::vector<T>: vector_min(omp_in, omp_out)) initializer(omp_priv = omp_orig)

    static void vector_max(std::vector<T>& omp_in, std::vector<T>& omp_out) {
        for (size_t index = 0; index < omp_out.size(); ++index) {
            omp_out[index] = std::max(omp_in[index], omp_out[index]);
        }
    }
    #pragma omp declare reduction(vector_max: std::vector<T>: vector_max(omp_in, omp_out)) initializer(omp_priv = omp_orig)

    static void merge_histograms(CellHistogram& omp_in, CellHistogram& omp_out) {
        for (const auto& cell: omp_in) {
            omp_out[cell.first] += cell.second;
        }
    }
    #pragma omp declare reduction(merge_histograms: CellHistogram: merge_histograms(omp_in, omp_out)) initializer(omp_priv = omp_orig)

private:
    void compute_initial_order() {
        #pragma omp parallel for
        for (size_t i = 0; i < m_data.m_chunk[0]; ++i) {
            m_initial_order[i] += i + m_data.m_offset[0];
        }
    }

    void compute_space_dimensions() {
        const size_t dimensions = m_minimums.size();
        const size_t bytes = m_cells.size() * dimensions;
        const T* end_point = static_cast<T*>(m_data.m_p) + bytes;

        // compute the local feature space minimums and maximums in parallel
        auto& minimums = m_minimums;
        auto& maximums = m_maximums;

        #pragma omp parallel for reduction(vector_min: minimums) reduction(vector_max: maximums)
        for (T* point = static_cast<T*>(m_data.m_p); point < end_point; point += dimensions) {
            for (size_t d = 0; d < dimensions; ++d) {
                const T& coordinate = point[d];
                minimums[d] = std::min(minimums[d], coordinate);
                maximums[d] = std::max(maximums[d], coordinate);
            }
        }

        // exchange globally, if necessary
        #ifdef WITH_MPI
        MPI_Allreduce(MPI_IN_PLACE, m_minimums.data(), dimensions, MPI_Types<T>::map(), MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, m_maximums.data(), dimensions, MPI_Types<T>::map(), MPI_MAX, MPI_COMM_WORLD);
        #endif
    }

    void compute_cell_dimensions() {
        for (size_t i = 0; i < m_cell_dimensions.size(); ++i) {
            size_t cells = static_cast<size_t>(std::ceil((m_maximums[i] - m_minimums[i]) / m_epsilon)) + 1;
            m_cell_dimensions[i] = cells;
            m_total_cells *= cells;
        }
        m_last_cell = m_total_cells;
    }

    void swap_dimensions() {
        // fill the dimensions with an initially correct order
        std::iota(m_swapped_dimensions.begin(), m_swapped_dimensions.end(), 0);
        // swap the dimensions descending by their cell sizes
        std::sort(m_swapped_dimensions.begin(), m_swapped_dimensions.end(), [&] (size_t a, size_t b) {
            return m_cell_dimensions[a] < m_cell_dimensions[b];
        });
        // determine the halo size
        m_halo = m_total_cells / m_cell_dimensions[m_swapped_dimensions.back()];
    }

    void compute_cells() {
        CellHistogram histogram;
        const size_t dimensions = m_data.m_chunk[1];

        #pragma omp parallel for reduction(merge_histograms: histogram)
        for (size_t i = 0; i < m_data.m_chunk[0]; ++i) {
            const T* point = static_cast<T*>(m_data.m_p) + i * dimensions;

            size_t cell = 0;
            size_t accumulator = 1;

            for (size_t d : m_swapped_dimensions) {
                size_t index = static_cast<size_t>(std::floor((point[d] - m_minimums[d]) / m_epsilon));
                cell += index * accumulator;
                accumulator *= m_cell_dimensions[d];
            }

            m_cells[i] = cell;
            ++histogram[cell];
        }
        m_cell_histogram.swap(histogram);
    }

    void compute_cell_index() {
        size_t accumulator = 0;

        // sum up the offset into the points array
        for (auto& cell : m_cell_histogram)
        {
            auto& index  = m_cell_index[cell.first];
            index.first  = accumulator;
            index.second = cell.second;
            accumulator += cell.second;
        }

        // introduce an end dummy
        m_cell_index[m_last_cell].first  = m_cells.size();
        m_cell_index[m_last_cell].second = 0;
    }

    void sort_by_cell() {
        const hsize_t items = m_data.m_chunk[0];
        const hsize_t dimensions = m_data.m_chunk[1];

        // initialize out-of-place buffers
        Cells reordered_cells(items);
        std::vector<size_t> reordered_indices(items);
        std::vector<T> reordered_points(items * dimensions);

        // memory for offset of already placed items
        std::unordered_map<Cell, std::atomic<size_t>> offsets;
        for (const auto&  cell_index : m_cell_index) {
            offsets[cell_index.first].store(0);
        }

        // sorting the points and cells out-of-place, memorize the original order
        #pragma omp parallel for
        for (size_t i = 0; i < items; ++i) {
            const Cell cell = m_cells[i];
            const auto& locator = m_cell_index[cell];
            const size_t copy_to = locator.first + (offsets[cell]++);

            reordered_cells[copy_to] = m_cells[i];
            reordered_indices[copy_to] = m_initial_order[i];
            for (size_t d = 0; d < dimensions; ++d) {
                reordered_points[copy_to * dimensions + d] = static_cast<T*>(m_data.m_p)[i * dimensions + d];
            }
        }

        // move the out-of-place results into the correct in-place buffers
        m_cells.swap(reordered_cells);
        m_initial_order.swap(reordered_indices);
        std::copy(reordered_points.begin(), reordered_points.end(), static_cast<T*>(m_data.m_p));
    }

    #ifdef WITH_MPI
    CellHistogram compute_global_histogram() {
        // fetch cell histograms across all nodes
        int send_counts[m_size];
        int send_displs[m_size];
        int recv_counts[m_size];
        int recv_displs[m_size];

        // determine the number of entries in each process' histogram
        for (int i = 0; i < m_size; ++i) {
            send_counts[i] = m_cell_histogram.size() * 2;
            send_displs[i] = 0;
        }
        MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

        // ... based on this information we can calculate the displacements into the buffer
        size_t entries_count = 0;
        for (int i = 0; i < m_size; ++i) {
            recv_displs[i] = entries_count;
            entries_count += recv_counts[i];
        }

        // serialize the local histogram into a flat buffer
        std::vector<size_t> send_buffer(m_cell_histogram.size() * 2);
        size_t send_buffer_index = 0;
        for (const auto& item : m_cell_histogram) {
            send_buffer[send_buffer_index++] = item.first;
            send_buffer[send_buffer_index++] = item.second;
        }

        // exchange the histograms
        std::vector<size_t> recv_buffer(entries_count);
        MPI_Alltoallv(
            send_buffer.data(), send_counts, send_displs, MPI_UNSIGNED_LONG,
            recv_buffer.data(), recv_counts, recv_displs, MPI_UNSIGNED_LONG, MPI_COMM_WORLD
        );

        // sum-up the entries into a global histogram
        CellHistogram global_histogram;
        for (size_t i = 0; i < entries_count; i += 2) {
            global_histogram[recv_buffer[i]] += recv_buffer[i + 1];
        }

        // remember the new globally last cell
        m_last_cell = global_histogram.rbegin()->first + 1;

        return global_histogram;
    }

    size_t compute_score(const Cell cell_id, const CellHistogram& cell_histogram) {
        const hsize_t dimensions = m_data.m_chunk[1];

        // allocate buffer for the dimensions steps
        Cells neighboring_cells;
        neighboring_cells.reserve(std::pow(3, dimensions));
        neighboring_cells.push_back(cell_id);

        // accumulators for sub-space traversal
        size_t cells_in_lower_space = 1;
        size_t cells_in_current_space = 1;
        size_t points_in_cell = cell_histogram.find(cell_id)->second;
        size_t number_of_points = points_in_cell;

        // iterate through all neighboring cells and up the number of points stored there
        for (size_t d : m_swapped_dimensions) {
            cells_in_current_space *= m_cell_dimensions[d];

            for (size_t i = 0, end = neighboring_cells.size(); i < end; ++i) {
                const Cell current = neighboring_cells[i];

                // cell to the left
                const Cell left = current - cells_in_lower_space;
                if (current % cells_in_current_space >= cells_in_lower_space) {
                    const auto& locator = cell_histogram.find(left);
                    number_of_points += locator != cell_histogram.end() ? locator->second : 0;
                    neighboring_cells.push_back(left);
                }
                // cell to the right
                const Cell right = current + cells_in_lower_space;
                if (current % cells_in_current_space < cells_in_current_space - cells_in_lower_space) {
                    const auto& locator = cell_histogram.find(right);
                    number_of_points += locator != cell_histogram.end() ? locator->second : 0;
                    neighboring_cells.push_back(right);
                }
            }
            cells_in_lower_space = cells_in_current_space;
        }

        return points_in_cell * number_of_points;
    }

    void compute_bounds(const CellHistogram& cell_histogram) {
        // make space in for the values in the bounds variables
        m_cell_bounds.resize(m_size);
        m_compute_bounds.resize(m_size);

        // compute the score value for each cell and accumulate the total score first...
        std::vector<size_t> scores(cell_histogram.size(), 0);
        size_t total_score = 0;
        size_t score_index = 0;

        for (auto& pair : cell_histogram) {
            const Cell cell = pair.first;
            const size_t score = compute_score(cell, cell_histogram);
            scores[score_index++] = score;
            total_score += score;
        }

        // ...to determine the actual bounds
        const size_t score_per_chunk = total_score / m_size + 1;
        size_t accumulator = 0;
        size_t target_rank = 0;
        size_t lower_split_point = 0;
        size_t bound_lower_start = 0;
        auto cell_buckets = cell_histogram.begin();

        // iterate over the score array and find the point where the score per chunk is exceeded
        for (size_t i = 0; i < scores.size(); ++ i) {
            const auto& cell_bucket = cell_buckets++;
            const Cell cell = cell_bucket->first;
            const size_t score = scores[i];

            accumulator += score;
            while (accumulator > score_per_chunk) {
                const size_t split_point = (accumulator - score_per_chunk) / (score / cell_bucket->second);

                // we have have identified the bounds in which the rank needs to compute locally
                m_compute_bounds[target_rank][0] = lower_split_point;
                m_compute_bounds[target_rank][1] = split_point;
                lower_split_point = split_point;

                // determine the cell bounds, i.e. all cells that we need including halo
                auto& bound = m_cell_bounds[target_rank];
                const size_t cell_offset = (bound_lower_start % m_halo) + m_halo;
                bound[0] = cell_offset > bound_lower_start ? 0 : bound_lower_start - cell_offset;
                bound[1] = bound_lower_start;
                bound[2] = cell + 1;
                bound[3] = std::min((bound[2] / m_halo) * m_halo + (m_halo * 2), m_last_cell);

                // update the state, a whole chunk has been assigned
                bound_lower_start = bound[2];
                accumulator = split_point * score / cell_bucket->second;
                // start assigning to the next rank
                ++target_rank;
            }

            // the left-overs are assigned to the current rank
            if (static_cast<int>(target_rank) == m_size - 1 or i == cell_histogram.size() - 1) {
                // compute bounds first
                m_compute_bounds[target_rank][0] = lower_split_point;
                m_compute_bounds[target_rank][1] = 0;

                // cell bounds including halo next
                auto& bound = m_cell_bounds[target_rank];
                const size_t cell_offset = (bound_lower_start % m_halo) + m_halo;
                bound[0] = cell_offset > bound_lower_start ? 0 : bound_lower_start - cell_offset;
                bound[1] = bound_lower_start;
                bound[2] = m_last_cell;
                bound[3] = m_last_cell;

                // we are done here
                break;
            }
        }
    }

    void redistribute_dataset() {
        const size_t dimensions = m_data.m_chunk[1];

        // calculate the send number of points to be transmitted to each rank
        int send_counts[m_size];
        int send_displs[m_size];
        int recv_counts[m_size];
        int recv_displs[m_size];

        for (int i = 0; i < m_size; ++i) {
            const auto& bound = m_cell_bounds[i];
            const size_t lower = m_cell_index.lower_bound(bound[0])->second.first;
            const size_t upper = m_cell_index.lower_bound(bound[3])->second.first;

            send_displs[i] = lower * dimensions;
            send_counts[i] = upper * dimensions - send_displs[i];
        }

        // exchange how much data we send/receive to and from each rank
        MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
        for (int i = 0; i < m_size; ++i) {
            recv_displs[i] = (i == 0) ? 0 : (recv_displs[i - 1] + recv_counts[i - 1]);
        }

        // calculate the corresponding send and receive counts for the label/order vectors
        size_t total_recv_items = 0;
        int send_counts_labels[m_size];
        int send_displs_labels[m_size];
        int recv_counts_labels[m_size];
        int recv_displs_labels[m_size];

        for (int i = 0; i < m_size; ++i) {
            total_recv_items += recv_counts[i];
            send_counts_labels[i] = send_counts[i] / dimensions;
            send_displs_labels[i] = send_displs[i] / dimensions;
            recv_counts_labels[i] = recv_counts[i] / dimensions;
            recv_displs_labels[i] = recv_displs[i] / dimensions;
        }

        // allocate new buffers for the points and the order vectors
        T* point_buffer = new T[total_recv_items];
        std::vector<size_t> order_buffer(total_recv_items / dimensions);

        // actually transmit the data
        MPI_Alltoallv(
            static_cast<T*>(m_data.m_p), send_counts, send_displs, MPI_Types<T>::map(),
            point_buffer, recv_counts, recv_displs, MPI_Types<T>::map(), MPI_COMM_WORLD
        );
        MPI_Alltoallv(
            m_initial_order.data(), send_counts_labels, send_displs_labels, MPI_UNSIGNED_LONG,
            order_buffer.data(), recv_counts_labels, recv_displs_labels, MPI_UNSIGNED_LONG, MPI_COMM_WORLD
        );

        // clean up the previous data
        delete[] static_cast<T*>(m_data.m_p);
        m_cells.clear();
        m_cell_index.clear();

        // assign the new data
        const hsize_t new_item_count = total_recv_items / dimensions;
        m_data.m_chunk[0] = new_item_count;
        m_cells.resize(new_item_count);
        m_data.m_p = point_buffer;
        m_initial_order.swap(order_buffer);
    }

    void compute_global_point_offset() {
        m_global_point_offset = upper_halo_bound() - lower_halo_bound();
        MPI_Exscan(MPI_IN_PLACE, &m_global_point_offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (m_rank == 0) m_global_point_offset = 0;
    }

    void sort_by_order(Clusters& clusters) {
        // allocate the radix buckets
        const size_t maximum_digit_count = static_cast<size_t>(std::ceil(std::log10(m_data.m_shape[0])));
        std::vector<std::vector<size_t>> buckets(maximum_digit_count, std::vector<size_t>(RADIX_BUCKETS));

        // count the items per bucket
        size_t lower_bound = lower_halo_bound();
        size_t upper_bound = upper_halo_bound();
        const size_t items = upper_bound - lower_bound;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < items; ++i) {
            for (size_t j = 0; j < maximum_digit_count; ++j) {
                const size_t base  = RADIX_POWERS[j];
                const size_t digit = m_initial_order[i + lower_bound] / base % RADIX_BUCKETS;
                #pragma omp atomic
                ++buckets[j][digit];
            }
        }

        // accumulate the bucket entries to get the offsets
        #pragma omp parallel for shared(buckets)
        for (size_t j = 0; j < maximum_digit_count; ++j) {
            for (size_t f = 1; f < RADIX_BUCKETS; ++f) {
                buckets[j][f] += buckets[j][f-1];
            }
        }

        // actually reorder the points out-of-place
        const hsize_t dimensions = m_data.m_chunk[1];
        Clusters cluster_buffer(items);
        std::vector<size_t> order_buffer(items);
        T* point_buffer = new T[items * dimensions];

        for (size_t j = 0; j < maximum_digit_count; ++j) {
            const size_t base = RADIX_POWERS[j];
            const size_t point_offset = lower_bound * dimensions;

            // assign the number to the respective radix bucket
            for (size_t i = items - 1; i < items; --i) {
                size_t unit = m_initial_order[i + lower_bound] / base % RADIX_BUCKETS;
                size_t pos  = --buckets[j][unit];

                order_buffer[pos] = m_initial_order[i + lower_bound];
                cluster_buffer[pos] = clusters[i + lower_bound];
                for (size_t d = 0; d < dimensions; ++d) {
                    point_buffer[pos * dimensions + d] = static_cast<T*>(m_data.m_p)[i * dimensions + d + point_offset];
                }
            }

            // swap the buffers
            clusters.swap(cluster_buffer);
            m_initial_order.swap(order_buffer);
            T* temp = static_cast<T*>(m_data.m_p);
            m_data.m_p = point_buffer;
            point_buffer = temp;

            // this is somewhat hacky, in the first round we have the original buffers including(!) halos
            // after the first swap, we do not anymore, since we reduced all the elements done to the non-halo zone
            // we can easily adjust to that by removing the initial halo lower_bound offset
            if (j == 0) {
                lower_bound = 0;
            }
        }

        // clean up
        delete[] point_buffer;
        m_data.m_chunk[0] = items;
    }
    #endif

public:
    SpatialIndex(Dataset& data, const float epsilon)
      : m_data(data),
        m_epsilon(epsilon),
        m_minimums(data.m_chunk[1], std::numeric_limits<T>::max()),
        m_maximums(data.m_chunk[1], std::numeric_limits<T>::min()),
        m_cell_dimensions(data.m_chunk[1], 0),
        m_total_cells(1),
        m_last_cell(0),
        m_cells(data.m_chunk[0], 0),
        m_swapped_dimensions(data.m_chunk[1], 0),
        m_halo(0),
        m_global_point_offset(0),
        m_initial_order(data.m_chunk[0]) {

        // determine the space dimensions, the corresponding number of cells for each feature dimension
        #ifdef WITH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &m_size);
        #endif

        #ifdef WITH_OUTPUT
            double start = omp_get_wtime();

            #ifdef WITH_MPI
            if (m_rank == 0) {
            #endif
            std::cout << "Computing cell space" << std::endl;
            std::cout << "\tComputing dimensions..." << std::flush;
            #ifdef WITH_MPI
            }
            #endif
        #endif
        compute_initial_order();
        compute_space_dimensions();
        compute_cell_dimensions();
        swap_dimensions();

        // determine the cell for each point, compute the cell histogram and index the points as if they were sorted...
        #ifdef WITH_OUTPUT
            #ifdef WITH_MPI
            if (m_rank == 0) {
            #endif
            std::cout << "[OK] in " << omp_get_wtime() - start << std::endl;
            std::cout << "\tComputing cells...     " << std::flush;
            #ifdef WITH_MPI
            }
            #endif
            start = omp_get_wtime();
        #endif
        compute_cells();
        compute_cell_index();

        // ... actually sort the points to allow for O(1) access during neighborhood queries
        #ifdef WITH_OUTPUT
            #ifdef WITH_MPI
            if (m_rank == 0) {
            #endif
            std::cout << "[OK] in " << omp_get_wtime() - start << std::endl;
            std::cout << "\tSorting points...      " << std::flush;
            #ifdef WITH_MPI
            }
            #endif
            start = omp_get_wtime();
        #endif
        sort_by_cell();

        #ifdef WITH_OUTPUT
            #ifdef WITH_MPI
            if (m_rank == 0) {
            #endif
            std::cout << "[OK] in " << omp_get_wtime() - start << std::endl;
            #ifdef WITH_MPI
            }
            #endif
        #endif

        // communicate the cell histograms and redistribute the points - only necessary when MPI is turned on
        #ifdef WITH_MPI
            #ifdef WITH_OUTPUT
            start = omp_get_wtime();
            if (m_rank == 0) {
                std::cout << "\tDistributing points... " << std::flush;
            }
            #endif
            // compute a global histogram and redistribute the points based on that
            CellHistogram global_histogram = compute_global_histogram();
            compute_bounds(global_histogram);
            global_histogram.clear();
            redistribute_dataset();
            // after the redistribution we have to reindex the new data yet again
            compute_cells();
            compute_cell_index();
            compute_global_point_offset();
            sort_by_cell();
            #ifdef WITH_OUTPUT
            if (m_rank == 0) {
                std::cout << "[OK] in " << omp_get_wtime() - start << std::endl;
            }
            #endif
        #endif
    }

    #ifdef WITH_MPI
    size_t lower_halo_bound() const {
        size_t lower = m_cell_index.lower_bound(m_cell_bounds[m_rank][1])->second.first;
        return lower - m_compute_bounds[m_rank][0];
    }

    size_t upper_halo_bound() const {
        size_t upper = m_cell_index.lower_bound(m_cell_bounds[m_rank][2])->second.first;
        return upper - m_compute_bounds[m_rank][1];
    }

    Cuts compute_cuts() const {
        Cuts cuts(m_size, Locator(0, 0));

        for (int i = 0; i < m_size; ++i) {
            // skip own rank
            if (i == m_rank) {
                continue;
            }
            const auto& cell_bound = m_cell_bounds[i];
            const auto& compute_bound = m_compute_bounds[i];

            // lower bound
            const auto& lower_cut = m_cell_index.lower_bound(cell_bound[1]);
            cuts[i].first = lower_cut->second.first;
            if (lower_cut->first != m_last_cell or m_cell_index.find(m_last_cell - 1) != m_cell_index.end()) {
                cuts[i].first = cuts[i].first < compute_bound[0] ? 0 : cuts[i].first - compute_bound[0];
            }
            // upper bound
            const auto& upper_cut = m_cell_index.lower_bound(cell_bound[2]);
            cuts[i].second =  upper_cut->second.first;
            if (upper_cut->first != m_last_cell || m_cell_index.find(m_last_cell - 1) != m_cell_index.end()) {
                cuts[i].second = cuts[i].second < compute_bound[1] ? 0 : cuts[i].second - compute_bound[1];
            }
        }

        return cuts;
    }
    #else
    size_t lower_halo_bound() const {
        return 0;
    }

    size_t upper_halo_bound() const {
        return m_data.m_chunk[0];
    }
    #endif

    inline Cell cell_of(size_t index) const {
        return m_cells[index];
    }

    std::vector<size_t> get_neighbors(const Cell cell) const {
        const hsize_t dimensions = m_data.m_chunk[1];

        // allocate some space for the neighboring cells, be pessimistic and reserve 3^dims for possibly all neighbors
        Cells neighboring_cells;
        neighboring_cells.reserve(std::pow(3, dimensions));
        neighboring_cells.push_back(cell);

        // cell accumulators
        size_t cells_in_lower_space = 1;
        size_t cells_in_current_space = 1;
        size_t number_of_points = m_cell_index.find(cell)->second.second;

        // fetch all existing neighboring cells
        for (size_t d : m_swapped_dimensions) {
            cells_in_current_space *= m_cell_dimensions[d];

            for (size_t i = 0, end = neighboring_cells.size(); i < end; ++i) {
                const Cell current_cell = neighboring_cells[i];

                // check "left" neighbor - a.k.a the cell in the current dimension that has a lower number
                const Cell left = current_cell - cells_in_lower_space;
                const auto found_left = m_cell_index.find(left);
                if (current_cell % cells_in_current_space >= cells_in_lower_space) {
                    neighboring_cells.push_back(left);
                    number_of_points += found_left != m_cell_index.end() ? found_left->second.second : 0;
                }
                // check "right" neighbor - a.k.a the cell in the current dimension that has a higher number
                const Cell right = current_cell + cells_in_lower_space;
                const auto found_right = m_cell_index.find(right);
                if (current_cell % cells_in_current_space < cells_in_current_space - cells_in_lower_space) {
                    neighboring_cells.push_back(right);
                    number_of_points += found_right != m_cell_index.end() ? found_right->second.second : 0;
                }
            }
            cells_in_lower_space = cells_in_current_space;
        }

        // copy the points from the neighboring cells over
        std::vector<size_t> neighboring_points;
        neighboring_points.reserve(number_of_points);

        for (size_t neighbor_cell : neighboring_cells) {
            const auto found = m_cell_index.find(neighbor_cell);
            // skip empty cells
            if (found == m_cell_index.end()) {
                continue;
            }
            // ... otherwise copy the points over
            const Locator& locator = found->second;
            neighboring_points.resize(neighboring_points.size() + locator.second);
            std::iota(neighboring_points.end() - locator.second, neighboring_points.end(), locator.first);
        }

        return neighboring_points;
    }

    Cluster region_query(const size_t point_index, const std::vector<size_t>& neighboring_points, const float EPS2,
                         const Clusters& clusters, std::vector<size_t>& min_points_area) const {
        const size_t dimensions = m_data.m_chunk[1];
        const T* point = static_cast<T*>(m_data.m_p) + point_index * dimensions;
        Cluster cluster_label = m_global_point_offset + point_index + 1;

        // iterate through all neighboring points and check whether they are in range
        for (size_t neighbor: neighboring_points) {
            double offset = 0.0;
            const T* other_point = static_cast<T*>(m_data.m_p) + neighbor * dimensions;

            // determine euclidean distance to other point
            for (size_t d = 0; d < dimensions; ++d) {
		const size_t distance = point[d] - other_point[d];
                offset += distance * distance;
            }
            // .. if in range, add it to the vector with in range points
            if (offset <= EPS2) {
                const Cluster neighbor_label = clusters[neighbor];

                min_points_area.push_back(neighbor);
                // if neighbor point has an assigned label and it is a core, determine what label to take
                if (neighbor_label != NOT_VISITED and neighbor_label < 0) {
                    cluster_label = std::min(cluster_label, std::abs(neighbor_label));
                }
            }
        }

        return cluster_label;
    }

    void recover_initial_order(Clusters& clusters) {
        const hsize_t dimensions = m_data.m_chunk[1];

        #ifdef WITH_MPI
        sort_by_order(clusters);

        // allocate buffers to do an inverse exchange
        int send_counts[m_size];
        int send_displs[m_size];
        int recv_counts[m_size];
        int recv_displs[m_size];

        const size_t lower_bound = lower_halo_bound();
        const size_t upper_bound = upper_halo_bound();
        const size_t items = upper_bound - lower_bound;
        const size_t chunk_size = m_data.m_shape[0] / m_size;
        const size_t remainder = m_data.m_shape[0] % static_cast<size_t>(m_size);
        size_t previous_offset = 0;

        // find all the points that have a global index less than each rank's chunk size
        for (size_t i = 1; i < static_cast<size_t>(m_size) + 1; ++i) {
            const size_t chunk_end = chunk_size * i + (remainder > i ? i : remainder);

            const auto split_iter = std::lower_bound(m_initial_order.begin(), m_initial_order.begin() + items, chunk_end);
            size_t split_index = split_iter - m_initial_order.begin();

            send_counts[i - 1] = static_cast<int>(split_index - previous_offset);
            send_displs[i - 1] = static_cast<int>(previous_offset);
            previous_offset = split_index;
        }

        // exchange the resulting item counts and displacements to get the incoming items for this rank
        MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
        for (int i = 0; i < m_size; ++i) {
            recv_displs[i] = (i == 0) ? 0 : recv_displs[i - 1] + recv_counts[i - 1];
        }

        // redistribute the dataset to their original owner ranks
        size_t total_recv_items = 0;
        int send_counts_points[m_size];
        int send_displs_points[m_size];
        int recv_counts_points[m_size];
        int recv_displs_points[m_size];

        for (int i = 0; i < m_size; ++i) {
            total_recv_items += recv_counts[i];
            send_counts_points[i] = send_counts[i] * dimensions;
            send_displs_points[i] = send_displs[i] * dimensions;
            recv_counts_points[i] = recv_counts[i] * dimensions;
            recv_displs_points[i] = recv_displs[i] * dimensions;
        }

        // allocate new buffers for the points and the order vectors
        T* point_buffer = new T[total_recv_items * dimensions];
        std::vector<size_t> order_buffer(total_recv_items);
        Clusters cluster_buffer(total_recv_items);

        // actually transmit the data
        MPI_Alltoallv(
            static_cast<T*>(m_data.m_p), send_counts_points, send_displs_points, MPI_Types<T>::map(),
            point_buffer, recv_counts_points, recv_displs_points, MPI_Types<T>::map(), MPI_COMM_WORLD
        );
        MPI_Alltoallv(
            m_initial_order.data(), send_counts, send_displs, MPI_Types<size_t>::map(),
            order_buffer.data(), recv_counts, recv_displs, MPI_LONG, MPI_COMM_WORLD
        );
        MPI_Alltoallv(
            clusters.data(), send_counts, send_displs, MPI_Types<size_t>::map(),
            cluster_buffer.data(), recv_counts, recv_displs, MPI_LONG, MPI_COMM_WORLD
        );

        // assign the new data
        delete[] static_cast<T*>(m_data.m_p);
        m_data.m_p = point_buffer;
        point_buffer = nullptr;
        m_data.m_chunk[0] = total_recv_items;
        m_initial_order.swap(order_buffer);
        order_buffer.clear();
        clusters.swap(cluster_buffer);
        cluster_buffer.clear();
        #endif

        // only reordering step needed for non-MPI implementation and final local reordering for MPI version
        // out-of-place rearranging of items
        T* local_point_buffer = new T[m_initial_order.size() * dimensions];
        std::vector<size_t> local_order_buffer(m_initial_order.size());
        Clusters local_cluster_buffer(m_initial_order.size());

        #pragma omp parallel for
        for (size_t i = 0; i < m_initial_order.size(); ++i) {
            const size_t copy_to = m_initial_order[i] - m_data.m_offset[0];

            local_order_buffer[copy_to] = m_initial_order[i];
            local_cluster_buffer[copy_to] = clusters[i];
            for (size_t d = 0; d < dimensions; ++d) {
                local_point_buffer[copy_to * dimensions + d] = static_cast<T*>(m_data.m_p)[i * dimensions + d];
            }
        }

        clusters.swap(local_cluster_buffer);
        m_initial_order.swap(local_order_buffer);
        delete[] static_cast<T*>(m_data.m_p);
        m_data.m_p = local_point_buffer;
    }
};

#endif // SPATIAL_INDEX_H
