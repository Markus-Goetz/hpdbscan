/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: Dataset abstraction
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#ifndef DATASET_H
#define DATASET_H

#include <algorithm>
#include <hdf5.h>
#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "constants.h"

struct Dataset {
    hsize_t m_shape[2];
    hsize_t m_chunk[2];
    hsize_t m_offset[2] = {0, 0};
    hid_t m_type;
    void* m_p;

    Dataset(const hsize_t shape[2], hid_t type) {
        std::copy(shape, shape + 2, m_shape);
        std::copy(shape, shape + 2, m_chunk);

        m_type = H5Tcopy(type);
        m_p = malloc(shape[0] * shape[1] * H5Tget_precision(type) / BITS_PER_BYTE);
    }

    template <typename T>
    Dataset(T* data, const hsize_t shape[2], hid_t type) : Dataset(shape, type) {
        std::copy(data, data + shape[0] * shape[1], static_cast<T*>(m_p));

        #ifdef WITH_MPI
        // determine the global shape...
        MPI_Allreduce(MPI_IN_PLACE, m_shape, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        // ... and the chunk offset
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Exscan(m_chunk, m_offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) m_offset[0] = 0;
        #endif
    }

    ~Dataset() {
        H5Tclose(m_type);
        free(m_p);
    }
};

#endif // DATASET_H
