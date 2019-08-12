/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: MPI type selection utility
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#ifndef MPI_UTIL_H
#define MPI_UTIL_H

#include <cstdint>

#include <mpi.h>

template <typename T>
struct MPI_Types;

#define SPECIALIZE_MPI_TYPE(type, mpi_type) \
template <> \
struct MPI_Types<type> { \
    static MPI_Datatype map() {\
        return mpi_type; \
    } \
}

SPECIALIZE_MPI_TYPE(uint8_t,  MPI_UINT8_T);
SPECIALIZE_MPI_TYPE(uint16_t, MPI_UINT16_T);
SPECIALIZE_MPI_TYPE(uint32_t, MPI_UINT32_T);
SPECIALIZE_MPI_TYPE(uint64_t, MPI_UINT64_T);

SPECIALIZE_MPI_TYPE(int8_t,  MPI_INT8_T);
SPECIALIZE_MPI_TYPE(int16_t, MPI_INT16_T);
SPECIALIZE_MPI_TYPE(int32_t, MPI_INT32_T);
SPECIALIZE_MPI_TYPE(int64_t, MPI_INT64_T);

SPECIALIZE_MPI_TYPE(float,  MPI_FLOAT);
SPECIALIZE_MPI_TYPE(double, MPI_DOUBLE);

#endif // MPI_UTIL_H
