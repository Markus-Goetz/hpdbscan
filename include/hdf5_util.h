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

#ifndef HDF5_UTIL_H
#define HDF5_UTIL_H

#include <cstdint>

#include <hdf5.h>

template <typename T>
struct HDF5_Types;

#define SPECIALIZE_HDF5_TYPE(type, hdf5_type) \
template <> \
struct HDF5_Types<type> { \
    static hid_t map() {\
        return hdf5_type; \
    } \
}

SPECIALIZE_HDF5_TYPE(uint8_t,  H5T_NATIVE_UCHAR);
SPECIALIZE_HDF5_TYPE(uint16_t, H5T_NATIVE_USHORT);
SPECIALIZE_HDF5_TYPE(uint32_t, H5T_NATIVE_UINT);
SPECIALIZE_HDF5_TYPE(uint64_t, H5T_NATIVE_ULONG);

SPECIALIZE_HDF5_TYPE(int8_t,  H5T_NATIVE_SCHAR);
SPECIALIZE_HDF5_TYPE(int16_t, H5T_NATIVE_SHORT);
SPECIALIZE_HDF5_TYPE(int32_t, H5T_NATIVE_INT);
SPECIALIZE_HDF5_TYPE(int64_t, H5T_NATIVE_LONG);

SPECIALIZE_HDF5_TYPE(float,  H5T_NATIVE_FLOAT);
SPECIALIZE_HDF5_TYPE(double, H5T_NATIVE_DOUBLE);

#endif // MPI_UTIL_H
