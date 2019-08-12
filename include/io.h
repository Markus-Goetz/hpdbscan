/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: HDF5 I/O layer for HPDBSCAN
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#ifndef IO_H
#define IO_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <hdf5.h>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef WITH_MPI
#include <mpi.h>
#include "mpi_util.h"
#endif

#include "constants.h"
#include "dataset.h"

class IO {
public:
    static Dataset read_hdf5(const std::string& path, const std::string& dataset_name) {
        #ifdef WITH_MPI
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        #endif

        // disable libhdf5 error printing
        herr_t (*old_func)(hid_t, void*);
        void* old_client_data;

        H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
        H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);

        // open the file
        hid_t file_handle = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_handle < 0) {
            throw std::invalid_argument("Could not open " + path);
        }

        // retrieve the dataset
        hid_t dataset_handle = H5Dopen1(file_handle, dataset_name.c_str());
        if (dataset_handle < 0) {
            throw std::invalid_argument("Could not open dataset " + dataset_name);
        }

        // retrieve the data
        hid_t type_handle = H5Dget_type(dataset_handle);
        if (type_handle < 0) {
            throw std::runtime_error("Could not retrieve dataset type");
        }

        // get space extent
        hid_t space_handle = H5Dget_space(dataset_handle);
        if (space_handle < 0) {
            throw std::runtime_error("Could not retrieve dataspace extent, file seems to be corrupted");
        }
        bool is_matrix = H5Sget_simple_extent_ndims(space_handle) == 2;
        if (not is_matrix) {
            throw std::invalid_argument("Data in dataset " + dataset_name + " should be a matrix");
        }

        // retrieve dataset extents
        hsize_t extents[2];
        if (H5Sget_simple_extent_dims(space_handle, extents, nullptr) < 0) {
            throw std::runtime_error("Could not retrieve dataset extents, file seems to be corrupted");
        }

        // initialize the Dataset object
        Dataset dataset(extents, type_handle);

        // calculate the offsets for MPI mode
        #ifdef WITH_MPI
        hsize_t remainder = extents[0] % size;
        dataset.m_chunk[0] /= size;
        if (remainder > static_cast<hsize_t>(rank)) {
            ++dataset.m_chunk[0];
            dataset.m_offset[0] = dataset.m_chunk[0] * rank;
        } else {
            dataset.m_offset[0] = dataset.m_chunk[0] * rank + remainder;
        }
        #endif

        // create the hyperslab
        hid_t hyperslab = H5Screate_simple(sizeof(extents) / BITS_PER_BYTE, dataset.m_chunk, nullptr);
        H5Sselect_hyperslab(space_handle, H5S_SELECT_SET, dataset.m_offset, nullptr, dataset.m_chunk, nullptr);

        // actually read in the data
        hid_t native_type = H5Tget_native_type(type_handle, H5T_DIR_DEFAULT);
        if (native_type < 0) {
            std::invalid_argument("Could not infer matching native data type from dataset data type");
        }
        if (H5Dread(dataset_handle, native_type, hyperslab, space_handle, H5P_DEFAULT, dataset.m_p) < 0) {
            throw std::runtime_error("Failed to read data");
        }

        // restore libhdf5 error printing - everything below is critical, hdf5 should do the diagnostics
        H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);

        // release the handles
        H5Tclose(native_type);
        H5Sclose(hyperslab);
        H5Sclose(space_handle);
        H5Tclose(type_handle);
        H5Dclose(dataset_handle);
        H5Fclose(file_handle);

        // return the resulting dataset
        return dataset;
    }

    static void write_hdf5(const std::string& path, const std::string& dataset_name, Clusters& clusters) {
        #ifdef WITH_MPI
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        #endif

        // disable libhdf5 error printing
        herr_t (*old_func)(hid_t, void*);
        void* old_client_data;

        H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
        H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);

        hsize_t total_count = clusters.size();
        hsize_t chunk_size = total_count;
        hsize_t offset = 0;

        #ifdef WITH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &total_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        chunk_size = total_count / size;
        hsize_t remainder = total_count % size;
        if (remainder > static_cast<hsize_t>(rank)) {
            ++chunk_size;
            offset = chunk_size * rank;
        } else {
            offset = chunk_size * rank + remainder;
        }
        #endif

        // make sure the file exists
        H5Fcreate(path.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
        // attempt to open the file
        hid_t file_handle = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        if (file_handle < 0) {
            throw std::invalid_argument("Could not open " + path);
        }

        // create the dataset
        hid_t data_space = H5Screate_simple(1, &total_count, nullptr);
        H5Dcreate1(file_handle, dataset_name.c_str(), H5T_NATIVE_LONG, data_space, H5P_DEFAULT);

        // ... dataset already exists, previous call fails, just open the existing one
        hid_t dataset_handle = H5Dopen1(file_handle, dataset_name.c_str());
        if (dataset_handle < 0) {
            throw std::invalid_argument("Could not open dataset " + dataset_name);
        }

        // get dataset space extent
        hid_t hyperslab = H5Screate_simple(1, &chunk_size, nullptr);
        hid_t space_handle = H5Dget_space(dataset_handle);
        if (space_handle < 0) {
            throw std::runtime_error("Could not retrieve dataspace extent, file seems to be corrupted");
        }
        // select the partial chunk (aka hyperslab) inside the data space
        H5Sselect_hyperslab(space_handle, H5S_SELECT_SET, &offset, nullptr, &chunk_size, nullptr);

        // write the data to disk
        if (H5Dwrite(dataset_handle, H5T_NATIVE_LONG, hyperslab, space_handle, H5P_DEFAULT, clusters.data()) < 0) {
            throw std::runtime_error("Failed to write data, target data set is probably of wrong size or type");
        }

        // clean up the handles
        H5Sclose(space_handle);
        H5Sclose(hyperslab);
        H5Dclose(dataset_handle);
        H5Sclose(data_space);
        H5Fclose(file_handle);

        // restore libhdf5 error printing - everything below is critical, hdf5 should do the diagnostics
        H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);
    }
};

#endif // IO_H
