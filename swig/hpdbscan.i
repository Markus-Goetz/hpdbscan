%module hpdbscan

%{
    #define SWIG_FILE_WITH_INIT
    #include <cstdint>
    #include "hpdbscan.h"

    #ifdef WITH_MPI
    #include <mpi.h>
    #endif
%}

%include "typemaps.i"
%include "stdint.i"
%include "std_string.i"
%include "std_vector.i"
%include "numpy.i"

namespace std {
    %template(ClusterVector) vector<ptrdiff_t>;
}

%init %{
    import_array();
%}

%apply(double* IN_ARRAY2, int DIM1, int DIM2){(double* data, int dim0, int dim1)};

class HPDBSCAN {
public:
    HPDBSCAN(float epsilon, size_t min_points) throw(std::invalid_argument);
    std::vector<ptrdiff_t> cluster(const std::string& path, const std::string& dataset) throw(std::invalid_argument, std::runtime_error);
    std::vector<ptrdiff_t> cluster(const std::string& path, const std::string& dataset, int threads) throw(std::invalid_argument, std::runtime_error);
    std::vector<ptrdiff_t> cluster(double* data, int dim0, int dim1) throw(std::invalid_argument, std::runtime_error);
};
