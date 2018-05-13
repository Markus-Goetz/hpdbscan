# HPDBSCAN

Highly parallel DBSCAN (HPDBSCAN) is a shared- and distributed-memory parallel implementation of the Density-Based Spatial Clustering for Applications with Noise (DBSCAN) algorithm. It is written in C++ and may be used as shared library, command line tool of Python module.  

## Dependencies

HPDBSCAN requires the following dependencies. Please make sure, that these are installed, before attempting to compile the code.

* CMake 3.10+
* C++11 compliant compiler (e.g. g++ 4.8+)
* OpenMP 4.0+
* (optional) HDF5 1.8+
* (optional) Message Passing Interface (MPI) 2.0+

## Compilation

HPDBSCAN follows the standard CMake project conventions. Create a build directory, change to it, generate the build script and compile it. A convenience short-hand can be found below.

``` bash
mkdir build && cd build && cmake .. && make
```

The provided CMake script checks, but does not install, all of the necessary dependencies listed above.

## Usage

The typical basic usage of HPDBSCAN is shown below. It assumes that the application is used in a typical high-performance computing setup with multiple distributed nodes and processing cores per node. The data is passed to the application in form of an HDF5 file with 'Data' being the input dataset and results stored in 'Clustering'. 

``` bash
mpirun -np <NODES> ./hpdbscan -t <THREADS> <PATH_TO_HDF5_FILE>
```

## Citation

If you wish to cite HPDBSCAN in your academic work, please use the following reference:

Plain reference
```
Götz, M., Bodenstein, C., Riedel M.,
HPDBSCAN: highly parallel DBSCAN,
Proceedings of the Workshop onf Machine Learning in High-Performance Computing Environments, ACM, 2015.
```

BibTex
``` bibtex
@inproceedings{gotz2015hpdbscan,
  title={HPDBSCAN: highly parallel DBSCAN},
  author={G{\"o}tz, Markus and Bodenstein, Christian and Riedel, Morris},
  booktitle={Proceedings of the Workshop on Machine Learning in High-Performance Computing Environments},
  pages={2},
  year={2015},
  organization={ACM}
}
```

## Contact

If you want to let us know about feature requests, bugs or issues you are kindly referred to the [issue tracker](https://bitbucket.org/markus.goetz/hpdbscan/issues?status=new&status=open).

For any other discussion, please write an [e-mail](mailto:markus.goetz@kit.edu).

