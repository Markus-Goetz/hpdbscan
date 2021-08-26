CMAKE_MINIMUM_REQUIRED(VERSION 3.10 FATAL_ERROR)
PROJECT(HPDBSCAN)

# include additional modules
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/)

# set c++ properties
ENABLE_LANGUAGE(CXX)
SET(CMAKE_CXX_STANDARD 11)

# set build options
SET(CMAKE_CXX_FLAGS "-Ofast -Wall -Wextra")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")

# add the include and source directories
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

# set up the build targets
## set up the shared library compilation
ADD_LIBRARY(hpdbscan SHARED src/hpdbscan.cpp)
# set up the cli executable
ADD_EXECUTABLE(hpdbscan-bin src/cli.cpp)

# define options
TARGET_COMPILE_DEFINITIONS(hpdbscan-bin PRIVATE WITH_OUTPUT)

# check dependencies
## mpi, optional for distributed computing
FIND_PACKAGE(MPI)
IF(MPI_FOUND)
    SET(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
    TARGET_COMPILE_DEFINITIONS(hpdbscan PRIVATE WITH_MPI)
    TARGET_COMPILE_DEFINITIONS(hpdbscan-bin PRIVATE WITH_MPI)
ELSE()
    MESSAGE("BUILDING WITHOUT MPI SUPPORT")
ENDIF()

## openmp
FIND_PACKAGE(OpenMP 4.0 REQUIRED)
TARGET_LINK_LIBRARIES(hpdbscan PRIVATE ${OpenMP_CXX_FLAGS})
TARGET_COMPILE_OPTIONS(hpdbscan PRIVATE ${OpenMP_CXX_FLAGS})
TARGET_LINK_LIBRARIES(hpdbscan-bin PRIVATE ${OpenMP_CXX_FLAGS})
TARGET_COMPILE_OPTIONS(hpdbscan-bin PRIVATE ${OpenMP_CXX_FLAGS})

## hdf5
FIND_PACKAGE(HDF5 1.8.10 REQUIRED)
INCLUDE_DIRECTORIES("${HDF5_INCLUDE_DIRS}")
TARGET_LINK_LIBRARIES(hpdbscan-bin PRIVATE "${HDF5_LIBRARIES}")

## swig and python detection for optional bindings
FIND_PACKAGE(SWIG)
IF(SWIG_FOUND)
    FIND_PACKAGE(PythonInterp)
    IF(PYTHONINTERP_FOUND)
        FIND_PACKAGE(PythonLibs)
        IF("${PYTHON_INCLUDE_DIRS}" STREQUAL "")
            MESSAGE("PYTHON HEADERS NOT FOUND, BUILDING WITHOUT BINDINGS")
            MESSAGE("TRY INSTALLING THE python-dev OR python-devel PACKAGE")
        ELSE()
            INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
	    FIND_PACKAGE(NumPy)
	    IF(NUMPY_FOUND)
		EXECUTE_PROCESS(COMMAND swig -c++ -python -I"${PYTHON_INCLUDE_DIRS}" -I"${NUMPY_INCLUDE_DIRS}" -o "${CMAKE_CURRENT_BINARY_DIR}/hpdbscan_wrap.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/swig/hpdbscan.i")
                ADD_LIBRARY(hpdbscan-binding SHARED ${CMAKE_CURRENT_BINARY_DIR}/hpdbscan_wrap.cpp)
                IF(MPI_FOUND)
                    FIND_PACKAGE(MPI4PY)
                    IF(MPI4PY_FOUND)
                        TARGET_COMPILE_DEFINITIONS(hpdbscan-binding PRIVATE WITH_MPI)
                        FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/hpdbscan.py "from mpi4py import MPI\n")
                    ELSE()
                        MESSAGE("MPI FOUND, BUT MPI4PY MISSING, BINDING WILL BE BUILT WITHOUT MPI SUPPORT")
                    ENDIF()
                ENDIF()
                TARGET_LINK_LIBRARIES(hpdbscan-binding PRIVATE ${OpenMP_CXX_FLAGS})
                TARGET_COMPILE_OPTIONS(hpdbscan-binding PRIVATE ${OpenMP_CXX_FLAGS})
                TARGET_LINK_LIBRARIES(hpdbscan-binding PRIVATE "${HDF5_LIBRARIES}")
                
                # rename the library
                SET_TARGET_PROPERTIES(hpdbscan-binding PROPERTIES PREFIX "" OUTPUT_NAME _hpdbscan)
            ELSE()
		MESSAGE("NUMPY NOT FOUND, BUILDINGS WITHOUT BINDINGS")
	    ENDIF()
        ENDIF()
    ELSE()
        MESSAGE("PYTHON INTERPRETER NOT FOUND, BUILDING WITHOUT BINDINGS")
    ENDIF()
ELSE()
    MESSAGE("SWIG NOT FOUND, BUILDING WITHOUT BINDINGS")
ENDIF()

# rename the binary
SET_TARGET_PROPERTIES(hpdbscan-bin PROPERTIES OUTPUT_NAME hpdbscan)
