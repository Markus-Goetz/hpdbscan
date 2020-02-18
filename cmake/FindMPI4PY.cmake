# modified from https://github.com/live-clones/xdmf/blob/master/CMake/FindMPI4PY.cmake

IF(NOT MPI4PY_INCLUDE_DIR)
    EXECUTE_PROCESS(COMMAND
      "${PYTHON_EXECUTABLE}" "-c" "import mpi4py; print(mpi4py.get_include())"
      OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
      RESULT_VARIABLE MPI4PY_COMMAND_RESULT
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    IF(MPI4PY_COMMAND_RESULT)
        MESSAGE("mpi4py not found")
        SET(MPI4PY_FOUND FALSE)
    ELSE()
        IF(MPI4PY_INCLUDE_DIR MATCHES "Traceback")
            MESSAGE("mpi4py matches traceback")
            ## Did not successfully include MPI4PY
            SET(MPI4PY_FOUND FALSE)
        ELSE()
            ## successful
            SET(MPI4PY_FOUND TRUE)
            SET(MPI4PY_INCLUDE_DIR ${MPI4PY_INCLUDE_DIR} CACHE STRING "mpi4py include path")
        ENDIF()
    ENDIF()
ELSE()
    SET(MPI4PY_FOUND TRUE)
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MPI4PY DEFAULT_MSG MPI4PY_INCLUDE_DIR)
