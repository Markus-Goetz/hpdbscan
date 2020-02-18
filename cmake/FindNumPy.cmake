# modified from https://github.com/live-clones/xdmf/blob/master/CMake/FindMPI4PY.cmake

IF(NOT NUMPY_INCLUDE_DIR)
    EXECUTE_PROCESS(COMMAND
      "${PYTHON_EXECUTABLE}" "-c" "import numpy as np; print(np.get_include())"
      OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
      RESULT_VARIABLE NUMPY_COMMAND_RESULT
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    IF(NUMPY_COMMAND_RESULT)
        MESSAGE("numpy not found")
	SET(NUMPY_FOUND FALSE)
    ELSE()
	    IF(NUMPY_INCLUDE_DIR MATCHES "Traceback")
            MESSAGE("numpy matches traceback")
	    ## Did not successfully include NUMPY
	    SET(NUMPY_FOUND FALSE)
        ELSE()
            ## successful
	    SET(NUMPY_FOUND TRUE)
	    SET(NUMPY_INCLUDE_DIR ${NUMPY_INCLUDE_DIR} CACHE STRING "numpy include path")
        ENDIF()
    ENDIF()
ELSE()
	SET(NUMPY_FOUND TRUE)
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(NUMPY DEFAULT_MSG NUMPY_INCLUDE_DIR)
