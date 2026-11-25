cmake_minimum_required(VERSION 3.20)
project(tier1_autocall_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DEEPVOL_NATIVE_ARCH "Optimize for the build host CPU instead of portable AVX2+FMA+BMI2" OFF)

# Discover GCC's omp.h and libgomp when Clang is built without standalone libomp.
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_program(_GCC_EXECUTABLE NAMES gcc g++)
    find_library(_GOMP_LIBRARY NAMES gomp)
    if(_GCC_EXECUTABLE AND _GOMP_LIBRARY)
        execute_process(
            COMMAND "${_GCC_EXECUTABLE}" -print-file-name=include/omp.h
            OUTPUT_VARIABLE _GCC_OMP_HEADER
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(EXISTS "${_GCC_OMP_HEADER}")
            get_filename_component(_GCC_OMP_INCLUDE_DIR "${_GCC_OMP_HEADER}" DIRECTORY)
            set(OpenMP_CXX_FLAGS "-fopenmp=libgomp -idirafter ${_GCC_OMP_INCLUDE_DIR}" CACHE STRING "" FORCE)
            set(OpenMP_CXX_LIB_NAMES "gomp" CACHE STRING "" FORCE)
            set(OpenMP_gomp_LIBRARY "${_GOMP_LIBRARY}" CACHE FILEPATH "" FORCE)
        endif()
    endif()
endif()
find_package(OpenMP REQUIRED)

add_library(tier1_autocall_cpu SHARED tier1_autocall_cpu.cpp)
set_target_properties(tier1_autocall_cpu PROPERTIES
    PREFIX ""
    OUTPUT_NAME "tier1_autocall_cpu"
)

if(DEEPVOL_NATIVE_ARCH)
    set(_DEEPVOL_ARCH_FLAGS -march=native -mavx2 -mfma)
else()
    set(_DEEPVOL_ARCH_FLAGS -mavx2 -mfma -mbmi2)
endif()

target_compile_options(tier1_autocall_cpu PRIVATE
    -O3
    ${_DEEPVOL_ARCH_FLAGS}
    -fno-fast-math
    -fno-math-errno
    -ffp-contract=off
    -Wall
    -Wextra
    -Werror
    -Wsign-conversion
)

target_link_libraries(tier1_autocall_cpu PRIVATE OpenMP::OpenMP_CXX)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../models/autocall" CACHE PATH "Default install directory" FORCE)
endif()

install(TARGETS tier1_autocall_cpu
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
