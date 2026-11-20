cmake_minimum_required(VERSION 3.20)
project(tier1_autocall_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tier1_autocall_cpu SHARED tier1_autocall_cpu.cpp)
set_target_properties(tier1_autocall_cpu PROPERTIES
    PREFIX ""
    OUTPUT_NAME "tier1_autocall_cpu"
)

target_compile_options(tier1_autocall_cpu PRIVATE
    -O3
    -march=native
    -fno-fast-math
    -ffp-contract=off
    -Wall
    -Wextra
)

install(TARGETS tier1_autocall_cpu
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
