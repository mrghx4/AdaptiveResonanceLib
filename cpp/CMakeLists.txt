cmake_minimum_required(VERSION 3.18)
project(artlib_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ARTLIB_CPP_BUILD_TESTS "Build C++ smoke tests" ON)

add_library(artlib_cpp_core STATIC
    src/art2_core.cpp
    src/bartmap_metrics_core.cpp
    src/cvi_metrics_core.cpp
    src/bayesian_art_core.cpp
    src/bayesian_artmap_core.cpp
    src/art1map_core.cpp
    src/binary_fuzzy_artmap_core.cpp
    src/binary_fuzzy_art_core.cpp
    src/backend_interface.cpp
    src/art1_core.cpp
    src/ellipsoid_art_core.cpp
    src/gaussian_art_core.cpp
    src/gaussian_artmap_core.cpp
    src/hypersphere_art_core.cpp
    src/hypersphere_artmap_core.cpp
    src/quadratic_neuron_art_core.cpp
    src/fuzzy_artmap_core.cpp
    src/fuzzy_art_core.cpp
    src/simple_artmap_core.cpp
    src/fusion_core.cpp
)

target_include_directories(artlib_cpp_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(ARTLIB_CPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
