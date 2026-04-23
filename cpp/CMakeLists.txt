cmake_minimum_required(VERSION 3.18)
project(artlib_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ARTLIB_CPP_BUILD_TESTS "Build C++ smoke/parity tests" ON)
option(ARTLIB_CPP_BUILD_FUZZ "Build bounded C++ fuzz-smoke tests" OFF)
option(ARTLIB_CPP_ENABLE_ASAN "Build with AddressSanitizer" OFF)
option(ARTLIB_CPP_ENABLE_UBSAN "Build with UndefinedBehaviorSanitizer" OFF)
option(ARTLIB_CPP_ENABLE_COVERAGE "Build with gcov-compatible coverage flags" OFF)

set(ARTLIB_CPP_COMPILE_OPTIONS "")
set(ARTLIB_CPP_LINK_OPTIONS "")

if(MSVC)
    if(ARTLIB_CPP_ENABLE_ASAN OR ARTLIB_CPP_ENABLE_UBSAN OR ARTLIB_CPP_ENABLE_COVERAGE)
        message(WARNING "ARTLib C++ sanitizer/coverage flags are currently configured for GCC/Clang builds only.")
    endif()
else()
    if(ARTLIB_CPP_ENABLE_ASAN)
        list(APPEND ARTLIB_CPP_COMPILE_OPTIONS -fsanitize=address -fno-omit-frame-pointer -g)
        list(APPEND ARTLIB_CPP_LINK_OPTIONS -fsanitize=address)
    endif()
    if(ARTLIB_CPP_ENABLE_UBSAN)
        list(APPEND ARTLIB_CPP_COMPILE_OPTIONS -fsanitize=undefined -fno-omit-frame-pointer -g)
        list(APPEND ARTLIB_CPP_LINK_OPTIONS -fsanitize=undefined)
    endif()
    if(ARTLIB_CPP_ENABLE_COVERAGE)
        list(APPEND ARTLIB_CPP_COMPILE_OPTIONS --coverage -O0 -g)
        list(APPEND ARTLIB_CPP_LINK_OPTIONS --coverage)
    endif()
endif()

function(artlib_cpp_apply_hardening target)
    if(ARTLIB_CPP_COMPILE_OPTIONS)
        target_compile_options(${target} PRIVATE ${ARTLIB_CPP_COMPILE_OPTIONS})
    endif()

    get_target_property(target_type ${target} TYPE)
    if(ARTLIB_CPP_LINK_OPTIONS AND NOT target_type STREQUAL "STATIC_LIBRARY")
        target_link_options(${target} PRIVATE ${ARTLIB_CPP_LINK_OPTIONS})
    endif()
endfunction()

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

artlib_cpp_apply_hardening(artlib_cpp_core)

target_include_directories(artlib_cpp_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(ARTLIB_CPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
