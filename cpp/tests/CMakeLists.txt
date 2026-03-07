add_executable(backend_interface_smoke
    test_backend_interface_smoke.cpp
)

target_link_libraries(backend_interface_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME backend_interface_smoke
    COMMAND backend_interface_smoke
)

add_executable(fuzzy_art_core_smoke
    test_fuzzy_art_core_smoke.cpp
)

target_link_libraries(fuzzy_art_core_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME fuzzy_art_core_smoke
    COMMAND fuzzy_art_core_smoke
)

add_executable(art1_core_smoke
    test_art1_core_smoke.cpp
)

target_link_libraries(art1_core_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME art1_core_smoke
    COMMAND art1_core_smoke
)

add_executable(art1map_core_smoke
    test_art1map_core_smoke.cpp
)

target_link_libraries(art1map_core_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME art1map_core_smoke
    COMMAND art1map_core_smoke
)

add_executable(binary_fuzzy_art_core_smoke
    test_binary_fuzzy_art_core_smoke.cpp
)

target_link_libraries(binary_fuzzy_art_core_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME binary_fuzzy_art_core_smoke
    COMMAND binary_fuzzy_art_core_smoke
)

add_executable(binary_fuzzy_artmap_core_smoke
    test_binary_fuzzy_artmap_core_smoke.cpp
)

target_link_libraries(binary_fuzzy_artmap_core_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME binary_fuzzy_artmap_core_smoke
    COMMAND binary_fuzzy_artmap_core_smoke
)

add_executable(fuzzy_artmap_core_smoke
    test_fuzzy_artmap_core_smoke.cpp
)

target_link_libraries(fuzzy_artmap_core_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME fuzzy_artmap_core_smoke
    COMMAND fuzzy_artmap_core_smoke
)

add_executable(gaussian_artmap_core_smoke
    test_gaussian_artmap_core_smoke.cpp
)

target_link_libraries(gaussian_artmap_core_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME gaussian_artmap_core_smoke
    COMMAND gaussian_artmap_core_smoke
)

add_executable(gaussian_art_core_smoke
    test_gaussian_art_core_smoke.cpp
)

target_link_libraries(gaussian_art_core_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME gaussian_art_core_smoke
    COMMAND gaussian_art_core_smoke
)

add_executable(hypersphere_art_core_smoke
    test_hypersphere_art_core_smoke.cpp
)

target_link_libraries(hypersphere_art_core_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME hypersphere_art_core_smoke
    COMMAND hypersphere_art_core_smoke
)

add_executable(ellipsoid_art_core_smoke
    test_ellipsoid_art_core_smoke.cpp
)

target_link_libraries(ellipsoid_art_core_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME ellipsoid_art_core_smoke
    COMMAND ellipsoid_art_core_smoke
)

add_executable(bayesian_art_core_smoke
    test_bayesian_art_core_smoke.cpp
)

target_link_libraries(bayesian_art_core_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME bayesian_art_core_smoke
    COMMAND bayesian_art_core_smoke
)

add_executable(bayesian_artmap_core_smoke
    test_bayesian_artmap_core_smoke.cpp
)

target_link_libraries(bayesian_artmap_core_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME bayesian_artmap_core_smoke
    COMMAND bayesian_artmap_core_smoke
)

add_executable(hypersphere_artmap_core_smoke
    test_hypersphere_artmap_core_smoke.cpp
)

target_link_libraries(hypersphere_artmap_core_smoke
    PRIVATE artlib_cpp_core
)

add_test(
    NAME hypersphere_artmap_core_smoke
    COMMAND hypersphere_artmap_core_smoke
)
