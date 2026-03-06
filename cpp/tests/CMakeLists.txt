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
