add_executable(test_layout test_layout.cpp)
target_link_libraries(test_layout PRIVATE orca_common)
add_test(NAME layout COMMAND test_layout)

add_executable(test_timing test_timing.cpp)
target_link_libraries(test_timing PRIVATE orca_common)
add_test(NAME timing COMMAND test_timing)

add_executable(test_ring test_ring.cpp)
target_link_libraries(test_ring PRIVATE orca_common)
add_test(NAME ring COMMAND test_ring)
