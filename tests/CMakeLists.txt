add_executable(test_layout test_layout.cpp)
target_link_libraries(test_layout PRIVATE orca_common)
add_test(NAME layout COMMAND test_layout)

add_executable(test_timing test_timing.cpp)
target_link_libraries(test_timing PRIVATE orca_common)
add_test(NAME timing COMMAND test_timing)

add_executable(test_ring test_ring.cpp)
target_link_libraries(test_ring PRIVATE orca_common)
add_test(NAME ring COMMAND test_ring)

add_executable(test_transport_loopback test_transport_loopback.cpp)
target_link_libraries(test_transport_loopback PRIVATE orca_common)
add_test(NAME transport_loopback COMMAND test_transport_loopback)

add_executable(test_fh test_fh.cpp)
target_link_libraries(test_fh PRIVATE orca_common)
add_test(NAME fh COMMAND test_fh)

add_executable(test_oru_process test_oru_process.cpp)
target_link_libraries(test_oru_process PRIVATE orca_common)
add_test(NAME oru_process COMMAND test_oru_process)

# The standalone program's smoke loop doubles as a test.
add_test(NAME oru_process_smoke COMMAND oru_process 28)

add_executable(test_e2e_identity test_e2e_identity.cpp)
target_link_libraries(test_e2e_identity PRIVATE orca_common orca_dsp)
add_test(NAME e2e_identity COMMAND test_e2e_identity)

add_executable(test_philox test_philox.cpp)
target_link_libraries(test_philox PRIVATE orca_common orca_dsp)
add_test(NAME philox COMMAND test_philox)

add_executable(test_precode test_precode.cpp)
target_link_libraries(test_precode PRIVATE orca_common orca_dsp)
add_test(NAME precode COMMAND test_precode)

# The ORCA app's end-to-end smoke loop doubles as a test.
add_test(NAME orca_app_smoke COMMAND orca_app 280)
