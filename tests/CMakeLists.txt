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

add_executable(test_channel_apply test_channel_apply.cpp)
target_link_libraries(test_channel_apply PRIVATE orca_common orca_dsp)
add_test(NAME channel_apply COMMAND test_channel_apply)

add_executable(test_combine test_combine.cpp)
target_link_libraries(test_combine PRIVATE orca_common orca_dsp)
add_test(NAME combine COMMAND test_combine)

add_executable(test_golden_pipeline test_golden_pipeline.cpp)
target_link_libraries(test_golden_pipeline PRIVATE orca_common orca_dsp)
add_test(NAME golden_pipeline COMMAND test_golden_pipeline)

add_executable(test_cir_table test_cir_table.cpp)
target_link_libraries(test_cir_table PRIVATE orca_common orca_channel)
add_test(NAME cir_table COMMAND test_cir_table)

# Spec G §G.8 Doppler handoff (header-only scenario math).
add_executable(test_doppler_handoff test_doppler_handoff.cpp)
target_link_libraries(test_doppler_handoff PRIVATE orca_common)
add_test(NAME doppler_handoff COMMAND test_doppler_handoff)

# Spec G §G.3 UE grid model (header-only scenario geometry).
add_executable(test_grid test_grid.cpp)
target_link_libraries(test_grid PRIVATE orca_common)
add_test(NAME grid COMMAND test_grid)

# Spec G §G.8 slow-plane channel refresh driver (grid → CIR lookup → expandLink).
add_executable(test_channel_refresh test_channel_refresh.cpp)
target_link_libraries(test_channel_refresh PRIVATE orca_common orca_channel)
add_test(NAME channel_refresh COMMAND test_channel_refresh)

# ADR 0002 §3/§4 serving-cell + interferer association / handover.
add_executable(test_association test_association.cpp)
target_link_libraries(test_association PRIVATE orca_common orca_channel)
add_test(NAME association COMMAND test_association)

# Scenario state container (grid + refresh + association + Doppler, on-move sync).
add_executable(test_scenario_state test_scenario_state.cpp)
target_link_libraries(test_scenario_state PRIVATE orca_common orca_channel)
add_test(NAME scenario_state COMMAND test_scenario_state)

# Integration: scenario slow-plane (ScenarioState) → dsp golden channel-apply.
add_executable(test_slow_plane_pipeline test_slow_plane_pipeline.cpp)
target_link_libraries(test_slow_plane_pipeline PRIVATE orca_common orca_channel orca_dsp)
add_test(NAME slow_plane_pipeline COMMAND test_slow_plane_pipeline)

# Cross-process IPC test (OruShm + VueShm, Linux shm backends).
# The test file compiles on any platform; the actual fork()-based IPC paths are
# guarded by #ifdef __linux__ so non-Linux builds pass trivially.
add_executable(test_ipc test_ipc.cpp)
target_link_libraries(test_ipc PRIVATE orca_common)
add_test(NAME ipc COMMAND test_ipc)

# The ORCA app's end-to-end smoke loop doubles as a test.
add_test(NAME orca_app_smoke COMMAND orca_app 280)
