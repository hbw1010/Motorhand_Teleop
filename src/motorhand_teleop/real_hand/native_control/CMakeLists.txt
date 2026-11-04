cmake_minimum_required(VERSION 3.22)
project(motorhand_native_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(NLopt CONFIG REQUIRED)
find_package(pinocchio CONFIG REQUIRED)
find_package(tinyxml2 CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

add_library(motorhand_native_core STATIC
  force_controller.cpp
  force_jacobian.cpp
  glove_output.cpp
  inverse_retarget.cpp
  inverse_worker.cpp
  jacobian_worker.cpp
  linear_force.cpp
  playback_control.cpp
  session.cpp
  telemetry.cpp
  trajectory.cpp
  ve33_model.cpp
  wuji_retarget.cpp)
target_include_directories(motorhand_native_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(motorhand_native_core PUBLIC
  Eigen3::Eigen NLopt::nlopt pinocchio::pinocchio_default pinocchio::pinocchio_parsers
  tinyxml2::tinyxml2 yaml-cpp::yaml-cpp nlohmann_json::nlohmann_json OpenSSL::Crypto Threads::Threads)

add_library(motorhand_native_hardware STATIC encos_driver.cpp wuji_device.cpp)
target_include_directories(motorhand_native_hardware PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(motorhand_native_hardware PUBLIC nlohmann_json::nlohmann_json Threads::Threads)

# Read-only device discovery and firmware verification use the pinned C SDK.
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(WUJI_SDK_ROOT "${PROJECT_ROOT}/.cache/wuji-sdk-c-2026.8.31-x86_64-linux-gnu" CACHE PATH "Pinned Wuji C SDK")
if(NOT EXISTS "${WUJI_SDK_ROOT}/include/wuji_sdk.h" OR NOT EXISTS "${WUJI_SDK_ROOT}/lib/libwuji_sdk_c.so")
  message(FATAL_ERROR "缺少本项目固定版本 Wuji C SDK；先执行 pixi run -e hardware build-real-hand")
endif()
add_library(wuji_sdk_c SHARED IMPORTED)
set_target_properties(wuji_sdk_c PROPERTIES
  IMPORTED_LOCATION "${WUJI_SDK_ROOT}/lib/libwuji_sdk_c.so"
  INTERFACE_INCLUDE_DIRECTORIES "${WUJI_SDK_ROOT}/include")
add_library(motorhand_native_identity STATIC wuji_identity.cpp)
target_include_directories(motorhand_native_identity PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(motorhand_native_identity PUBLIC wuji_sdk_c)

add_library(motorhand_wuji_transport STATIC ../native/real_hand.cpp ../native/sdk_transport.cpp)
target_include_directories(motorhand_wuji_transport PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../native")
target_link_libraries(motorhand_wuji_transport PUBLIC wuji_sdk_c Threads::Threads)
add_library(motorhand_native_control INTERFACE)
target_link_libraries(motorhand_native_control INTERFACE
  motorhand_native_core motorhand_native_hardware motorhand_native_identity motorhand_wuji_transport)

find_library(WUJI_UDEV_LIBRARY NAMES udev libudev.so.1
  PATHS "/lib/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux-gnu" "/usr/lib/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux-gnu"
  NO_CMAKE_FIND_ROOT_PATH REQUIRED)
add_executable(motorhand_native main.cpp)
target_link_libraries(motorhand_native PRIVATE motorhand_native_control "${WUJI_UDEV_LIBRARY}")
set_target_properties(motorhand_native PROPERTIES BUILD_RPATH "${WUJI_SDK_ROOT}/lib")
add_library(inference_bridge SHARED inference_bridge.cpp)
target_link_libraries(inference_bridge PRIVATE motorhand_native_control "${WUJI_UDEV_LIBRARY}")
set_target_properties(inference_bridge PROPERTIES BUILD_RPATH "${WUJI_SDK_ROOT}/lib")

option(MOTORHAND_NATIVE_TESTS "Build offline native control regression tests" OFF)
if(MOTORHAND_NATIVE_TESTS)
  enable_testing()
  add_executable(encos_native_test "${PROJECT_ROOT}/tests/test_encos_driver_native.cpp" encos_driver.cpp)
  target_compile_definitions(encos_native_test PRIVATE MOTORHAND_ENCOS_OFFLINE_TEST)
  target_include_directories(encos_native_test PRIVATE "${PROJECT_ROOT}/src" "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(encos_native_test PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
  add_test(NAME encos_native_test COMMAND encos_native_test)
  set_tests_properties(encos_native_test PROPERTIES TIMEOUT 30)
  add_executable(playback_native_test "${PROJECT_ROOT}/tests/test_playback_control_native.cpp" playback_control.cpp)
  target_include_directories(playback_native_test PRIVATE "${PROJECT_ROOT}/src" "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(playback_native_test PRIVATE Threads::Threads)
  add_test(NAME playback_native_test COMMAND playback_native_test)
  set_tests_properties(playback_native_test PROPERTIES TIMEOUT 30)
  add_executable(trajectory_native_test "${PROJECT_ROOT}/tests/test_trajectory_native.cpp" trajectory.cpp)
  target_include_directories(trajectory_native_test PRIVATE "${PROJECT_ROOT}/src" "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(trajectory_native_test PRIVATE nlohmann_json::nlohmann_json OpenSSL::Crypto Threads::Threads)
  add_test(NAME trajectory_native_test COMMAND trajectory_native_test)
  set_tests_properties(trajectory_native_test PROPERTIES TIMEOUT 30)
  add_executable(glove_output_native_test "${PROJECT_ROOT}/tests/test_glove_output_native.cpp" glove_output.cpp)
  target_include_directories(glove_output_native_test PRIVATE "${PROJECT_ROOT}/src" "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(glove_output_native_test PRIVATE Threads::Threads)
  add_test(NAME glove_output_native_test COMMAND glove_output_native_test)
  set_tests_properties(glove_output_native_test PROPERTIES TIMEOUT 30)
  add_executable(session_native_test "${PROJECT_ROOT}/tests/test_session_native.cpp")
  target_include_directories(session_native_test PRIVATE "${PROJECT_ROOT}/src")
  target_link_libraries(session_native_test PRIVATE motorhand_native_core)
  add_test(NAME session_native_test COMMAND session_native_test)
  set_tests_properties(session_native_test PROPERTIES WORKING_DIRECTORY "${PROJECT_ROOT}" TIMEOUT 60)
  add_executable(inference_bridge_native_test "${PROJECT_ROOT}/tests/test_inference_bridge_native.cpp" inference_bridge.cpp)
  target_include_directories(inference_bridge_native_test PRIVATE "${PROJECT_ROOT}/src")
  target_link_libraries(inference_bridge_native_test PRIVATE motorhand_native_core)
  add_test(NAME inference_bridge_native_test COMMAND inference_bridge_native_test)
  set_tests_properties(inference_bridge_native_test PROPERTIES WORKING_DIRECTORY "${PROJECT_ROOT}" TIMEOUT 30)
  add_executable(linear_force_native_test "${PROJECT_ROOT}/tests/test_linear_force_native.cpp" linear_force.cpp)
  target_include_directories(linear_force_native_test PRIVATE "${PROJECT_ROOT}/src")
  target_link_libraries(linear_force_native_test PRIVATE Eigen3::Eigen)
  add_test(NAME linear_force_native_test COMMAND linear_force_native_test)
  set_tests_properties(linear_force_native_test PROPERTIES TIMEOUT 30)
endif()
