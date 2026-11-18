cmake_minimum_required(VERSION 3.22)
project(motorhand_sharpa_control LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(NLopt CONFIG REQUIRED)
find_package(pinocchio CONFIG REQUIRED)
find_package(tinyxml2 CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(Python COMPONENTS Interpreter REQUIRED)
get_filename_component(CONTROL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(PROJECT_ROOT "${CONTROL_ROOT}/../../../.." ABSOLUTE)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import pathlib,mujoco; print(pathlib.Path(mujoco.__file__).parent)"
                OUTPUT_VARIABLE MUJOCO_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_library(MUJOCO_LIBRARY NAMES mujoco libmujoco.so.4.0.0 PATHS "${MUJOCO_ROOT}" NO_DEFAULT_PATH REQUIRED)
add_library(sharpa_control_core STATIC
  adaptive_retarget.cpp
  "${CONTROL_ROOT}/common/driver_impedance.cpp" "${CONTROL_ROOT}/common/force_jacobian.cpp"
  "${CONTROL_ROOT}/common/glove_output.cpp" "${CONTROL_ROOT}/common/inverse_retarget.cpp"
  "${CONTROL_ROOT}/common/inverse_worker.cpp" "${CONTROL_ROOT}/common/jacobian_worker.cpp"
  "${CONTROL_ROOT}/common/session.cpp" "${CONTROL_ROOT}/ve33_model.cpp" "${CONTROL_ROOT}/playback_control.cpp"
  "${CONTROL_ROOT}/encos_driver.cpp" "${CONTROL_ROOT}/simulation/fake_glove.cpp"
  "${CONTROL_ROOT}/simulation/sharpa_device.cpp" model.cpp device.cpp identity.cpp transport.cpp feedback.cpp)
target_include_directories(sharpa_control_core PUBLIC "${CONTROL_ROOT}" "${MUJOCO_ROOT}/include"
  "${PROJECT_ROOT}/../hardware/encos_g4_gateway_sdk/vendor/include")
target_link_libraries(sharpa_control_core PUBLIC Eigen3::Eigen NLopt::nlopt pinocchio::pinocchio_default
  pinocchio::pinocchio_parsers tinyxml2::tinyxml2 yaml-cpp::yaml-cpp nlohmann_json::nlohmann_json
  Threads::Threads "${MUJOCO_LIBRARY}")
option(MOTORHAND_SHARPA_HARDWARE "启用Sharpa官方SDK；关闭后仍可构建纯仿真" ON)
set(SHARPA_SDK_ROOT "${PROJECT_ROOT}/.cache/hardware-sdk/sharpa/5.0.11" CACHE PATH "Sharpa SDK 5.0.11")
if(MOTORHAND_SHARPA_HARDWARE)
  if(NOT EXISTS "${SHARPA_SDK_ROOT}/include/SharpaWaveSDK.h" OR NOT EXISTS "${SHARPA_SDK_ROOT}/lib/libsharpa-wave-sdk.so")
    message(FATAL_ERROR "缺少Sharpa SDK 5.0.11；执行tools/install_sharpa_sdk.py，或用-DMOTORHAND_SHARPA_HARDWARE=OFF构建纯仿真")
  endif()
  target_compile_definitions(sharpa_control_core PRIVATE MOTORHAND_WITH_SHARPA_SDK=1)
  target_include_directories(sharpa_control_core PRIVATE "${SHARPA_SDK_ROOT}/include")
  target_link_libraries(sharpa_control_core PUBLIC "${SHARPA_SDK_ROOT}/lib/libsharpa-wave-sdk.so")
endif()
add_executable(motorhand_sharpa main.cpp)
target_compile_definitions(motorhand_sharpa PRIVATE MOTORHAND_PROJECT_ROOT="${PROJECT_ROOT}")
target_link_libraries(motorhand_sharpa PRIVATE sharpa_control_core)
set_target_properties(motorhand_sharpa PROPERTIES BUILD_RPATH "${MUJOCO_ROOT};${SHARPA_SDK_ROOT}/lib")
add_executable(sharpa_retarget_probe retarget_probe.cpp)
target_compile_definitions(sharpa_retarget_probe PRIVATE MOTORHAND_PROJECT_ROOT="${PROJECT_ROOT}")
target_link_libraries(sharpa_retarget_probe PRIVATE sharpa_control_core)
enable_testing()
foreach(pair IN ITEMS "model" "device" "feedback" "simulation" "entry" "motion" "inverse")
  add_executable(sharpa_${pair}_native_test "${PROJECT_ROOT}/tests/test_sharpa_${pair}_native.cpp")
  target_link_libraries(sharpa_${pair}_native_test PRIVATE sharpa_control_core)
  target_compile_options(sharpa_${pair}_native_test PRIVATE -UNDEBUG)
  target_include_directories(sharpa_${pair}_native_test PRIVATE "${PROJECT_ROOT}/src")
  add_test(NAME sharpa_${pair}_native_test COMMAND sharpa_${pair}_native_test)
  set_tests_properties(sharpa_${pair}_native_test PROPERTIES WORKING_DIRECTORY "${PROJECT_ROOT}" TIMEOUT 60)
endforeach()
set_tests_properties(sharpa_inverse_native_test PROPERTIES WORKING_DIRECTORY "${PROJECT_ROOT}/..")
