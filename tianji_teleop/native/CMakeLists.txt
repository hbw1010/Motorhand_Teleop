# motorhand_glove_native: the Motorhand glove worker for Tianji's Wuji Hand 2 (tianji_teleop/README.md), in C++, from
# this repository's own C++ (src/motorhand_teleop/real_hand/native_control): the glove's CAN driver, the vE3.3 model,
# the WujiRetarget, its inverse and the force feedback. Not its session, nor the Wuji SDK: Tianji's driver owns the
# Wuji hand.
#   bash tianji_teleop/scripts/build_motorhand_glove.sh      (in this checkout's Pixi env, which has these libraries)
# On simulated CAN adapters (the driver's offline mode; the worker's --test-ports), in a build dir of its own:
#   -DCMAKE_CXX_FLAGS=-DMOTORHAND_ENCOS_OFFLINE_TEST
cmake_minimum_required(VERSION 3.22)
project(motorhand_glove_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(DEFAULT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(MOTORHAND_ROOT "${DEFAULT_ROOT}" CACHE PATH "The Motorhand_Teleop checkout (default: this one)")
set(NC "${MOTORHAND_ROOT}/src/motorhand_teleop/real_hand/native_control")
if(NOT EXISTS "${NC}/encos_driver.cpp")
  message(FATAL_ERROR "no Motorhand C++ at ${NC}")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(NLopt CONFIG REQUIRED)
find_package(pinocchio CONFIG REQUIRED)
find_package(tinyxml2 CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Threads REQUIRED)

# The settings this worker uses are the driver's and the retarget's own (EncosDriver::set_slot, set_position_kp,
# mark_homed; the retarget yaml's pinch_thresholds and pinch_weight_max): their sources compiled as they are.
add_library(motorhand_glove_core STATIC
  "${NC}/encos_driver.cpp"
  "${NC}/ve33_model.cpp"
  "${NC}/wuji_retarget.cpp"
  "${NC}/force_controller.cpp"
  "${NC}/force_jacobian.cpp"
  "${NC}/jacobian_worker.cpp"
  "${NC}/linear_force.cpp"
  "${NC}/glove_output.cpp"
  "${NC}/inverse_retarget.cpp"
  "${NC}/inverse_worker.cpp")
target_include_directories(motorhand_glove_core PUBLIC "${NC}")
target_link_libraries(motorhand_glove_core PUBLIC
  Eigen3::Eigen NLopt::nlopt pinocchio::pinocchio_default pinocchio::pinocchio_parsers
  tinyxml2::tinyxml2 yaml-cpp::yaml-cpp nlohmann_json::nlohmann_json Threads::Threads)

add_executable(motorhand_glove_bench bench.cpp)
target_link_libraries(motorhand_glove_bench PRIVATE motorhand_glove_core)

add_executable(motorhand_glove_native main.cpp glove_control.cpp)
target_link_libraries(motorhand_glove_native PRIVATE motorhand_glove_core)
