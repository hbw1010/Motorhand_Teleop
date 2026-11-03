cmake_minimum_required(VERSION 3.16)
project(real_hand LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)
option(REAL_HAND_BUILD_SDK "构建真实官方SDK适配器" ON)
option(REAL_HAND_BUILD_TESTS "构建不连接设备的离线回归与C ABI冒烟" OFF)
set(WUJI_SDK_ROOT "" CACHE PATH "官方C SDK 2026.8.31根目录")
get_filename_component(_project_root "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(_project_cache "${_project_root}/.cache")
set(_sdk_cache "${_project_cache}/wuji-sdk-c-2026.8.31-x86_64-linux-gnu")
if(NOT WUJI_SDK_ROOT AND EXISTS "${_sdk_cache}/include/wuji_sdk.h")
  set(WUJI_SDK_ROOT "${_sdk_cache}")
endif()

if(REAL_HAND_BUILD_SDK)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
     NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$" OR
     NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "固定SDK二进制要求GNU/Linux x86_64")
  endif()
  if(NOT WUJI_SDK_ROOT)
    include(FetchContent)
    if(POLICY CMP0135)
      cmake_policy(SET CMP0135 NEW)
    endif()
    FetchContent_Declare(wuji_sdk
      SOURCE_DIR "${_sdk_cache}"
      URL "https://github.com/wuji-technology/wuji-sdk/releases/download/v2026.8.31/wuji-sdk-c-2026.8.31-x86_64-linux-gnu.tar.gz"
      URL_HASH SHA256=c7292de6377a266c8eb151560114dadb6f1fc3f1003c10a5ddffcf29879e3ba7)
    FetchContent_MakeAvailable(wuji_sdk)
    set(WUJI_SDK_ROOT "${wuji_sdk_SOURCE_DIR}")
  endif()
  get_filename_component(WUJI_SDK_ROOT "${WUJI_SDK_ROOT}" ABSOLUTE)
  # SDK及RPATH必须属于本项目，不能意外复用参考工程或全局缓存。
  get_filename_component(_sdk_real "${WUJI_SDK_ROOT}" REALPATH)
  file(RELATIVE_PATH _sdk_relative "${_project_cache}" "${_sdk_real}")
  if(_sdk_relative MATCHES "^\\.\\." OR IS_ABSOLUTE "${_sdk_relative}")
    message(FATAL_ERROR "WUJI_SDK_ROOT必须位于本项目.cache内；请使用默认固定版本下载")
  endif()
  if(NOT EXISTS "${WUJI_SDK_ROOT}/include/wuji_sdk.h" OR
     NOT EXISTS "${WUJI_SDK_ROOT}/lib/libwuji_sdk_c.so")
    message(FATAL_ERROR "WUJI_SDK_ROOT必须包含2026.8.31的include/wuji_sdk.h和lib/libwuji_sdk_c.so")
  endif()
  add_library(wuji_sdk_c SHARED IMPORTED)
  set_target_properties(wuji_sdk_c PROPERTIES
    IMPORTED_LOCATION "${WUJI_SDK_ROOT}/lib/libwuji_sdk_c.so"
    INTERFACE_INCLUDE_DIRECTORIES "${WUJI_SDK_ROOT}/include")
  add_library(real_hand SHARED real_hand.cpp sdk_transport.cpp)
  target_link_libraries(real_hand PRIVATE wuji_sdk_c Threads::Threads)
  target_include_directories(real_hand PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  set_target_properties(real_hand PROPERTIES
    BUILD_RPATH "${WUJI_SDK_ROOT}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${_project_cache}/native-real-hand/build")
  if(REAL_HAND_BUILD_TESTS)
    add_executable(real_hand_c_smoke tests/c_smoke.c)
    # 源码仍由C编译器检查；最终链接需要C++运行库和官方Linux SDK的udev依赖。
    # Pixi工具链sysroot不包含宿主udev，明确找运行该固定SDK的宿主库。
    find_library(WUJI_UDEV_LIBRARY NAMES udev libudev.so.1 NO_CMAKE_FIND_ROOT_PATH)
    if(NOT WUJI_UDEV_LIBRARY)
      message(FATAL_ERROR "官方Linux SDK需要宿主libudev；请安装系统libudev开发库")
    endif()
    set_target_properties(real_hand_c_smoke PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(real_hand_c_smoke PRIVATE real_hand "${WUJI_UDEV_LIBRARY}")
  endif()
endif()

if(REAL_HAND_BUILD_TESTS)
  enable_testing()
  if(TARGET real_hand_c_smoke)
    add_test(NAME real_hand_c_smoke COMMAND real_hand_c_smoke)
  endif()
  # 与生产完全相同的控制器和C ABI，只替换运输边界；不链接官方SDK。
  add_executable(real_hand_offline real_hand.cpp tests/offline.cpp)
  target_include_directories(real_hand_offline PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(real_hand_offline PRIVATE Threads::Threads)
  add_test(NAME real_hand_offline COMMAND real_hand_offline)
  set_tests_properties(real_hand_offline PROPERTIES TIMEOUT 30)
  if(NOT EXISTS "${WUJI_SDK_ROOT}/include/wuji_sdk.h")
    message(FATAL_ERROR "SDK适配器离线回归需要固定SDK头文件，请设置WUJI_SDK_ROOT；不加载或链接SDK库")
  endif()
  # 仅测试产物：生产core及SDKAdapter，替换C SDK符号，不链接任何真实SDK。
  # 不安装、不供运行时自动选择；ctypes集成smoke须显式指定此文件。
  add_library(real_hand_sdk_offline_library SHARED real_hand.cpp sdk_transport.cpp tests/fake_sdk.cpp)
  set_target_properties(real_hand_sdk_offline_library PROPERTIES OUTPUT_NAME real_hand_sdk_offline)
  target_include_directories(real_hand_sdk_offline_library PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${WUJI_SDK_ROOT}/include")
  target_link_libraries(real_hand_sdk_offline_library PRIVATE Threads::Threads)
  add_executable(real_hand_sdk_offline tests/sdk_offline.cpp)
  target_include_directories(real_hand_sdk_offline PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(real_hand_sdk_offline PRIVATE real_hand_sdk_offline_library Threads::Threads)
  add_test(NAME real_hand_sdk_offline COMMAND real_hand_sdk_offline)
  set_tests_properties(real_hand_sdk_offline PROPERTIES TIMEOUT 30)
endif()
