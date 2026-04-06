cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
find_package(
  Python 3.8
  COMPONENTS Interpreter Development.Module
  REQUIRED)

# Find nanobind
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Find MLX
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# ----------------------------- C++ Library -----------------------------

add_library(turboquant_ext)

target_sources(turboquant_ext PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/polar_ops.cpp)

target_include_directories(turboquant_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(turboquant_ext PUBLIC mlx)

# ----------------------------- Metal Kernels -----------------------------

if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET turboquant_ext_metallib
    TITLE turboquant_ext
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/polar_kernels.metal
    INCLUDE_DIRS
      ${PROJECT_SOURCE_DIR}
      ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(turboquant_ext turboquant_ext_metallib)
endif()

# ----------------------------- Python Bindings -----------------------------

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)

target_link_libraries(_ext PRIVATE turboquant_ext)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
