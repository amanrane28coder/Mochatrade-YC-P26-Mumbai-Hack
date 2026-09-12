cmake_minimum_required(VERSION 3.18)
project(gpu_engine LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable/disable CUDA
option(USE_CUDA "Build with CUDA support" ON)

# If USE_CUDA is ON, try to find CUDA and add CUDA language
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    set(CUDA_SEPARABLE_COMPILATION ON)
    add_compile_definitions(USE_CUDA)
endif()

# Find Python (required by nanobind)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind - use Python to locate the cmake directory portably (Issue #10)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE nanobind_RESULT
)
if(NOT nanobind_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not find nanobind. Install it with: pip install nanobind")
endif()
find_package(nanobind REQUIRED)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Engine source files (Issue #18: bindings.cpp removed from static lib)
set(SOURCES
    src/engine.cpp
)

# Conditionally add CUDA sources
if(USE_CUDA)
    list(APPEND SOURCES src/kernels.cu)
endif()

# Build nanobind static library
nanobind_build_library(nanobind-static)

# Create static library from engine sources only
add_library(gpu_engine_static ${SOURCES})

# Link libraries to static library
if(USE_CUDA)
    target_link_libraries(gpu_engine_static PRIVATE ${CUDA_LIBRARIES})
endif()
# Link nanobind static library to our static library
target_link_libraries(gpu_engine_static PRIVATE nanobind-static)

# Create Python module using nanobind (bindings.cpp compiled only here)
nanobind_add_module(gpu_engine MODULE src/bindings.cpp)

# Set properties for the module
set_target_properties(gpu_engine PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Link the module with our static library
target_link_libraries(gpu_engine PRIVATE gpu_engine_static)
if(USE_CUDA)
    target_link_libraries(gpu_engine PRIVATE ${CUDA_LIBRARIES})
endif()

# Install
install(TARGETS gpu_engine DESTINATION .)