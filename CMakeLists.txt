cmake_minimum_required(VERSION 3.18)
project(povs CXX CUDA)

message(STATUS "#### Find Python ####")

set(PYTHON_VENV_BIN "${CMAKE_SOURCE_DIR}/.venv/bin/python" )

execute_process(
    COMMAND readlink -f ${PYTHON_VENV_BIN}
    OUTPUT_VARIABLE PYTHON_REAL_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

get_filename_component(PYTHON_BIN_DIR "${PYTHON_REAL_PATH}" DIRECTORY)
get_filename_component(Python3_ROOT_DIR "${PYTHON_BIN_DIR}" DIRECTORY)
find_package(Python3 COMPONENTS Development REQUIRED)

message(STATUS "#### Find CUDA ####")

find_package(CUDAToolkit)

# Detect CUDA architecture from device 0 (if not manually specified)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURE OR CMAKE_CUDA_ARCHITECTURE STREQUAL "")
    execute_process(
            COMMAND ${PYTHON_VENV_BIN} -c "import torch; print(torch.cuda.get_device_capability(0)[0] * 10 + torch.cuda.get_device_capability(0)[1])"
            OUTPUT_VARIABLE CUDA_ARCH_DETECTED
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
    )

    if(CUDA_ARCH_DETECTED AND NOT CUDA_ARCH_DETECTED STREQUAL "")
        message(STATUS "Detected CUDA architecture from device 0: ${CUDA_ARCH_DETECTED}")
        set(CMAKE_CUDA_ARCHITECTURE ${CUDA_ARCH_DETECTED})
    else()
        message(STATUS "Could not detect CUDA architecture, defaulting to 75 (Turing)")
        set(CMAKE_CUDA_ARCHITECTURE 75)
    endif()
else()
    message(STATUS "Using manually specified CUDA architecture: ${CMAKE_CUDA_ARCHITECTURE}")
endif()

message(STATUS "#### Find CUTLASS ####")

execute_process(
        COMMAND ${PYTHON_VENV_BIN} -c "import cutlass_library; import os; print(os.path.join(os.path.dirname(cutlass_library.__file__), 'source', 'include'))"
        OUTPUT_VARIABLE CUTLASS_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
)
message(STATUS "CUTLASS_INCLUDE_DIR: ${CUTLASS_INCLUDE_DIR}")

message(STATUS "#### Find PyTorch ####")

execute_process(
    COMMAND ${PYTHON_VENV_BIN} -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "TORCH_PATH: ${TORCH_PATH}")

execute_process(
    COMMAND ${PYTHON_VENV_BIN} -c "import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.include_paths()))"
    OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")

execute_process(
    COMMAND ${PYTHON_VENV_BIN} -c "import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.library_paths()))"
    OUTPUT_VARIABLE TORCH_LIBRARY_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "TORCH_LIBRARY_DIRS: ${TORCH_LIBRARY_DIRS}")

message(STATUS "#### Configure POVS.cpp library ####")

add_compile_definitions(STANDALONE_BUILD)

file(GLOB_RECURSE SOURCES
        src/povs/__cuda/*.cpp
        src/povs/__cuda/*.cu
)
add_library(povs_cpp SHARED ${SOURCES})

set_property(TARGET povs_cpp PROPERTY CXX_STANDARD 20)
set_property(TARGET povs_cpp PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET povs_cpp PROPERTY CUDA_STANDARD 17)
set_property(TARGET povs_cpp PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURE})

target_include_directories(povs_cpp PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUTLASS_INCLUDE_DIR}
        src/povs/__cuda
)

target_link_directories(povs_cpp PRIVATE ${TORCH_LIBRARY_DIRS})

target_link_libraries(povs_cpp PRIVATE
    ${Python3_LIBRARIES}
    torch
    torch_cpu
    c10
    torch_cuda
    CUDA::cudart
)

target_compile_options(povs_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>
)
