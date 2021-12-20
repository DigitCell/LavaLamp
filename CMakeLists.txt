cmake_minimum_required(VERSION 3.5)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
  message(CMAKE_CUDA_ARCHITECTURES)
endif()

project(SPH-ishii LANGUAGES CXX)

project(Cuda_particles LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(EXECUTABLE_NAME "particles")
set(homeDir "/home/geotyper/CPP_SPH")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL2 2.0.17 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

set(IMGUI_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/imgui)
include_directories(${IMGUI_INCLUDE_DIR})
#set(IMGUI_LIBRARIES imgui)

SET(TARGET_H

    structs.h

    mainloop.hpp
    graphmodule.hpp
    Constants.hpp
)

SET(TARGET_SRC


    mainloop.cpp
    graphmodule.cpp
    main.cpp
)

SET(TARGET_CUDA_H
    cuda_solver.cuh
    support/CudaHelper.cuh

)

SET(TARGET_CUDA_SRC
    cuda_solver.cu

)



find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
link_libraries(${GLEW_LIBRARIES})


find_package(CUDA REQUIRED)
set(Cuda_common_include "/usr/local/cuda/samples/common/inc")
include_directories(${Cuda_common_include})

message(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})



add_executable(${EXECUTABLE_NAME}  ${TARGET_CUDA_H} ${TARGET_CUDA_SRC} ${TARGET_H} ${TARGET_SRC} )


set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_ARCHITECTURES "61")
target_link_libraries(${EXECUTABLE_NAME} ${CUDA_LIBRARIES})

target_link_libraries(${EXECUTABLE_NAME}  -lGL ${SDL2_LIBRARIES})
target_link_libraries(${EXECUTABLE_NAME}   ${CMAKE_SOURCE_DIR}/libimgui.a)
