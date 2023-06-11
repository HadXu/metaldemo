cmake_minimum_required(VERSION 3.25)
project(metaldemo)

set(CMAKE_CXX_STANDARD 17)

find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(METALPERFORMANCE_FRAMEWORK MetalPerformanceShaders REQUIRED)

set(LIBS ${FOUNDATION_LIBRARY} ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK} ${METALPERFORMANCE_FRAMEWORK})

configure_file(libmetal.metal libmetal.metal COPYONLY)

add_executable(metaldemo main.cpp libmetal.m libmetal.h)

target_link_libraries(metaldemo ${LIBS})
