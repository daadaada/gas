cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE core_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE core_header "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(dada_src STATIC ${core_src} ${core_header})

target_include_directories(dada_src PUBLIC ${CMAKE_SOURCE_DIR})