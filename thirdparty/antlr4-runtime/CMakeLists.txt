cmake_minimum_required(VERSION 3.8)

project(antlr4)

set(CXX_STANDARD_REQUIRED ON)
set(CXX_STANDARD 11)

file(GLOB_RECURSE libantlrcpp_src
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(antlr4_static STATIC ${libantlrcpp_src})

set(libantlrcpp_include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/atn
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dfa
  ${CMAKE_CURRENT_SOURCE_DIR}/src/misc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/support
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tree
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tree/pattern
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tree/xpath
)


target_include_directories(antlr4_static PUBLIC ${libantlrcpp_include})

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  target_link_libraries(antlr4_static ${UUID_LIBRARIES})
endif()
