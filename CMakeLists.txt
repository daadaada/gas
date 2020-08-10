cmake_minimum_required(VERSION 3.8) # C++17 required

project(gas LANGUAGES CXX VERSION 0.1)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

add_executable(gas main.cpp)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_subdirectory(src)
add_subdirectory(thirdparty)
add_subdirectory(grammar)

target_link_libraries(gas PRIVATE grammar dada_src antlr4_static) 
target_include_directories(gas INTERFACE ${PROJECT_SOURCE_DIR})
